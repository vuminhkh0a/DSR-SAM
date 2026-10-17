"""SOTA runner for y_DG (DG/sota/run_sota.py).

Runs every trainable single-source DG method consecutively and writes one
continuous ``log.txt`` for the whole invocation. This mirrors
``z_SSL/SSL/sota/run_sota.py``, adapted to y_DG's per-method interface:

    DG.<method>.main exposes
        CONFIG                  -> dict of hyper-parameters
        SINGLE_SOURCE_RUNS      -> [(source, [targets...]), ...]   (optional)
        [DATASETS]              -> source domains (used when runs are absent)
        run_single_source(source, targets, cfg[, device], runner)

Nothing in any method's files is modified: the same per-domain
train->test flow runs with each method's own CONFIG, writing to the same
``weights/<method>/`` checkpoints and ``results.json`` entries.

Usage (from ``y_DG/`` with khoa-env active):
    python3 -m DG.sota.run_sota
    python3 -m DG.sota.run_sota --methods unet,maxstyle
    python3 -m DG.sota.run_sota --datasets OTU,OVATUS
    # train on a single source: edit SOURCE_DATASETS (below ALL_DATASETS), e.g. ['OTU']
    # DSR-SAM without / with the L_tsd loss, then SR-SAM:
    python3 -m DG.sota.run_sota --methods dsr_sam_noLtsd,dsr_sam_ltsd,sr_sam

``run.sh`` launches it detached, with every line streamed to ``log.txt``:
    nohup python3 -u -m DG.sota.run_sota > "$LOG_FILE" 2>&1 &

Monitor:  tail -f ~/KhoaVM/y_DG/log.txt
Stop:     pkill -f 'DG[.]sota[.]run_sota'
"""

import argparse
import datetime
import importlib
import inspect
import os
import sys

import torch

from utils.seed import get_generator, set_seed, worker_init_fn
from utils.sequential_training import SequentialDomainRunner, cleanup_resources

# ---------------------------------------------------------------------------
# Logging: one fresh log.txt per invocation. run.sh also redirects stdout to
# this file, so this header + every subsequent print land in the same file.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_FILE = os.path.join(PROJECT_ROOT, 'log.txt')


def reset_log(tag='sota'):
    """Truncate/create log.txt with a startup header (once per process)."""
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    header = f'=== NEW RUN: {tag} | started {ts} ===\n'
    os.makedirs(os.path.dirname(LOG_FILE) or '.', exist_ok=True)
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(header)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    print(header.strip())
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Method registry. Short name -> module + optional CONFIG overrides.
# Order matches run.sh: baseline first, then the SAM-based methods.
# The DSR-SAM ablation/hyperparameter studies are NOT part of this SOTA run;
# they live in DG.dsr_sam.main_a (ablation) and DG.dsr_sam.main_h (HP grid).
#
# `dsr_sam_noLtsd` is DSR-SAM with the L_tsd loss turned off
# (compute_l_tsd=False, so no L_tsd term is added).
#
# `dsr_sam_ltsd` is DSR-SAM with its single L_tsd loss: the
# log-squashed rate = (bb / (a + eps))**2 squashed with
# log(1 + alpha * rate) (no deltaW Frobenius norm in the denominator), enabled
# via compute_l_tsd.
#
# `sr_sam` is the SR-SAM reference method (DG.sr_sam.main).
#
# All three save per-epoch experiment snapshots (experiment_process_save_epochs
# is turned on here via the run_sota overrides: even epochs 2, 4, ..., 20; the
# method CONFIGs keep it off for direct 'python -m DG.<x>.main' runs).
# ---------------------------------------------------------------------------
METHOD_SPECS = [
    # ('unet',             'DG.unet.main',             {}),
    # ('dualnormalization', 'DG.dualnormalization.main', {}),
    # ('mi_segnet',        'DG.mi_segnet.main',        {}),
    # ('maxstyle',         'DG.maxstyle.main',         {}),
    # ('desam',            'DG.desam.main',            {}),
    # ('masam',            'DG.masam.main',            {}),
    # ('sr_sam',           'DG.sr_sam.main',           {}),
    # ('dsr_sam',          'DG.dsr_sam.main',          {}),
    # ('sammed',           'DG.sammed.main',           {}),
    # ('samus',            'DG.samus.main',            {}),
    # ('dapsam',           'DG.dapsam.main',           {}),
    # ('cosam',            'DG.cosam.main',            {}),
    # ('medsa',            'DG.medsa.main',            {}),
    # ('bussam',           'DG.bussam.main',           {}),
    # ('transsam',         'DG.transsam.main',         {}),
    # ('nora',             'DG.nora.main',             {}),
    
    ('dsr_sam_noLtsd',   'DG.dsr_sam.main',
     {'compute_l_tsd': False, 'experiment_process_save_epochs': True,
      'run_name': 'noLtsd',   'result_tag': 'dsr_sam_noLtsd'}),
    ('dsr_sam_ltsd',     'DG.dsr_sam.main',
     {'compute_l_tsd': True, 'experiment_process_save_epochs': True,
      'run_name': 'ltsd',   'result_tag': 'dsr_sam_ltsd'}),
    # ('sr_sam',           'DG.sr_sam.main',
    #  {'experiment_process_save_epochs': True})
]

LOOKUP = {name: (module, overrides)
          for name, module, overrides in METHOD_SPECS}
ALL_METHODS = [name for name, _, _ in METHOD_SPECS]

ALL_DATASETS = [
                'OTU',
                'OVATUS',
                'USOVA',
                ]

# ---------------------------------------------------------------------------
# Which SOURCE domain(s) to TRAIN on in this run. Edit this line to restrict
# the SOTA run. Each source is trained once and tested on the remaining
# domains (leave-one-out), so the targets are automatic:
#   ['OTU']                    -> only OTU source  (test on OVATUS + USOVA)
#   ['OTU', 'OVATUS', 'USOVA'] -> every source (full leave-one-out)
#   None                       -> same as every source
# ---------------------------------------------------------------------------
SOURCE_DATASETS = ['OTU']


def _csv_list(value):
    """Turn 'a,b c' into ['a', 'b', 'c']."""
    return [item.strip() for item in value.replace(',', ' ').split() if item.strip()]


def parse_methods_argv(argv=None):
    """Resolve the method list: CLI --methods > SOTA_METHODS env > all."""
    parser = argparse.ArgumentParser(
        description='Run DG methods consecutively (train->test per source).')
    parser.add_argument(
        '--methods', default=None,
        help='Comma/space-separated subset, e.g. --methods unet,maxstyle. '
             'Defaults to every method in METHOD_SPECS.')
    args, _ = parser.parse_known_args(argv)
    if args.methods:
        selected = _csv_list(args.methods)
    elif os.environ.get('SOTA_METHODS'):
        selected = _csv_list(os.environ['SOTA_METHODS'])
    else:
        selected = list(ALL_METHODS)
    unknown = [m for m in selected if m not in LOOKUP]
    if unknown:
        raise SystemExit(f'Unknown method(s): {unknown}. Choose from: {ALL_METHODS}')
    if not selected:
        raise SystemExit('Empty method list: nothing to run.')
    return list(dict.fromkeys(selected))


def parse_datasets_argv(argv=None):
    """Resolve the dataset list: CLI --datasets > SOTA_DATASETS env > all."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--datasets', default=None)
    args, _ = parser.parse_known_args(argv)
    if args.datasets:
        selected = _csv_list(args.datasets)
    elif os.environ.get('SOTA_DATASETS'):
        selected = _csv_list(os.environ['SOTA_DATASETS'])
    else:
        selected = list(ALL_DATASETS)
    unknown = [d for d in selected if d not in ALL_DATASETS]
    if unknown:
        raise SystemExit(f'Unknown dataset(s): {unknown}. Choose from: {ALL_DATASETS}')
    if not selected:
        raise SystemExit('Empty dataset list: nothing to run.')
    return list(dict.fromkeys(selected))


def _runs_for(module, datasets):
    """Return [(source, [targets...])] for a method module, filtered by datasets."""
    runs = getattr(module, 'SINGLE_SOURCE_RUNS', None)
    if runs is None:
        domains = getattr(module, 'DATASETS', None) or ALL_DATASETS
        runs = [(src, [t for t in domains if t != src]) for src in domains]
    runs = [(src, list(targets)) for src, targets in runs]
    # SOURCE_DATASETS (edit near the top of this file) restricts which SOURCE
    # domains run; each source keeps its full leave-one-out target list.
    if SOURCE_DATASETS is not None:
        runs = [(src, targets) for src, targets in runs if src in SOURCE_DATASETS]
    runs = [(src, [t for t in targets if t in datasets])
            for src, targets in runs if src in datasets]
    return [(src, targets) for src, targets in runs if targets]


def run_method(name, datasets):
    module_name, overrides = LOOKUP[name]
    module = importlib.import_module(module_name)

    cfg = dict(module.CONFIG)
    cfg.update(overrides)

    set_seed(cfg.get('seed', 42))
    device = torch.device(cfg.get('device') or 'cpu')
    runner = SequentialDomainRunner(
        device=device,
        generator=get_generator(cfg.get('seed', 42)),
        worker_init_fn=worker_init_fn,
    )

    wants_device = 'device' in inspect.signature(module.run_single_source).parameters
    runs = _runs_for(module, datasets)
    if not runs:
        print(f'# {name}: no (source, targets) runs for datasets={datasets}; skipping.')
        sys.stdout.flush()
        return

    for source, targets in runs:
        cleanup_resources(device)
        if wants_device:
            module.run_single_source(source, targets, cfg, device, runner)
        else:
            module.run_single_source(source, targets, cfg, runner)


def main(argv=None):
    methods = parse_methods_argv(argv)
    datasets = parse_datasets_argv(argv)
    reset_log('sota')
    print(f'\n{"#" * 70}')
    print(f'# SOTA run: {len(methods)} method(s) consecutively')
    print(f'# methods : {methods}')
    print(f'# datasets: {datasets}')
    print(f'# sources : {SOURCE_DATASETS if SOURCE_DATASETS is not None else "all"}')
    print(f'# log     : {LOG_FILE}')
    print(f'# {"#" * 68}')
    sys.stdout.flush()

    for i, name in enumerate(methods, 1):
        print(f'\n{"#" * 70}')
        print(f'# METHOD {i}/{len(methods)}: {name} [{LOOKUP[name][0]}]')
        print(f'{"#" * 70}')
        sys.stdout.flush()
        run_method(name, datasets)
        print(f'\n# METHOD {name} DONE')
        sys.stdout.flush()

    print(f'\n{"#" * 70}')
    print('# SOTA run complete.')
    print(f'{"#" * 70}')
    sys.stdout.flush()


if __name__ == '__main__':
    main()
