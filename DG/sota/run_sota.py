"""SOTA runner for y_DG (DG/sota/run_sota.py).

Runs every benchmark DG method consecutively and writes one continuous
``log.txt`` for the whole invocation. This mirrors
``z_SSL/SSL/sota/run_sota.py``, adapted to y_DG's per-method interface:

    DG.<method>.main exposes
        CONFIG                  -> dict of hyper-parameters
        SINGLE_SOURCE_RUNS      -> [(source, [targets...]), ...]   (optional)
        [DATASETS]              -> domains (fallback split / test-only targets)
        run_single_source(source, targets, cfg[, device], runner)   # trainable
        run_testonly(targets, cfg, runner)                          # inference-only

Nothing in any method's files is modified: the same per-domain
train->test flow runs with each method's own CONFIG, writing to the same
``weights/<method>/`` checkpoints and ``results.json`` entries.

A method that raises is logged with its full traceback and skipped; the
remaining methods still run (a summary of failures is printed at the end).

Usage (from ``y_DG/`` with khoa-env active):
    python3 -m DG.sota.run_sota
    python3 -m DG.sota.run_sota --methods unet,maxstyle
    python3 -m DG.sota.run_sota --datasets OTU,OVATUS   # source domains only
    # train on a single source (tests both held-out targets):
    python3 -m DG.sota.run_sota --datasets OTU
    # DSR-SAM2 CKA ablation, then SR-SAM:
    python3 -m DG.sota.run_sota --methods dsr_sam2_ab_full_seed42,sr_sam

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
import traceback

import torch

from utils.seed import get_generator, set_seed, worker_init_fn
from utils.sequential_training import SequentialDomainRunner, cleanup_resources

# ---------------------------------------------------------------------------
# Logging: one fresh log.txt per invocation. run.sh also redirects stdout to
# this file, so this header + every subsequent print land in the same file.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_FILE = os.path.join(PROJECT_ROOT, 'log.txt')


def _stdout_is_log_file():
    """True when stdout already points at LOG_FILE (run.sh ``> log.txt``)."""
    try:
        out = os.fstat(sys.stdout.fileno())
        log = os.stat(LOG_FILE)
    except (OSError, AttributeError, ValueError):
        return False
    return (out.st_dev, out.st_ino) == (log.st_dev, log.st_ino)


def reset_log(tag='sota'):
    """Truncate/create log.txt with a startup header (once per process)."""
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    header = f'=== NEW RUN: {tag} | started {ts} ===\n'
    os.makedirs(os.path.dirname(LOG_FILE) or '.', exist_ok=True)
    if not os.path.exists(LOG_FILE):
        open(LOG_FILE, 'a', encoding='utf-8').close()

    if _stdout_is_log_file():
        # run.sh redirects stdout straight to LOG_FILE. Truncate and rewind
        # *that* fd, then let stdout carry the header. A separate file handle
        # here would race with the redirected stdout, garbling the first line
        # ("=== NEW RUN ... === NEW RUN") or leaving a NUL hole when anything
        # was printed before reset_log().
        sys.stdout.flush()
        try:
            fd = sys.stdout.fileno()
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
        except (OSError, AttributeError, ValueError):
            pass
    else:
        # Not redirected: reset the on-disk log with the header; the rest of
        # this process's output still goes to the terminal.
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
# Every benchmark method is listed. The DSR-SAM2 ablation study
# (CNN branch / CBA / AttentionGate / CKA fusion / L_CKA) is registered at
# the end of the list; the remaining hyper-parameter studies stay in
# DG.dsr_sam.main_a / main_h and DG.dsr_sam2.main_a / main_h.
#
# Trainable methods expose CONFIG + run_single_source() and are driven over
# their SINGLE_SOURCE_RUNS (3 sources x 2 held-out targets).
# Inference-only methods expose CONFIG + run_testonly() and are evaluated
# once on every benchmark domain.
# ---------------------------------------------------------------------------

# ---- DSR-SAM2 ablation study: CNN / CBA / AttentionGate / CKA fusion / L_CKA
# Unique cases only (duplicate combinations pruned).  Full/on CBA uses
# cba_last_n=8; CBA-off and no-CNN cases use cba_last_n=0.
# Fixed for every variant:
#   cnn_pretrained=True
#   attention_gate_channels=768
#   cka_loss_weight=0.1
# CKA and gate parameters are always differentiable.
# Fusion behavior:
#   gate=True : f_vit + ((1-C_weight)*f_cnn + f_gated_cnn) / 2
#   gate=False: f_vit + (1-C_weight)*f_cnn
#   cka_fusion=False: f_vit + f_cnn (gate bypassed)
# All variants share weight_name_tag='dsr_sam2_cka_ablation' to save disk;
# each seed keeps its own result_name_tag.  Every variant runs with seeds
# 42 (default), 0 and 100.
DSR_SAM2_ABLATIONS = (
    # tag, cnn_branch, cba_last_n, gate, cka_fusion, lcka
    ('no_cnn', False, 0, False, False, False),
    ('no_cba', True, 0, False, False, False),
    ('full', True, 8, True, True, True),
    ('no_lcka', True, 8, True, True, False),
    ('no_gate', True, 8, False, True, True),
    ('no_lcka_no_gate', True, 8, False, True, False),
    ('no_cka_channel_fusion', True, 8, False, False, True),
    ('no_lcka_no_cka_channel_fusion', True, 8, False, False, False),
)
DSR_SAM2_ABLATION_SEEDS = (0, 100)
DSR_SAM2_ABLATION_FIXED = {
    'cnn_pretrained': True,
    'attention_gate_channels': 768,
    'cka_loss_weight': 0.1,
    'weight_name_tag': 'dsr_sam2_cka_ablation',
}

METHOD_SPECS = [
    # ('unet',             'DG.unet.main',             {}),
    # ('dualnormalization', 'DG.dualnormalization.main', {}),
    # ('mi_segnet',        'DG.mi_segnet.main',        {}),
    # ('maxstyle',         'DG.maxstyle.main',         {}),
    # ('desam',            'DG.desam.main',            {}),
    # ('masam',            'DG.masam.main',            {}),
    # ('sr_sam',           'DG.sr_sam.main',           {}),
    # ('dsr_sam2',         'DG.dsr_sam2.main',         {}),
    # ('sammed',           'DG.sammed.main',           {}),
    # ('samus',            'DG.samus.main',            {}),
    # ('dapsam',           'DG.dapsam.main',           {}),
    # ('cosam',            'DG.cosam.main',            {}),
    # ('medsa',            'DG.medsa.main',            {}),
    # ('bussam',           'DG.bussam.main',           {}),
    # ('transsam',         'DG.transsam.main',         {}),
    # ('nora',             'DG.nora.main',             {}),
    # ---- inference-only official checkpoints ----
    # ('medsam',           'DG.medsam.main',           {}),
    # ('sammed2d',         'DG.sammed2d.main',         {}),
    # ('ultrasam',         'DG.ultrasam.main',         {}),
] + [
    (f'dsr_sam2_ab_{tag}_seed{seed}', 'DG.dsr_sam2.main', {
        **DSR_SAM2_ABLATION_FIXED,
        'seed': seed,
        'cnn_branch': cnn_branch,
        'cba_last_n': cba_last_n,
        'use_attention_gate': gate,
        'use_attention_gate_cba': gate,
        'cka_fusion': cka_fusion,
        'compute_cka_loss': lcka,
        'result_name_tag': f'dsr_sam2_ab_{tag}_seed{seed}',
    })
    for tag, cnn_branch, cba_last_n, gate, cka_fusion, lcka in DSR_SAM2_ABLATIONS
    for seed in DSR_SAM2_ABLATION_SEEDS
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
# Which SOURCE domain(s) to TRAIN on in this run. Each source is trained once
# and tested on the remaining domains (leave-one-out), so the targets are
# automatic:
#   None                       -> every source (full leave-one-out, 3x2 cases)
#   ['OTU']                    -> only OTU source  (test on OVATUS + USOVA)
#   ['OTU', 'OVATUS', 'USOVA'] -> same as None
# Per-source CLI restriction: --datasets OTU (or SOTA_DATASETS=OTU).
# ---------------------------------------------------------------------------
SOURCE_DATASETS = None


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
    """Return [(source, [targets...])] for a method module.

    Methods declare their runs as ``SINGLE_SOURCE_RUNS`` (preferred); modules
    that only define ``DATASETS`` fall back to a full leave-one-out split.
    ``datasets`` (CLI ``--datasets`` / ``SOTA_DATASETS``) and ``SOURCE_DATASETS``
    select which SOURCE domains to train. Each selected source keeps the full
    target list declared in ``SINGLE_SOURCE_RUNS`` -- e.g. ``--datasets OTU``
    trains OTU and still tests on both OVATUS and USOVA.
    """
    runs = getattr(module, 'SINGLE_SOURCE_RUNS', None)
    if runs is None:
        domains = getattr(module, 'DATASETS', None) or ALL_DATASETS
        runs = [(src, [t for t in domains if t != src]) for src in domains]
    runs = [(src, list(dict.fromkeys(targets))) for src, targets in runs]
    sources = set(datasets)
    if SOURCE_DATASETS is not None:
        sources &= set(SOURCE_DATASETS)
    return [(src, targets) for src, targets in runs
            if src in sources and targets]


def run_method(name, datasets):
    module_name, overrides = LOOKUP[name]
    module = importlib.import_module(module_name)

    is_trainable = hasattr(module, 'run_single_source')
    is_testonly = hasattr(module, 'run_testonly')
    if not (is_trainable or is_testonly):
        raise RuntimeError(
            f'{name} ({module_name}) exposes neither run_single_source() nor '
            f'run_testonly(); remove it from METHOD_SPECS or run it directly.')

    cfg = dict(module.CONFIG)
    cfg.update(overrides)

    set_seed(cfg.get('seed', int(os.environ.get('SEED', 42))))
    device = torch.device(cfg.get('device') or 'cpu')
    runner = SequentialDomainRunner(
        device=device,
        generator=get_generator(cfg.get('seed', int(os.environ.get('SEED', 42)))),
        worker_init_fn=worker_init_fn,
    )

    # Inference-only official checkpoints: one evaluation over every selected
    # benchmark domain (no source/target split, no training).
    if not is_trainable:
        targets = [d for d in datasets if d in getattr(module, 'DATASETS', ALL_DATASETS)]
        if SOURCE_DATASETS is not None:
            targets = [d for d in targets if d in SOURCE_DATASETS]
        if not targets:
            print(f'# {name}: no test domains for datasets={datasets}; skipping.')
            sys.stdout.flush()
            return
        print(f'# {name}: test-only -> {", ".join(targets)}')
        sys.stdout.flush()
        cleanup_resources(device)
        module.run_testonly(targets, cfg, runner)
        return

    wants_device = 'device' in inspect.signature(module.run_single_source).parameters
    runs = _runs_for(module, datasets)
    if not runs:
        print(f'# {name}: no (source, targets) runs for datasets={datasets}; skipping.')
        sys.stdout.flush()
        return

    print(f'# {name}: {len(runs)} run(s) -> ' +
          ', '.join(f'{src}->{"+".join(targets)}' for src, targets in runs))
    sys.stdout.flush()

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
    print(f'# sources : {datasets} (train each source, test the held-out targets)')
    print(f'# from    : {SOURCE_DATASETS if SOURCE_DATASETS is not None else "all sources allowed"}')
    print(f'# log     : {LOG_FILE}')
    print(f'# {"#" * 68}')
    sys.stdout.flush()

    failed = []
    for i, name in enumerate(methods, 1):
        print(f'\n{"#" * 70}')
        print(f'# METHOD {i}/{len(methods)}: {name} [{LOOKUP[name][0]}]')
        print(f'{"#" * 70}')
        sys.stdout.flush()
        try:
            run_method(name, datasets)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            # Skip the failing method and keep the SOTA sweep going. The full
            # traceback is written to log.txt for debugging.
            failed.append((name, f'{type(exc).__name__}: {exc}'))
            print(f'\n# METHOD {name} FAILED: {type(exc).__name__}: {exc}')
            traceback.print_exc()
            print(f'# Skipping {name}; continuing with the next method.')
            sys.stdout.flush()
            try:
                cleanup_resources(
                    torch.device('cuda') if torch.cuda.is_available() else None)
            except Exception:
                pass
            continue
        print(f'\n# METHOD {name} DONE')
        sys.stdout.flush()

    print(f'\n{"#" * 70}')
    if failed:
        print(f'# SOTA run finished with {len(failed)} failed method(s):')
        for name, err in failed:
            print(f'#   - {name}: {err}')
    else:
        print('# SOTA run complete. All methods finished.')
    print(f'{"#" * 70}')
    sys.stdout.flush()


if __name__ == '__main__':
    main()
