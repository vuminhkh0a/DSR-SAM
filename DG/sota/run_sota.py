"""SOTA runner for y_DG (DG/sota/run_sota.py).

Runs every benchmark DG method except DSR-SAM and DSR-SAM2, with seeds
42, 0 and 100.  Every method config is given
``prompt_method='yolov8'`` and ``prompt_yolo_size='x'`` so the box-prompt
methods read ``box_coords_yolov8_x.json``.  Methods that do not consume box
prompts ignore those keys.

Results are written to ``results.json`` with a
``_yolov8x_seed<seed>`` suffix, e.g.
``gpu1_vit_b_sr_sam_s_OTU_t_OVATUS_best_yolov8x_seed42``.

Usage (from ``y_DG/`` with khoa-env active):
    python3 -m DG.sota.run_sota
    python3 -m DG.sota.run_sota --methods unet,maxstyle
    python3 -m DG.sota.run_sota --datasets OTU,OVATUS
    python3 -m DG.sota.run_sota --datasets OTU

``run.sh`` launches it detached, with every line streamed to ``log.txt``:
    nohup python3 -u -m DG.sota.run_sota > "$LOG_FILE" 2>&1 &

Monitor:  tail -f ~/KhoaVM/y_DG/log.txt
Stop:     pkill -f 'DG[.]sota[.]run_sota'
"""

import argparse
import datetime
import glob
import importlib
import inspect
import os
import sys
import time
import traceback

import torch

from utils.seed import get_generator, set_seed, worker_init_fn
from utils.sequential_training import SequentialDomainRunner, cleanup_resources
import utils.metrics as metrics_utils

# ---------------------------------------------------------------------------
# One fresh log.txt per invocation. run.sh also redirects stdout to this file,
# so this header + every subsequent print land in the same file.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_FILE = os.path.join(PROJECT_ROOT, 'log.txt')

# Fixed benchmark protocol for this runner.
SEEDS = (42, 0, 100)
PROMPT_METHOD = 'yolov8'
PROMPT_YOLO_SIZE = 'x'
PROMPT_TAG = 'yolov8x'


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
        sys.stdout.flush()
        try:
            fd = sys.stdout.fileno()
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
        except (OSError, AttributeError, ValueError):
            pass
    else:
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
# Method registry. Every benchmark method except DSR-SAM and DSR-SAM2.
# Trainable methods expose CONFIG + run_single_source(); inference-only
# methods expose CONFIG + run_testonly().
# ---------------------------------------------------------------------------
METHOD_SPECS = [
    # unet, DN, MI-SegNet, MaxStyle, DAP-SAM and NORA are removed from the
    # benchmark; their code stays in DG/ but they are disabled here.
    # ('unet', 'DG.unet.main', {}),
    # ('dualnormalization', 'DG.dualnormalization.main', {}),
    # ('mi_segnet', 'DG.mi_segnet.main', {}),
    # ('maxstyle', 'DG.maxstyle.main', {}),
    ('desam', 'DG.desam.main', {}),
    ('masam', 'DG.masam.main', {}),

    # dsr_sam and dsr_sam2 are intentionally excluded.
    ('sammed', 'DG.sammed.main', {}),
    ('samus', 'DG.samus.main', {}),
    # ('dapsam', 'DG.dapsam.main', {}),
    ('cosam', 'DG.cosam.main', {}),
    ('medsa', 'DG.medsa.main', {}),
    ('bussam', 'DG.bussam.main', {}),
    ('transsam', 'DG.transsam.main', {}),
    ('sr_sam', 'DG.sr_sam.main', {}),
    # ('nora', 'DG.nora.main', {}),
    # Inference-only official checkpoints.
    ('medsam', 'DG.medsam.main', {}),
    ('sammed2d', 'DG.sammed2d.main', {}),
    ('ultrasam', 'DG.ultrasam.main', {}),
]

LOOKUP = {name: (module, overrides)
          for name, module, overrides in METHOD_SPECS}
ALL_METHODS = [name for name, _, _ in METHOD_SPECS]

ALL_DATASETS = [
    'OTU',
    'OVATUS',
    'USOVA',
]

# Optional external restriction on source domains. The CLI --datasets /
# SOTA_DATASETS path below is the normal way to select sources.
SOURCE_DATASETS = None

# Weight directories for trainable methods. Most use weights/<method>; the
# dual-normalization baseline uses weights/dn.
WEIGHT_DIR_OVERRIDES = {
    'dualnormalization': 'weights/dn',
}


def _weight_dir(method):
    return WEIGHT_DIR_OVERRIDES.get(method, os.path.join('weights', method))


def _expected_best_path(method, cfg, source):
    """Path of the main per-source best checkpoint for a trainable method."""
    model_type = cfg.get('model_type', '')
    if method == 'unet':
        prefix = f'unet_s_{source}'
    elif method == 'dualnormalization':
        prefix = f'dn_s_{source}'
    elif method == 'mi_segnet':
        prefix = f'mi_segnet_s_{source}'
    elif method == 'maxstyle':
        prefix = f'maxstyle_s_{source}'
    else:
        prefix = f'{model_type}_{method}_s_{source}'
    return os.path.join(_weight_dir(method), f'{prefix}_best.pth')


def _clean_known_extras(method, sources):
    """Remove old _last / _experiment outputs for the selected sources.

    The per-source *_best.pth files are overwritten by the seed loop.  This
    cleanup only removes the known extra outputs, not the best checkpoints.
    """
    weight_dir = _weight_dir(method)
    removed = []
    for source in sources:
        for pattern in (f'*_s_{source}_last.pth',
                        f'*_s_{source}_experiment_epoch_*.pth'):
            for path in glob.glob(os.path.join(weight_dir, pattern)):
                os.remove(path)
                removed.append(path)
    if removed:
        print(f'# {method}: cleaned {len(removed)} old extra weight file(s)')
        for path in removed:
            print(f'#   - {path}')


def _verify_weights(method, cfg, sources, last_seed_start):
    """Check the expected 3 best files exist and were written by the last seed."""
    paths = [_expected_best_path(method, cfg, src) for src in sources]
    missing = [p for p in paths if not os.path.exists(p)]
    stale = [p for p in paths
             if os.path.exists(p) and os.path.getmtime(p) < last_seed_start]
    extras = []
    for source in sources:
        for pattern in (f'*_s_{source}_last.pth',
                        f'*_s_{source}_experiment_epoch_*.pth'):
            extras.extend(glob.glob(os.path.join(_weight_dir(method), pattern)))

    print(f'# {method}: weight check')
    for path in paths:
        print(f'#   best: {path}')
    if missing:
        print(f'#   MISSING: {missing}')
    if stale:
        print(f'#   NOT WRITTEN BY SEED {SEEDS[-1]}: {stale}')
    if extras:
        print(f'#   EXTRA last/experiment files: {extras}')
    if missing or stale or extras:
        raise RuntimeError(
            f'{method}: weight check failed (missing={missing}, '
            f'stale={stale}, extras={extras})')
    print(f'#   OK: {len(paths)} best file(s), last writer seed {SEEDS[-1]}')


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
    ``datasets`` selects which SOURCE domains to train. Each selected source
    keeps the full target list declared in ``SINGLE_SOURCE_RUNS``.
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

    base_cfg = dict(module.CONFIG)
    base_cfg.update(overrides)
    # Fixed protocol for this runner: YOLOv8x prompt boxes.
    base_cfg['prompt_method'] = PROMPT_METHOD
    base_cfg['prompt_yolo_size'] = PROMPT_YOLO_SIZE
    # One best checkpoint per source; no _last / experiment checkpoints.
    base_cfg['save_last_epoch'] = False
    base_cfg['experiment_process_save_epochs'] = False
    device = torch.device(base_cfg.get('device') or 'cpu')

    if not is_trainable:
        targets = [d for d in datasets
                   if d in getattr(module, 'DATASETS', ALL_DATASETS)]
        if SOURCE_DATASETS is not None:
            targets = [d for d in targets if d in SOURCE_DATASETS]
        if not targets:
            print(f'# {name}: no test domains for datasets={datasets}; skipping.')
            sys.stdout.flush()
            return
    else:
        runs = _runs_for(module, datasets)
        if not runs:
            print(f'# {name}: no (source, targets) runs for datasets={datasets}; skipping.')
            sys.stdout.flush()
            return
        wants_device = 'device' in inspect.signature(
            module.run_single_source).parameters
        sources = list(dict.fromkeys(src for src, _ in runs))
        _clean_known_extras(name, sources)

    try:
        seed_start = {}
        for seed in SEEDS:
            seed_start[seed] = time.time()
            cfg = dict(base_cfg)
            cfg['seed'] = seed
            set_seed(seed)
            metrics_utils.RESULT_SUFFIX = f'_{PROMPT_TAG}_seed{seed}'
            runner = SequentialDomainRunner(
                device=device,
                generator=get_generator(seed),
                worker_init_fn=worker_init_fn,
            )

            if not is_trainable:
                print(f'# {name}: seed={seed} test-only -> '
                      f'{", ".join(targets)} (prompt={PROMPT_TAG})')
                sys.stdout.flush()
                cleanup_resources(device)
                module.run_testonly(targets, cfg, runner)
                continue

            print(f'# {name}: seed={seed} {len(runs)} run(s) -> ' +
                  ', '.join(f'{src}->{"+".join(targets)}'
                            for src, targets in runs) +
                  f' (prompt={PROMPT_TAG})')
            sys.stdout.flush()
            for source, targets in runs:
                cleanup_resources(device)
                if wants_device:
                    module.run_single_source(source, targets, cfg, device, runner)
                else:
                    module.run_single_source(source, targets, cfg, runner)

        if is_trainable:
            _verify_weights(name, base_cfg, sources, seed_start[SEEDS[-1]])
        else:
            print(f'# {name}: zero-shot; no weight files written.')
    finally:
        metrics_utils.RESULT_SUFFIX = ''


def main(argv=None):
    methods = parse_methods_argv(argv)
    datasets = parse_datasets_argv(argv)
    metrics_utils.RESULT_SUFFIX = ''
    reset_log('sota')
    print(f'\n{"#" * 70}')
    print(f'# SOTA run: {len(methods)} method(s) consecutively')
    print(f'# methods : {methods}')
    print(f'# sources : {datasets} (train each source, test the held-out targets)')
    print(f'# seeds   : {list(SEEDS)}')
    print(f'# prompt  : {PROMPT_TAG} ({PROMPT_METHOD}, size {PROMPT_YOLO_SIZE})')
    print(f'# prompt file: box_coords_yolov8_{PROMPT_YOLO_SIZE}.json')
    print(f'# results : results.json with suffix _{PROMPT_TAG}_seed<seed>')
    print(f'# weights : one <prefix>_best.pth per source; seeds 42->0->100 overwrite')
    print(f'#           final weight files are seed {SEEDS[-1]}; zero-shot: none')
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
