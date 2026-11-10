"""SOTA runner for y_DG (DG/sota/run_sota.py).

Active configuration: the DSR-SAM2 ablation study (see ``ACTIVE_STUDY``).
The regular METHOD_SPECS benchmark entries below are commented out, so
``python3 -m DG.sota.run_sota`` launches only the study.

DSR-SAM2 ablation:
  * 8 variants: Full, No L_CKA, No gate, No L_CKA + no gate,
    No CKA fusion, No L_CKA + no CKA fusion, No CBA, No CNN.
  * all three sources OTU, OVATUS, USOVA, each tested on its two held-out
    targets (single-source DG).
  * seeds 42, 0, 100 by default; every source/variant/seed overwrites ONE
    checkpoint at
    ``weights/dsr_sam2/vit_b_dsr_sam2_ablation_overwrite_best.pth``.
  * box prompts are selected by ``SOTA_PROMPT_METHOD`` / ``SOTA_PROMPT_TAG``:
    ``gt`` -> manual boxes from ``box_coords.json`` (tag ``manual``),
    ``yolov8`` -> ``box_coords_yolov8_<size>.json`` (tag ``yolov8x``).
  * results appended to ``results.json`` as
    ``vit_b_dsr_sam2_ab2_<tag>_sam_s_<source>_t_<target>_best_<prompt>_seed<seed>``.
  * for one process per seed, set ``SOTA_SEEDS``, ``SOTA_LOG_FILE``,
    ``SOTA_RESULTS_FILE``, ``SOTA_WEIGHT_PREFIX`` and ``SOTA_WEIGHT_SUFFIX``
    (see ``run.sh``); each process then writes its own log, results file and
    checkpoint.

Usage (from ``y_DG/`` with khoa-env active):
    python3 -m DG.sota.run_sota                 # run all 3 sources
    python3 -m DG.sota.run_sota --dry-run       # print the 72-run plan only
    DSR_SAM_DATASETS=OTU python3 -m DG.sota.run_sota      # one source only
    SOTA_SEEDS=42 SOTA_WEIGHT_SUFFIX=_seed42 SOTA_LOG_FILE=log_seed42.txt \\
        SOTA_RESULTS_FILE=results_seed42.json python3 -m DG.sota.run_sota
    python3 -m DG.sota.run_sota --study none    # use METHOD_SPECS instead
    python3 -m DG.sota.run_sota --study none --methods sr_sam --datasets OTU

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
# One fresh log per invocation. run.sh also redirects stdout to this file,
# so this header + every subsequent print land in the same file.
#
# Parallel-seed support (all optional, default = old single-process behavior):
#   SOTA_SEEDS          e.g. "42" or "42,0,100"  (default 42,0,100)
#   SOTA_LOG_FILE       log path                (default <root>/log.txt)
#   SOTA_RESULTS_FILE   results path            (default results.json)
#   SOTA_PROMPT_METHOD  'gt' (manual boxes) or 'yolov8' (default 'yolov8')
#   SOTA_PROMPT_YOLO_SIZE  YOLOv8 size for 'yolov8' prompts (default 'x')
#   SOTA_PROMPT_TAG     suffix tag in result names (default 'yolov8x')
#   SOTA_WEIGHT_PREFIX  checkpoint name prefix  (default dsr_sam2_ablation_overwrite)
#   SOTA_WEIGHT_SUFFIX  checkpoint tag suffix   (default none)
#   SOTA_WEIGHT_DIR     checkpoint directory    (default weights/dsr_sam2)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _project_path(value):
    """Resolve a relative path against the project root."""
    return value if os.path.isabs(value) else os.path.join(PROJECT_ROOT, value)


LOG_FILE = _project_path(os.environ.get('SOTA_LOG_FILE', '').strip()
                         or 'log.txt')

# Per-process results file (keeps parallel seed runs from racing on one JSON).
if os.environ.get('SOTA_RESULTS_FILE', '').strip():
    metrics_utils.RESULT_FILE = _project_path(
        os.environ['SOTA_RESULTS_FILE'].strip())


def _parse_seeds(raw):
    return tuple(int(part) for part in raw.replace(',', ' ').split())


# Fixed benchmark protocol for this runner. SOTA_SEEDS lets run.sh launch
# one process per seed; omit it to keep the original 42 -> 0 -> 100 loop.
# Prompt source is configurable so the same runner can use manual boxes
# (box_coords.json) or YOLOv8-predicted boxes without code changes.
SEEDS = _parse_seeds(os.environ['SOTA_SEEDS']) if os.environ.get('SOTA_SEEDS') \
    else (42, 0, 100)
PROMPT_METHOD = (os.environ.get('SOTA_PROMPT_METHOD', 'yolov8').strip()
                 or 'yolov8')
PROMPT_YOLO_SIZE = (os.environ.get('SOTA_PROMPT_YOLO_SIZE', 'x').strip()
                    or 'x')
PROMPT_TAG = (os.environ.get('SOTA_PROMPT_TAG', 'yolov8x').strip()
              or 'yolov8x')
PROMPT_FILE = ('box_coords.json' if PROMPT_METHOD.lower() in ('gt', 'manual')
               else f'box_coords_yolov8_{PROMPT_YOLO_SIZE}.json')


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
# The benchmark methods are commented out while the DSR-SAM2 ablation study
# is the active configuration. Uncomment the entries and run with
# ``--study none`` to use them again.
METHOD_SPECS = [
    # ('desam', 'DG.desam.main', {}),
    # ('masam', 'DG.masam.main', {}),
    # ('sammed', 'DG.sammed.main', {}),
    # ('samus', 'DG.samus.main', {}),
    # ('cosam', 'DG.cosam.main', {}),
    # ('medsa', 'DG.medsa.main', {}),
    # ('bussam', 'DG.bussam.main', {}),
    # ('transsam', 'DG.transsam.main', {}),
    # ('sr_sam', 'DG.sr_sam.main', {}),
    # Inference-only official checkpoints.
    # ('medsam', 'DG.medsam.main', {}),
    # ('sammed2d', 'DG.sammed2d.main', {}),
    # ('ultrasam', 'DG.ultrasam.main', {}),
    # Removed methods (code kept in DG/): unet, dualnormalization,
    # mi_segnet, maxstyle, dapsam, nora.
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

# ---------------------------------------------------------------------------
# Active study: DSR-SAM2 ablation (8 variants x 3 seeds x 3 sources).
# This ab2 study re-runs all sources with the new global CKA.
# Set ACTIVE_STUDY = None to use METHOD_SPECS again.
# ---------------------------------------------------------------------------
ACTIVE_STUDY = 'dsr_sam2_ablation'

DSR_SAM2_ABLATION_SOURCES = ('OTU', 'OVATUS', 'USOVA')
DSR_SAM2_ABLATION_COMPLETED = ()  # all sources re-run with the new CKA
DSR_SAM2_ABLATION_TARGETS_FOR = {
    'OTU': ('OVATUS', 'USOVA'),
    'OVATUS': ('OTU', 'USOVA'),
    'USOVA': ('OTU', 'OVATUS'),
}
DSR_SAM2_ABLATION_OVERWRITE = 'dsr_sam2_ablation_overwrite'
# (tag, cnn_branch, cba_last_n, gate, cka_fusion, lcka) for the ab2 study.
# tools/make_ablation_table.py still describes the legacy ab grid.
DSR_SAM2_ABLATIONS = (
    ('full', True, 8, True, True, True),
    ('no_lcka', True, 8, True, True, False),
    ('no_gate', True, 8, False, True, True),
    ('no_lcka_no_gate', True, 8, False, True, False),
    ('no_cka_channel_fusion', True, 8, True, False, False),
    ('no_lcka_no_cka_channel_fusion', True, 8, False, False, False),
    ('no_cba', True, 0, False, False, False),
    ('no_cnn', False, 0, False, False, False),
)

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


def parse_study_argv(argv=None):
    """Resolve the active study (--study > ACTIVE_STUDY) and --dry-run."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--study', default=ACTIVE_STUDY,
                        choices=('dsr_sam2_ablation', 'none'))
    parser.add_argument('--dry-run', action='store_true')
    args, _ = parser.parse_known_args(argv)
    return args


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


def run_dsr_sam2_ablation_study(dry_run=False):
    """Run the 8-variant DSR-SAM2 ab2 ablation on all three sources."""
    module = importlib.import_module('DG.dsr_sam2.main')
    base_cfg = dict(module.CONFIG)
    # One checkpoint per seed process: suffix the overwrite name instead of
    # sharing vit_b_dsr_sam2_ablation_overwrite_best.pth. The prefix can be
    # overridden so prompt-specific runs get distinct checkpoint names.
    weight_prefix = (os.environ.get('SOTA_WEIGHT_PREFIX', '').strip()
                     or DSR_SAM2_ABLATION_OVERWRITE)
    weight_suffix = os.environ.get('SOTA_WEIGHT_SUFFIX', '').strip()
    overwrite_name = f'{weight_prefix}{weight_suffix}'
    overwrite_dir = os.environ.get('SOTA_WEIGHT_DIR', '').strip() \
        or os.path.join('weights', 'dsr_sam2')
    base_cfg.update({
        'prompt_method': PROMPT_METHOD,
        'prompt_yolo_size': PROMPT_YOLO_SIZE,
        'save_last_epoch': False,
        'experiment_process_save_epochs': False,
        'overwrite_name': overwrite_name,
        'overwrite_dir': overwrite_dir,
    })
    if os.environ.get('DSR_SAM_EPOCHS'):
        base_cfg['n_epochs'] = int(os.environ['DSR_SAM_EPOCHS'])
    device = torch.device(base_cfg.get('device') or 'cpu')

    # Default: all three source domains. Override with e.g.
    # DSR_SAM_DATASETS=OTU or DSR_SAM_DATASETS=OTU,OVATUS,USOVA.
    if os.environ.get('DSR_SAM_DATASETS'):
        sources = [s.strip().upper() for s in
                   os.environ['DSR_SAM_DATASETS'].replace(',', ' ').split()
                   if s.strip()]
        unknown = [s for s in sources
                   if s not in DSR_SAM2_ABLATION_TARGETS_FOR]
        if unknown:
            raise SystemExit(
                f'Unknown source(s) in DSR_SAM_DATASETS: {unknown}; '
                f'choose from {list(DSR_SAM2_ABLATION_TARGETS_FOR)}')
    else:
        sources = list(DSR_SAM2_ABLATION_SOURCES)
    skipped = [s for s in sources if s in DSR_SAM2_ABLATION_COMPLETED]
    if skipped:
        print(f'# skipped completed source(s): {", ".join(skipped)}')
    sources = [s for s in sources if s not in DSR_SAM2_ABLATION_COMPLETED]
    if not sources:
        raise SystemExit(
            'No source left to run after skipping completed source(s): '
            f'{list(DSR_SAM2_ABLATION_COMPLETED)}')
    runs = [(source, list(DSR_SAM2_ABLATION_TARGETS_FOR[source]))
            for source in sources]
    n_trainings = len(runs) * len(DSR_SAM2_ABLATIONS) * len(SEEDS)
    sources_desc = '; '.join(
        f'{source}->{"+".join(targets)}' for source, targets in runs)

    weight_path = os.path.join(
        overwrite_dir,
        f'{base_cfg["model_type"]}_{overwrite_name}_best.pth')

    print(f'\n{"#" * 70}')
    print('# DSR-SAM2 ABLATION STUDY')
    print(f'# sources  : {sources_desc}')
    print(f'# trainings: {len(runs)} source(s) x {len(DSR_SAM2_ABLATIONS)} '
          f'variants x {len(SEEDS)} seeds = {n_trainings}')
    print(f'# prompt   : {PROMPT_TAG} ({PROMPT_METHOD}, file {PROMPT_FILE})')
    print(f'# config   : rank={base_cfg["rank"]} '
          f'cba_last_n={base_cfg["cba_last_n"]} '
          f'cka_loss_weight={base_cfg["cka_loss_weight"]} '
          f'epochs={base_cfg["n_epochs"]} patience={base_cfg["patience"]}')
    print(f'# weight   : {weight_path} (ONE file per process, overwritten)')
    print(f'# results  : {metrics_utils.RESULT_FILE}, '
          f'vit_b_dsr_sam2_ab2_<tag>_sam_s_<source>_t_<target>_best'
          f'_{PROMPT_TAG}_seed<seed>')
    print(f'# log      : {LOG_FILE}')
    print(f'# {"#" * 68}')
    sys.stdout.flush()

    if dry_run:
        for seed in SEEDS:
            for source, targets in runs:
                for tag, cnn_branch, cba_n, gate, cka_fusion, lcka in DSR_SAM2_ABLATIONS:
                    print(f'#   seed={seed} '
                          f'{source}->{"+".join(targets):12s} {tag:32s} '
                          f'cnn={int(cnn_branch)} n={cba_n} gate={int(gate)} '
                          f'cka={int(cka_fusion)} lcka={int(lcka)}')
        print('# DRY RUN: nothing executed.')
        return

    os.makedirs(overwrite_dir, exist_ok=True)
    # Only remove this process's own stale checkpoint; other seed processes
    # use a different suffix and must not be touched.
    stale_prefix = f'{base_cfg["model_type"]}_{overwrite_name}'
    for name in os.listdir(overwrite_dir):
        if name.startswith(stale_prefix) and name.endswith('.pth'):
            os.remove(os.path.join(overwrite_dir, name))

    failed = []
    try:
        for seed in SEEDS:
            set_seed(seed)
            metrics_utils.RESULT_SUFFIX = f'_{PROMPT_TAG}_seed{seed}'
            for source, targets in runs:
                runner = SequentialDomainRunner(
                    device=device,
                    generator=get_generator(seed),
                    worker_init_fn=worker_init_fn,
                )
                for idx, (tag, cnn_branch, cba_n, gate, cka_fusion,
                          lcka) in enumerate(DSR_SAM2_ABLATIONS, start=1):
                    cfg = dict(base_cfg)
                    cfg.update({
                        'seed': seed,
                        'cnn_branch': cnn_branch,
                        'cba_last_n': cba_n,
                        'use_attention_gate': gate,
                        'use_attention_gate_cba': gate,
                        'cka_fusion': cka_fusion,
                        'compute_cka_loss': lcka,
                        'result_name_tag': f'dsr_sam2_ab2_{tag}',
                    })
                    print(f'\n{"#" * 70}')
                    print(f'# ABLATION {idx}/{len(DSR_SAM2_ABLATIONS)} '
                          f'| seed={seed} | {source}->{"+".join(targets)} '
                          f'| {tag}')
                    print(f'#   cnn_branch={cnn_branch} cba_last_n={cba_n} '
                          f'gate={gate} cka_fusion={cka_fusion} lcka={lcka}')
                    print(f'# {"#" * 68}')
                    sys.stdout.flush()
                    cleanup_resources(device)
                    try:
                        module.run_single_source(source, targets, cfg, runner)
                    except Exception as exc:
                        failed.append(
                            (source, tag, seed, f'{type(exc).__name__}: {exc}'))
                        print(f'\n# ABLATION {source} {tag} seed {seed} '
                              f'FAILED: {type(exc).__name__}: {exc}')
                        traceback.print_exc()
                        sys.stdout.flush()
                cleanup_resources(device)
    finally:
        metrics_utils.RESULT_SUFFIX = ''

    print(f'\n{"#" * 70}')
    if failed:
        print(f'# DSR-SAM2 ablation finished with {len(failed)} failed run(s):')
        for source, tag, seed, err in failed:
            print(f'#   - {source} {tag} seed {seed}: {err}')
    else:
        print('# DSR-SAM2 ablation complete. All runs finished.')
    print(f'# weight: {weight_path}')
    print(f'{"#" * 70}')
    sys.stdout.flush()


def main(argv=None):
    study_args = parse_study_argv(argv)
    if study_args.study and study_args.study != 'none':
        metrics_utils.RESULT_SUFFIX = ''
        if not study_args.dry_run:
            reset_log(study_args.study)
        run_dsr_sam2_ablation_study(dry_run=study_args.dry_run)
        return

    methods = parse_methods_argv(argv)
    datasets = parse_datasets_argv(argv)
    metrics_utils.RESULT_SUFFIX = ''
    reset_log('sota')
    print(f'\n{"#" * 70}')
    print(f'# SOTA run: {len(methods)} method(s) consecutively')
    print(f'# methods : {methods}')
    print(f'# sources : {datasets} (train each source, test the held-out targets)')
    print(f'# seeds   : {list(SEEDS)}')
    print(f'# prompt  : {PROMPT_TAG} ({PROMPT_METHOD}, file {PROMPT_FILE})')
    print(f'# prompt file: {PROMPT_FILE}')
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
