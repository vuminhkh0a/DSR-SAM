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
    # DSR-SAM L_tsd alpha -> d_delta study (2 stages, auto-chained, rank fixed
    # at 8, sources default to OVATUS + USOVA unless --datasets is given):
    python3 -m DG.sota.run_sota --methods dsr_sam_study

``run.sh`` launches it detached, with every line streamed to ``log.txt``:
    nohup python3 -u -m DG.sota.run_sota > "$LOG_FILE" 2>&1 &

Monitor:  tail -f ~/KhoaVM/y_DG/log.txt
Stop:     pkill -f 'DG[.]sota[.]run_sota'
"""

import argparse
import datetime
import importlib
import inspect
import json
import os
import re
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
# the end of the list, as is the DSR-SAM rank / L_tsd(alpha) / d_delta study
# (method ``dsr_sam_study``); the remaining hyper-parameter studies stay in
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
DSR_SAM2_ABLATION_SEEDS = (42, 0, 100)
DSR_SAM2_ABLATION_FIXED = {
    'cnn_pretrained': True,
    'attention_gate_channels': 768,
    'cka_loss_weight': 0.1,
    'weight_name_tag': 'dsr_sam2_cka_ablation',
}

# ---- DSR-SAM L_tsd(alpha) / d_delta study (rank fixed) -------------------
# Rank is fixed at DSR_SAM_STUDY_RANK (the rank sweep is skipped). One
# auto-chained 2-stage study (method name ``dsr_sam_study``):
#   stage 1: L_tsd ON, alpha in DSR_SAM_STUDY_ALPHAS, d_delta OFF, rank fixed
#   stage 2: best alpha, d_delta ON (the OFF rows come from the stage-1
#            winner, which already carries ``deltaOFF`` in its name)
# Every case runs on each seed in DSR_SAM_STUDY_SEEDS. "Best" = highest mean
# Dice over all (source -> target) cases x seeds, ties broken by lower mean
# HD95. results.json names are readable:
#   vit_b_dsr_sam_rank8_ltsdON_alpha0.01_deltaOFF_seed42_s_OTU_t_OVATUS_best
#   vit_b_dsr_sam_rank8_ltsdON_alpha0.01_deltaON_seed42_s_OTU_t_OVATUS_best
# Weights: all variants/seeds of one source overwrite ONE file
#   weights/dsr_sam/analysis/vit_b_dsr_sam_study_s_<source>_best.pth
DSR_SAM_STUDY_METHOD = 'dsr_sam_study'
DSR_SAM_STUDY_RANK = 8          # fixed; the rank sweep is skipped
DSR_SAM_STUDY_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 2.0)
DSR_SAM_STUDY_SEEDS = (42, 0, 100)
DSR_SAM_STUDY_WEIGHT = 'dsr_sam_study'
# Default scope: the remaining sources OVATUS and USOVA (OTU is already done);
# an explicit --datasets / SOTA_DATASETS overrides this.
DSR_SAM_STUDY_DEFAULT_DATASETS = ('OVATUS', 'USOVA')

METHOD_SPECS = [
    # ('unet',             'DG.unet.main',             {}),
    # ('dualnormalization', 'DG.dualnormalization.main', {}),
    # ('mi_segnet',        'DG.mi_segnet.main',        {}),
    # ('maxstyle',         'DG.maxstyle.main',         {}),
    # ('desam',            'DG.desam.main',            {}),
    # ('masam',            'DG.masam.main',            {}),
    # ('sr_sam',           'DG.sr_sam.main',           {}),
    # ('dsr_sam',         'DG.dsr_sam.main',         {}),
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
] + [
    # DSR-SAM L_tsd(alpha) / d_delta study (2 stages, fixed rank 8).
    (DSR_SAM_STUDY_METHOD, 'DG.dsr_sam.main', {}),
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
# True when --datasets / SOTA_DATASETS was passed (lets the DSR-SAM study keep
# its OTU-only default scope otherwise).
_DATASETS_EXPLICIT = False


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
    global _DATASETS_EXPLICIT
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--datasets', default=None)
    args, _ = parser.parse_known_args(argv)
    if args.datasets:
        selected = _csv_list(args.datasets)
        _DATASETS_EXPLICIT = True
    elif os.environ.get('SOTA_DATASETS'):
        selected = _csv_list(os.environ['SOTA_DATASETS'])
        _DATASETS_EXPLICIT = True
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


# ---------------------------------------------------------------------------
# DSR-SAM rank / L_tsd(alpha) / d_delta study (method: dsr_sam_study)
# ---------------------------------------------------------------------------

def _fmt_num(value):
    """Compact numeric tag, e.g. 8 -> '8', 0.01 -> '0.01', 1e-4 -> '0.0001'."""
    return f'{value:g}'


def _strip_gpu_prefix(name):
    return re.sub(r'^(?:gpu\d+|cpu)_', '', str(name))


def _load_results_index(path='results.json'):
    """name (without gpuN_ prefix) -> metrics dict."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            rows = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    if isinstance(rows, dict):
        rows = [rows]
    index = {}
    for row in rows:
        if isinstance(row, dict):
            index[_strip_gpu_prefix(row.get('name', ''))] = row.get('results', {})
    return index


def _study_tag(stage, rank, alpha, d_delta, seed):
    """results.json identity of one study case (readable, no weight collision)."""
    ltsd = 'OFF' if stage == 1 else 'ON'
    alpha_part = f'_alpha{_fmt_num(alpha)}' if stage >= 2 else ''
    return (f'dsr_sam_rank{rank}_ltsd{ltsd}{alpha_part}'
            f'_delta{"ON" if d_delta else "OFF"}_seed{seed}')


def _study_case_names(model_type, tags, runs, weight_tag='best'):
    names = []
    for tag in tags:
        for source, targets in runs:
            for target in targets:
                names.append(f'{model_type}_{tag}_s_{source}_t_{target}_{weight_tag}')
    return names


def _score_tags(index, model_type, tags, runs):
    """(mean Dice, mean HD95) over every target case and seed, or None."""
    names = _study_case_names(model_type, tags, runs)
    if not names or any(name not in index for name in names):
        return None
    dice = [float(index[name]['dice']) for name in names]
    hd95 = [float(index[name].get('hd95', 0.0)) for name in names]
    return sum(dice) / len(dice), sum(hd95) / len(hd95)


def _score_candidates(candidates, index, model_type, runs):
    """Print one row per candidate; return them sorted best-first.

    ``candidates`` is [(label, tags, value)]; the ranking key is highest mean
    Dice, ties broken by lowest mean HD95.
    """
    scored = []
    for label, tags, value in candidates:
        score = _score_tags(index, model_type, tags, runs)
        if score is None:
            print(f'#   {label}: skipped (incomplete results)')
            continue
        mean_dice, mean_hd95 = score
        print(f'#   {label}: mean Dice {mean_dice:.2f} | mean HD95 {mean_hd95:.2f}')
        scored.append((mean_dice, -mean_hd95, label, tags, value))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored


def _study_cfg(base, *, rank, alpha, compute_l_tsd, d_delta, seed, tag):
    cfg = dict(base)
    cfg.update({
        'rank': rank,
        'alpha': alpha,
        'compute_l_tsd': compute_l_tsd,
        'd_delta': d_delta,
        'beta1': base.get('beta1', 0.1) if compute_l_tsd else 0.0,
        'seed': seed,
        # Case identity lives only in results.json; the checkpoint is shared.
        'result_tag': tag,
        'plain_result_name': True,
        'run_name': DSR_SAM_STUDY_WEIGHT,
        'overwrite_name': DSR_SAM_STUDY_WEIGHT,
        'overwrite_dir': os.path.join('weights', 'dsr_sam', 'analysis'),
        'save_last_epoch': False,
        'experiment_process_save_epochs': False,
    })
    return cfg


def _run_study_case(module, cfg, runs, device):
    """Train+test one study case on every (source -> targets) run."""
    seed = cfg['seed']
    set_seed(seed)
    runner = SequentialDomainRunner(
        device=device,
        generator=get_generator(seed),
        worker_init_fn=worker_init_fn,
    )
    for source, targets in runs:
        cleanup_resources(device)
        module.run_single_source(source, targets, cfg, runner)


def run_dsr_sam_study(datasets):
    """2-stage DSR-SAM study at a fixed rank, auto-chained on one invocation.

    Stage 1 sweeps the L_tsd alpha (L_tsd ON, d_delta OFF); stage 2 tests
    d_delta ON at the winning alpha. The rank is fixed at
    DSR_SAM_STUDY_RANK, so no rank sweep is run. See the constants above.
    """
    module = importlib.import_module('DG.dsr_sam.main')
    base = dict(module.CONFIG)
    if os.environ.get('DSR_SAM_EPOCHS'):
        base['n_epochs'] = int(os.environ['DSR_SAM_EPOCHS'])
    # Explicit --datasets / SOTA_DATASETS wins over the default scope.
    selected = datasets if _DATASETS_EXPLICIT else list(DSR_SAM_STUDY_DEFAULT_DATASETS)
    runs = _runs_for(module, selected)
    if not runs:
        print(f'# {DSR_SAM_STUDY_METHOD}: no (source, targets) runs for '
              f'datasets={selected}; skipping.')
        return

    device = torch.device(base.get('device') or 'cpu')
    model_type = base.get('model_type', 'vit_b')
    best_rank = DSR_SAM_STUDY_RANK
    alphas_txt = ', '.join(_fmt_num(a) for a in DSR_SAM_STUDY_ALPHAS)

    print(f'\n{"#" * 70}')
    print('# DSR-SAM STUDY: L_tsd alpha -> d_delta (2 stages, auto-chained)')
    print(f'# seeds    : {list(DSR_SAM_STUDY_SEEDS)}')
    print(f'# sources  : {[s for s, _ in runs]} -> targets '
          f'{sorted({t for _, ts in runs for t in ts})}')
    print(f'# rank     : {best_rank} (fixed; rank sweep skipped)')
    print(f'# alphas   : {alphas_txt}')
    print(f'# weight   : weights/dsr_sam/analysis/{model_type}_'
          f'{DSR_SAM_STUDY_WEIGHT}_s_<source>_best.pth (shared per source)')
    print(f'# epochs   : {base["n_epochs"]}')
    print(f'# selection: highest mean Dice over all cases x seeds '
          f'(tie -> lower mean HD95)')
    print(f'{"#" * 70}')
    sys.stdout.flush()

    # ---- Stage 1: alpha with L_tsd ON (d_delta OFF), fixed rank ----------
    print(f'\n{"=" * 70}\nSTAGE 1/2: L_tsd alpha sweep at rank={best_rank} '
          f'(d_delta OFF)\n{"=" * 70}')
    sys.stdout.flush()
    stage2 = []
    for alpha in DSR_SAM_STUDY_ALPHAS:
        tags = []
        for seed in DSR_SAM_STUDY_SEEDS:
            tag = _study_tag(2, best_rank, alpha, False, seed)
            cfg = _study_cfg(base, rank=best_rank, alpha=alpha,
                             compute_l_tsd=True, d_delta=False, seed=seed, tag=tag)
            print(f'\n--- stage1 rank={best_rank} alpha={_fmt_num(alpha)} '
                  f'seed={seed} -> {tag}')
            sys.stdout.flush()
            try:
                _run_study_case(module, cfg, runs, device)
            except Exception:
                traceback.print_exc()
                print(f'# stage1 alpha={alpha} seed={seed} FAILED; continuing.')
            tags.append(tag)
        stage2.append((f'alpha={_fmt_num(alpha)}', tags, alpha))

    ranked2 = _score_candidates(stage2, _load_results_index(), model_type, runs)
    if not ranked2:
        raise RuntimeError('DSR-SAM study: stage 1 produced no complete results.')
    best_alpha = ranked2[0][4]
    best_delta_off_tags = ranked2[0][3]
    print(f'\n# STAGE 1 WINNER: alpha={_fmt_num(best_alpha)} '
          f'(mean Dice {ranked2[0][0]:.2f})')

    # ---- Stage 2: d_delta ON vs OFF at the winning alpha -----------------
    # The OFF rows are the stage-1 winner (already named ..._deltaOFF_...);
    # only the ON cases need training.
    print(f'\n{"=" * 70}\nSTAGE 2/2: d_delta ON/OFF at rank={best_rank}, '
          f'alpha={_fmt_num(best_alpha)}\n{"=" * 70}')
    sys.stdout.flush()
    stage3 = [('delta=OFF (reused stage-1 winner)', best_delta_off_tags, False)]
    delta_on_tags = []
    for seed in DSR_SAM_STUDY_SEEDS:
        tag = _study_tag(3, best_rank, best_alpha, True, seed)
        cfg = _study_cfg(base, rank=best_rank, alpha=best_alpha,
                         compute_l_tsd=True, d_delta=True, seed=seed, tag=tag)
        print(f'\n--- stage2 rank={best_rank} alpha={_fmt_num(best_alpha)} '
              f'delta=ON seed={seed} -> {tag}')
        sys.stdout.flush()
        try:
            _run_study_case(module, cfg, runs, device)
        except Exception:
            traceback.print_exc()
            print(f'# stage2 delta=ON seed={seed} FAILED; continuing.')
        delta_on_tags.append(tag)
    stage3.append(('delta=ON', delta_on_tags, True))

    ranked3 = _score_candidates(stage3, _load_results_index(), model_type, runs)
    best_delta = ranked3[0][4] if ranked3 else None

    print(f'\n{"#" * 70}')
    print('# DSR-SAM STUDY COMPLETE')
    print(f'# rank (fixed)   : {best_rank}')
    print(f'# best L_tsd alpha: {_fmt_num(best_alpha)}')
    print(f'# best d_delta   : {"ON" if best_delta else "OFF" if best_delta is not None else "n/a"}')
    print('# selection      : highest mean Dice over all cases x seeds')
    print('# results        : results.json (readable name per case)')
    print('# weights        : weights/dsr_sam/analysis/'
          f'{model_type}_{DSR_SAM_STUDY_WEIGHT}_s_<source>_best.pth')
    print(f'{"#" * 70}')
    sys.stdout.flush()


def run_method(name, datasets):
    # The DSR-SAM study is a multi-stage job, not a single method module.
    if name == DSR_SAM_STUDY_METHOD:
        return run_dsr_sam_study(datasets)
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
    if DSR_SAM_STUDY_METHOD in methods and not _DATASETS_EXPLICIT:
        print(f'#           dsr_sam_study defaults to sources='
              f'{list(DSR_SAM_STUDY_DEFAULT_DATASETS)} (pass --datasets to change)')
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
