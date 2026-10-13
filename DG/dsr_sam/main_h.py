"""
DSR-SAM hyperparameter analysis (main_h.py).

Grid search over every DSR-SAM hyperparameter EXCEPT the number of epochs and
the early-stopping patience. Edit `HP_GRID` below to choose the values; every
case is a combination of the listed values (see `GRID_MODE`).

    GRID_MODE = 'product'  -> full cartesian product (true grid search)
    GRID_MODE = 'sweep'    -> one-factor-at-a-time (vary one HP, rest default)

All cases are run for EVERY chosen training (source) domain and tested on the
held-out (target) domains right after training.

Dataset choice (source domains to train on):
    DSR_SAM_DATASETS=OTU,OVATUS,USOVA    # default: all three
    # `DSR_SAM_RUNS` is accepted as an alias. Targets are always the other
    # domains of the leave-one-out benchmark.

Weight policy (save disk):
    ALL cases x every source overwrite ONE file:
        weights/dsr_sam/analysis/vit_b_hparam_overwrite_best.pth
    The results.json name lists every hyperparameter=value as a suffix, e.g.
        vit_b_dsr_sam_hparam_base_lr0.0005_rank64_kd_weight1e-07_..._s_OTU_t_OVATUS_best

Usage (from y_DG/ with khoa-env active):
    python3 -u -m DG.dsr_sam.main_h
    DSR_SAM_DATASETS=OTU python3 -u -m DG.dsr_sam.main_h
    DSR_SAM_EPOCHS=2 python3 -u -m DG.dsr_sam.main_h      # smoke test
"""
import itertools
import os
import sys

import numpy as np
import torch

from utils.seed import set_seed
from utils.env import get_device
from utils.sequential_training import SequentialDomainRunner, cleanup_resources
from utils.metrics import save_results
from DG.dsr_sam.data import get_dsr_sam_loaders
from DG.dsr_sam.model import build_dsr_sam
from DG.dsr_sam.train2 import train_dsr_sam, OVERWRITE_DIR, shared_best
from DG.dsr_sam.test import test_dsr_sam_on_target

set_seed()

ALL_DATASETS = ['OTU', 'OVATUS', 'USOVA']
SOURCE_RUNS = [
    ('OTU', ['OVATUS', 'USOVA']),
    ('OVATUS', ['OTU', 'USOVA']),
    ('USOVA', ['OTU', 'OVATUS']),
]
OVERWRITE_NAME = 'hparam_overwrite'

# ---------------------------------------------------------------------------
# hyperparameter grid: edit the value lists to change the search.
# Every key here is also a config key (except the model-build args, which are
# mapped explicitly in run_one_case). `n_epochs` and `patience` are excluded.
# ---------------------------------------------------------------------------
GRID_MODE = 'product'   # 'product' (full grid) or 'sweep' (one-at-a-time)

HP_GRID = {
    # data / optimization
    'image_size': [256],
    'batch_size': [8],
    'base_lr': [0.0001, 0.0005, 0.001],
    'warmup_period': [250],
    # LoRA / EMA
    'rank': [32, 64],
    'ema_mode': [True],
    'ema_rate': [0.999],
    # losses
    'kd_weight': [1e-7, 1e-6],
    'beta1': [0.1],
    # subspace regularization / L_tsd
    'compute_l_tsd': [True],
    'd_delta': [True],
    'truncation': [True],
    'truncation_size': [96],
    'top_s_change_rates': [96],
    'truncation_period': [4],
    'dash_warm': [300],
}


def get_base_config():
    """Base configuration (grid keys overwritten per case)."""
    return {
        'image_size': 256,
        'batch_size': 8,
        'num_workers': 4,
        'pin_memory': True,
        'device': get_device(),  # CUDA_DEVICE from y_DG/.env

        'n_epochs': 100,
        'base_lr': 0.0005,
        'warmup_period': 250,
        'patience': 20,          # early stopping (never part of the grid)

        'num_classes': 1,
        'rank': 64,
        'ema_mode': True,
        'ema_rate': 0.999,
        'kd_weight': 1e-7,
        'beta1': 0.1,
        'truncation': True,
        'truncation_size': 96,
        'top_s_change_rates': 96,
        'truncation_period': 4,
        'dash_warm': 300,
        'compute_l_tsd': True,
        'd_delta': True,

        'experiment_process_save_epochs': False,
        'save_last_epoch': False,

        'phase': 'train',
        'model_type': 'vit_b',
        'checkpoint': 'weights/sam/sam_vit_b_01ec64.pth',
    }


def get_runs():
    """Source (train) domains -> held-out targets, filtered by the dataset env."""
    sel = os.environ.get('DSR_SAM_DATASETS') or os.environ.get('DSR_SAM_RUNS')
    if not sel:
        return SOURCE_RUNS
    sel = [s.strip().upper() for s in sel.replace(',', ' ').split() if s.strip()]
    unknown = [s for s in sel if s not in ALL_DATASETS]
    if unknown:
        raise SystemExit(f'Unknown dataset(s): {unknown}. Choose from {ALL_DATASETS}')
    return [(s, list(ts)) for s, ts in SOURCE_RUNS if s in sel]


def generate_cases():
    """Return the list of hyperparameter dicts to run (grid or sweep)."""
    keys = list(HP_GRID.keys())
    if GRID_MODE == 'product':
        combos = itertools.product(*[HP_GRID[k] for k in keys])
        return [dict(zip(keys, c)) for c in combos]
    if GRID_MODE == 'sweep':
        base = {k: HP_GRID[k][0] for k in keys}
        cases = [dict(base)]
        for k in keys:
            for v in HP_GRID[k][1:]:
                c = dict(base)
                c[k] = v
                cases.append(c)
        return cases
    raise SystemExit(f"GRID_MODE must be 'product' or 'sweep', got {GRID_MODE!r}")


def _val(value):
    """Compact value tag for results names."""
    if isinstance(value, bool):
        return 'T' if value else 'F'
    return str(value)


def hp_tag(case):
    """List every hyperparameter=value (HP_GRID order) as a results suffix."""
    return '_'.join(f'{k}{_val(case[k])}' for k in HP_GRID)


def result_name(model_type, tag, source, target):
    return f'{model_type}_dsr_sam_hparam_{tag}_s_{source}_t_{target}_best'


def run_one_case(case_idx, n_cases, case, source, targets, base_config, runner):
    """Train ONE hyperparameter case (source) into the shared weight, then test."""
    cfg = dict(base_config)
    cfg.update(case)
    # L_tsd off -> no L_tsd term regardless of the beta1 grid value.
    if not cfg.get('compute_l_tsd', False):
        cfg['beta1'] = 0.0
    # d_delta is only meaningful with L_tsd on.
    if not cfg.get('compute_l_tsd', False):
        cfg['d_delta'] = False
    cfg['source'] = source
    cfg['overwrite_name'] = OVERWRITE_NAME

    tag = hp_tag(case)
    model_type = cfg['model_type']
    best_path = shared_best(model_type, OVERWRITE_NAME)

    print(f'\n{"=" * 70}')
    print(f'HYPERPARAMETER CASE {case_idx:03d}/{n_cases} [{source}]: {tag}')
    print(f'  Source: {source} | Targets: {", ".join(targets)}')
    print(f'  Weight file (shared, overwritten): {best_path}')
    print(f'{"=" * 70}')
    sys.stdout.flush()

    with runner.domain_context():
        device = cfg['device']
        model = build_dsr_sam(
            checkpoint=cfg['checkpoint'], model_type=model_type,
            image_size=cfg['image_size'], num_classes=cfg['num_classes'],
            rank=cfg['rank'], ema_mode=cfg['ema_mode'],
            truncation_size=cfg['truncation_size'],
            top_s_change_rates=cfg['top_s_change_rates'],
            d_delta=cfg['d_delta'],
        ).to(device)

        train_loader, val_loader = get_dsr_sam_loaders(
            source, image_size=cfg['image_size'], batch_size=cfg['batch_size'],
            num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'],
        )
        runner.register_loaders(train_loader)
        runner.register_loaders(val_loader)

        train_dsr_sam(model, train_loader, val_loader, device, cfg)
        runner.destroy_loaders()

        if not os.path.exists(best_path):
            print(f'  WARNING: shared weight not found at {best_path}; skipping test.')
            return
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        for target in targets:
            metrics = test_dsr_sam_on_target(
                model, target, device,
                image_size=cfg['image_size'], batch_size=cfg['batch_size'],
                num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'],
                source_name=source, model_type=model_type,
                write_results=False, weight_tag=tag,
            )
            name = result_name(model_type, tag, source, target)
            save_results(name, {
                'dice': float(np.round(metrics[0], 2)),
                'iou': float(np.round(metrics[1], 2)),
                'precision': float(np.round(metrics[2], 2)),
                'recall': float(np.round(metrics[3], 2)),
                'hd95': float(np.round(metrics[4], 2)),
            })
            print(f'  [TEST] {name} | Dice: {metrics[0]:.2f} | IoU: {metrics[1]:.2f} | '
                  f'Prec: {metrics[2]:.2f} | Rec: {metrics[3]:.2f} | HD95: {metrics[4]:.2f}')
            sys.stdout.flush()


def main():
    cfg = get_base_config()
    if os.environ.get('DSR_SAM_EPOCHS'):
        cfg['n_epochs'] = int(os.environ['DSR_SAM_EPOCHS'])

    runs = get_runs()
    cases = generate_cases()

    device = torch.device(cfg['device'])
    runner = SequentialDomainRunner(device=device)

    # Drop any stale analysis weight so only the current run's file remains.
    try:
        os.makedirs(OVERWRITE_DIR, exist_ok=True)
        for f in os.listdir(OVERWRITE_DIR):
            fp = os.path.join(OVERWRITE_DIR, f)
            if os.path.isfile(fp) and f.endswith('.pth'):
                os.remove(fp)
    except OSError:
        pass

    print(f'\n{"#" * 70}')
    print('# DSR-SAM HYPERPARAMETER ANALYSIS (main_h)')
    print(f'# mode     : {GRID_MODE}')
    print(f'# grid     : {HP_GRID}')
    print(f'# cases    : {len(cases)} per training domain')
    print(f'# sources  : {[s for s, _ in runs]}')
    print(f'# weight   : weights/dsr_sam/analysis/{cfg["model_type"]}_{OVERWRITE_NAME}_best.pth')
    print(f'# epochs   : {cfg["n_epochs"]}')
    print(f'{"#" * 70}')
    sys.stdout.flush()

    for source, targets in runs:
        for case_idx, case in enumerate(cases, start=1):
            cleanup_resources(device)
            run_one_case(case_idx, len(cases), case, source, targets, cfg, runner)

    print(f'\n{"=" * 70}')
    print('DSR-SAM HYPERPARAMETER ANALYSIS COMPLETE')
    print('Results: results.json (unique name per case)')
    print(f'{"=" * 70}')
    sys.stdout.flush()


if __name__ == '__main__':
    main()
