"""
DSR-SAM ablation study (main_a.py).

Full 2^4 = 16-case ablation over the four DSR-SAM switches:

    l_tsd      -> cfg['compute_l_tsd']   (beta1 = 0.1 when True, else 0.0)
    d_delta    -> cfg['d_delta']
    truncation -> cfg['truncation']
    ema_lora   -> cfg['ema_mode']

All 16 cases are run for EVERY chosen training (source) domain, and each case
is tested on the held-out (target) domains right after training.

Dataset choice (source domains to train on):
    DSR_SAM_DATASETS=OTU,OVATUS,USOVA    # default: all three
    # `DSR_SAM_RUNS` is accepted as an alias. Targets are always the other
    # domains of the leave-one-out benchmark.

Weight policy (save disk):
    ALL 16 cases x every source overwrite ONE file:
        weights/dsr_sam2/analysis/vit_b_ablation_overwrite_best.pth
    Per-case identity is kept ONLY in results.json; the name lists all four
    flags, e.g.
        vit_b_dsr_sam2_ablation_ltsdT_ddeltaF_truncT_emaF_s_OTU_t_OVATUS_best

Usage (from y_DG/ with khoa-env active):
    python3 -u -m DG.dsr_sam2.main_a
    DSR_SAM_DATASETS=OTU python3 -u -m DG.dsr_sam2.main_a
    DSR_SAM_EPOCHS=2 python3 -u -m DG.dsr_sam2.main_a      # smoke test
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
from DG.dsr_sam2.data import get_dsr_sam2_loaders
from DG.dsr_sam2.model import build_dsr_sam2
from DG.dsr_sam2.train2 import train_dsr_sam2, OVERWRITE_DIR, shared_best
from DG.dsr_sam2.test import test_dsr_sam2_on_target

set_seed()

ALL_DATASETS = ['OTU', 'OVATUS', 'USOVA']
SOURCE_RUNS = [
    ('OTU', ['OVATUS', 'USOVA']),
    ('OVATUS', ['OTU', 'USOVA']),
    ('USOVA', ['OTU', 'OVATUS']),
]
# Order matches the four switches requested for the ablation.
ABLATION_VARS = ['l_tsd', 'd_delta', 'truncation', 'ema_lora']
L_TSD_BETA = 0.1
OVERWRITE_NAME = 'ablation_overwrite'


def get_base_config():
    """Base configuration shared by all 16 cases (switches overwritten)."""
    return {
        'image_size': 256,
        'batch_size': 8,
        'num_workers': 4,
        'pin_memory': True,
        'device': get_device(),  # CUDA_DEVICE from y_DG/.env

        'seed': 42,              # RNG seed (reset per case for reproducibility)

        'n_epochs': 100,
        'base_lr': 0.0005,
        'warmup_period': 250,
        'patience': 20,          # early stopping on val loss (0 = disabled)

        'num_classes': 1,
        'rank': 64,
        'ema_mode': False,       # overwritten per case by ema_lora
        'ema_rate': 0.999,
        'kd_weight': 1e-7,
        'beta1': 0.0,            # overwritten per case by l_tsd
        'truncation': False,     # overwritten per case
        'truncation_size': 96,
        'top_s_change_rates': 96,
        'truncation_period': 4,
        'dash_warm': 300,
        'compute_l_tsd': False,  # overwritten per case by l_tsd
        'alpha': 0.1,            # coefficient of the L_tsd log-squash term
        'd_delta': False,        # overwritten per case

        'experiment_process_save_epochs': False,
        'save_last_epoch': False,

        'phase': 'train',
        'model_type': 'vit_b',
        'checkpoint': 'weights/sam/sam_vit_b_01ec64.pth',

        # DSR-SAM2 dual-branch architecture
        'cnn_pretrained': True,
        'cba_last_n': 4,
        'grassmann_grad': False,  # False -> detach G_score (fixed gate)
        'truncate_before_compute_grassmann': 0,
        'truncate_ref': False,
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


def flag(value):
    """Compact True/False tag: True -> 'T', False -> 'F'."""
    return 'T' if value else 'F'


def case_tag(l_tsd, d_delta, truncation, ema_lora):
    """Results tag carrying ALL four ablation switches."""
    return (f'ltsd{flag(l_tsd)}_ddelta{flag(d_delta)}'
            f'_trunc{flag(truncation)}_ema{flag(ema_lora)}')


def result_name(model_type, tag, source, target):
    return f'{model_type}_dsr_sam2_ablation_{tag}_s_{source}_t_{target}_best'


def run_one_case(case_idx, flags, source, targets, base_config, runner):
    """Train ONE ablation case (source) into the shared weight, then test."""
    l_tsd, d_delta, truncation, ema_lora = flags
    cfg = dict(base_config)
    cfg.update({
        'compute_l_tsd': bool(l_tsd),
        'beta1': L_TSD_BETA if l_tsd else 0.0,
        'd_delta': bool(d_delta),
        'truncation': bool(truncation),
        'ema_mode': bool(ema_lora),
        'source': source,
        'overwrite_name': OVERWRITE_NAME,
    })
    # Phase 2 (reproducibility): reseed per case so each case is independent
    # of the RNG consumed by previously run cases.
    set_seed(cfg.get('seed', 42))
    tag = case_tag(*flags)
    model_type = cfg['model_type']
    best_path = shared_best(model_type, OVERWRITE_NAME)

    print(f'\n{"=" * 70}')
    print(f'ABLATION CASE {case_idx:02d}/16 [{source}]: {tag}')
    print(f'  l_tsd={l_tsd} (beta1={cfg["beta1"]}) | d_delta={d_delta} | '
          f'truncation={truncation} | lora_ema={ema_lora}')
    print(f'  Source: {source} | Targets: {", ".join(targets)}')
    print(f'  Weight file (shared, overwritten): {best_path}')
    print(f'{"=" * 70}')
    sys.stdout.flush()

    with runner.domain_context():
        device = cfg['device']
        model = build_dsr_sam2(
            checkpoint=cfg['checkpoint'], model_type=model_type,
            image_size=cfg['image_size'], num_classes=cfg['num_classes'],
            rank=cfg['rank'], ema_mode=cfg['ema_mode'],
            truncation_size=cfg['truncation_size'],
            top_s_change_rates=cfg['top_s_change_rates'],
            d_delta=cfg['d_delta'],
            alpha=cfg.get('alpha', 0.1),
            cnn_pretrained=cfg.get('cnn_pretrained', True),
            cba_last_n=cfg.get('cba_last_n', 4),
            grassmann_detach=not cfg.get('grassmann_grad', False),
            truncate_before_compute_grassmann=cfg.get(
                'truncate_before_compute_grassmann', 0),
            truncate_ref=cfg.get('truncate_ref', False),
        ).to(device)

        train_loader, val_loader = get_dsr_sam2_loaders(
            source, image_size=cfg['image_size'], batch_size=cfg['batch_size'],
            num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'],
        )
        runner.register_loaders(train_loader)
        runner.register_loaders(val_loader)

        train_dsr_sam2(model, train_loader, val_loader, device, cfg)
        runner.destroy_loaders()

        if not os.path.exists(best_path):
            print(f'  WARNING: shared weight not found at {best_path}; skipping test.')
            return
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        for target in targets:
            metrics = test_dsr_sam2_on_target(
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
    grid = list(itertools.product([False, True], repeat=len(ABLATION_VARS)))

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
    print('# DSR-SAM ABLATION STUDY (main_a)')
    print(f'# switches : {ABLATION_VARS}')
    print(f'# cases    : {len(grid)} per training domain')
    print(f'# sources  : {[s for s, _ in runs]}')
    print(f'# weight   : weights/dsr_sam2/analysis/{cfg["model_type"]}_{OVERWRITE_NAME}_best.pth')
    print(f'# epochs   : {cfg["n_epochs"]}')
    print(f'{"#" * 70}')
    sys.stdout.flush()

    for source, targets in runs:
        for case_idx, flags in enumerate(grid, start=1):
            cleanup_resources(device)
            run_one_case(case_idx, flags, source, targets, cfg, runner)

    print(f'\n{"=" * 70}')
    print('DSR-SAM ABLATION STUDY COMPLETE')
    print('Results: results.json (unique name per case)')
    print(f'{"=" * 70}')
    sys.stdout.flush()


if __name__ == '__main__':
    main()
