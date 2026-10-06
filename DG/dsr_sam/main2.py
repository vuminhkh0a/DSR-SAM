"""
DSR-SAM 8-case ablation runner (OTU -> OVATUS only).
# Previous direction (commented out): OVATUS -> OTU only.

Grid search: 2^3 = 8 True/False combinations of:
  - ema_lora   (bool) -> cfg['ema_mode']
  - truncation (bool) -> cfg['truncation']
  - l_tsd      (bool) -> True:  compute_l_tsd=True,  beta1=0.1 (hardcoded)
                             False: compute_l_tsd=False, beta1=0.0

Scope: ONLY the OTU -> OVATUS direction. Each case trains on OTU,
then tests on OVATUS immediately (before the next case overwrites
the shared weight).
# Old scope (commented out): ONLY the OVATUS -> OTU direction. Each case
# trained on OVATUS, then tested on OTU immediately.

Weight policy (save disk):
  - ONE shared file, overwritten every run:
        weights/dsr_sam/vit_b_dsr_sam_overwrite_best.pth
    (see DG/dsr_sam/train2.py). No per-case weight files are kept.
Results policy:
  - results.json gets a DIFFERENT name per case, e.g.
      dsr_sam_abl_case03_emaT_truncF_ltsdT_sOTU_tOVATUS
    written IMMEDIATELY after that case's test.
  # Old naming (commented out):
  #     dsr_sam_abl_case03_emaT_truncF_ltsdT_frzF_sOVATUS_tOTU
Logging policy:
  - test metrics are printed to stdout AND appended to y_DG/log.txt
    (logging FileHandler, mode='a') IMMEDIATELY after each case.
"""
import itertools
import os
import sys
import torch
import logging

import numpy as np

from utils.seed import set_seed
from utils.env import get_device
from utils.sequential_training import SequentialDomainRunner, cleanup_resources
from utils.metrics import save_results
from DG.dsr_sam.data import get_dsr_sam_loaders
from DG.dsr_sam.model import build_dsr_sam
from DG.dsr_sam.train2 import (
    train_dsr_sam,
    OVERWRITE_DIR,
    shared_prefix,
    shared_best,
)
from DG.dsr_sam.test import test_dsr_sam_on_target

set_seed()

# Setup logging to file (append mode) and stdout
log_dir = '.'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'log.txt')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'),  # append mode
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def log_print(msg):
    """Print to both stdout and log file."""
    logger.info(msg)
    sys.stdout.flush()


# ---- Scope: OTU -> OVATUS only ----
# Old scope (commented out): OVATUS -> OTU only
# SOURCE = 'OVATUS'
# TARGET = 'OTU'
SOURCE = 'OTU'
TARGET = 'OVATUS'

# Hardcoded ablation constants per user spec.
L_TSD_BETA = 0.1


def get_base_config():
    """Base configuration for all 8 cases."""
    return {
        'image_size': 256,
        'batch_size': 8,
        'num_workers': 4,
        'pin_memory': True,
        'device': get_device(),  # CUDA_DEVICE from y_DG/.env

        'n_epochs': 50,
        'base_lr': 0.0005,
        'warmup_period': 250,

        'num_classes': 1,
        'rank': 64,
        'ema_mode': False,   # overwritten per case by ema_lora
        'ema_rate': 0.999,
        'kd_weight': 1e-7,
        'beta1': 0.0,        # overwritten per case by l_tsd
        'truncation': False,  # overwritten per case
        'truncation_size': 96,
        'top_s_change_rates': 96,  # number of top change rates contributing to L_tsd
        'truncation_period': 4,
        'dash_warm': 300,
        'compute_l_tsd': False,  # overwritten per case by l_tsd
        'patience': 5,

        'experiment_process_save_epochs': False,
        'save_last_epoch': False,  # never keep a 'last' copy in ablation mode

        'phase': 'train',
        'model_type': 'vit_b',
        'checkpoint': 'weights/sam/sam_vit_b_01ec64.pth',
    }


def flag_tag(value):
    """Compact True/False tag for results names: True->'T', False->'F'."""
    return 'T' if value else 'F'


def case_base_name(case_idx, ema_lora, truncation, l_tsd):
    """Unique results.json base name for one ablation case."""
    return (f'dsr_sam_abl_case{case_idx:02d}_'
            f'ema{flag_tag(ema_lora)}_'
            f'trunc{flag_tag(truncation)}_'
            f'ltsd{flag_tag(l_tsd)}')


def run_one_case(case_idx, ema_lora, truncation, l_tsd, config, runner):
    """Train ONE ablation case (OTU) into the shared weight, then test OVATUS.
    # Old direction (commented out): Train ONE ablation case (OVATUS) into the
    # shared weight, then test OTU.

    - Trains into the shared weight (overwritten every call).
    - Right after training, loads that weight and tests OVATUS:
        * prints metrics to log.txt immediately,
        * appends `f'{base}_sOTU_tOVATUS'` to results.json immediately
          (before the next case overwrites the weight).
    # Old naming (commented out): `f'{base}_sOVATUS_tOTU'`.
    """
    os.makedirs(OVERWRITE_DIR, exist_ok=True)

    config = dict(config)
    # Constant prefix -> train.py always saves to the SAME single file.
    # The shared file carries the ViT size (default vit_b).
    model_type = config.get('model_type', 'vit_b')
    config['model_dir'] = OVERWRITE_DIR
    config['prefix'] = shared_prefix(model_type)
    config['best_weight_path'] = shared_best(model_type)
    best_path = shared_best(model_type)

    base_name = case_base_name(case_idx, ema_lora, truncation, l_tsd)

    log_print(f'\n{"="*70}')
    log_print(f'ABLATION CASE {case_idx:02d}/8: {base_name}')
    log_print(f'Source: {SOURCE} | Target: {TARGET}')
    log_print(f'Config: ema_lora={ema_lora}, truncation={truncation}, '
              f'l_tsd={l_tsd} (beta1={config["beta1"]})')
    log_print(f'Weight file (shared, overwritten): {best_path}')
    log_print(f'{"="*70}')

    with runner.domain_context():
        device = config['device']
        model = build_dsr_sam(
            checkpoint=config['checkpoint'], model_type=config['model_type'],
            image_size=config['image_size'], num_classes=config['num_classes'],
            rank=config['rank'], ema_mode=config['ema_mode'],
            truncation_size=config['truncation_size'],
            top_s_change_rates=config['top_s_change_rates'],
        ).to(device)

        train_loader, val_loader = get_dsr_sam_loaders(
            SOURCE, image_size=config['image_size'],
            batch_size=config['batch_size'], num_workers=config['num_workers'],
            pin_memory=config['pin_memory'],
        )
        runner.register_loaders(train_loader)
        runner.register_loaders(val_loader)

        best_val_loss = train_dsr_sam(model, train_loader, val_loader, device, config)
        runner.destroy_loaders()

        metrics_dict = {}
        if os.path.exists(best_path):
            # Load the JUST-trained shared weight and test immediately.
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
            metrics = test_dsr_sam_on_target(
                model, TARGET, device,
                image_size=config['image_size'], batch_size=config['batch_size'],
                num_workers=config['num_workers'], pin_memory=config['pin_memory'],
                source_name=SOURCE, model_type=config['model_type'],
                write_results=False,  # we write our own exact name below
                weight_tag=base_name,
            )
            metrics_dict = {
                'dice': metrics[0], 'iou': metrics[1],
                'precision': metrics[2], 'recall': metrics[3], 'hd95': metrics[4]
            }

            # Unique results.json name per case.
            entry_name = f'{base_name}_s{SOURCE}_t{TARGET}'
            save_results(entry_name, {
                'dice': float(np.round(metrics[0], 2)),
                'iou': float(np.round(metrics[1], 2)),
                'precision': float(np.round(metrics[2], 2)),
                'recall': float(np.round(metrics[3], 2)),
                'hd95': float(np.round(metrics[4], 2)),
            })

            # Print test results to log.txt RIGHT AFTER this case.
            log_print(f'  [TEST] {entry_name} | '
                      f'Dice: {metrics[0]:.2f} | IoU: {metrics[1]:.2f} | '
                      f'Prec: {metrics[2]:.2f} | Rec: {metrics[3]:.2f} | HD95: {metrics[4]:.2f}')
        else:
            best_val_loss = None
            log_print(f'  WARNING: shared weight not found at {best_path}, skipping test.')

    return best_val_loss, metrics_dict


def main():
    if os.environ.get('DSR_SAM_EPOCHS'):
        n_epochs = int(os.environ['DSR_SAM_EPOCHS'])
    else:
        n_epochs = 50

    device = torch.device(get_device())
    runner = SequentialDomainRunner(device=device)

    # Shared weight path follows the configured backbone (default vit_b).
    keep_best = shared_best(get_base_config().get('model_type', 'vit_b'))

    # Remove stale per-study weights; only the single shared file is kept.
    try:
        os.makedirs(OVERWRITE_DIR, exist_ok=True)
        for f in os.listdir(OVERWRITE_DIR):
            fp = os.path.join(OVERWRITE_DIR, f)
            if (fp != keep_best and f.endswith('.pth')
                    and os.path.isfile(fp)):
                os.remove(fp)
                log_print(f'Removed stale per-study weight: {fp}')
    except OSError:
        pass

    # ============================================================
    # 8-case grid: all True/False combos of
    # (ema_lora, truncation, l_tsd), OTU -> OVATUS only.
    # Old direction (commented out): OVATUS -> OTU only.
    # ============================================================
    log_print(f'\n{"#"*70}')
    log_print('# DSR-SAM 8-CASE ABLATION: OTU -> OVATUS')
    # Old log (commented out): '# DSR-SAM 16-CASE ABLATION: OVATUS -> OTU'
    log_print(f'# l_tsd=True -> beta={L_TSD_BETA}')
    log_print(f'# Shared weight (overwritten): {keep_best}')
    log_print(f'{"#"*70}')

    grid = list(itertools.product([False, True], repeat=3))

    for case_idx, (ema_lora, truncation, l_tsd) in enumerate(grid, start=1):
        cleanup_resources(device)
        config = get_base_config()
        config.update({
            'n_epochs': n_epochs,
            # 1) ema_lora
            'ema_mode': bool(ema_lora),
            # 2) truncation
            'truncation': bool(truncation),
            # 3) l_tsd (hardcoded beta when True)
            'compute_l_tsd': bool(l_tsd),
            'beta1': L_TSD_BETA if l_tsd else 0.0,
        })
        run_one_case(case_idx, ema_lora, truncation, l_tsd, config, runner)

    log_print(f'\n{"="*70}')
    log_print('ALL 8 ABLATION CASES COMPLETED! (OTU -> OVATUS)')
    # Old log (commented out): 'ALL 16 ABLATION CASES COMPLETED! (OVATUS -> OTU)'
    log_print('Results: results.json (unique name per case)')
    log_print(f'Weights: single overwritten file {keep_best}')
    log_print(f'{"="*70}')


if __name__ == '__main__':
    main()
