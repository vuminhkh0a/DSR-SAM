"""
DSR-SAM 16-case ablation runner (OTU -> OVATUS only).
# Previous direction (commented out): OVATUS -> OTU only.

Grid search: 2^4 = 16 True/False combinations of:
  - ema_lora   (bool) -> cfg['ema_mode']
  - truncation (bool) -> cfg['truncation']
  - l_tsd      (bool) -> True:  compute_l_tsd=True,  beta1=0.1 (hardcoded)
                             False: compute_l_tsd=False, beta1=0.0
  - freeze_a   (bool) -> True:  is_freeze_A_after_N_epoch=True,
                                 N_e=2, freeze_A_after_N_epoch=n_epochs-2
                             False: is_freeze_A_after_N_epoch=False,
                                 N_e=0, freeze_A_after_N_epoch=0

Scope: ONLY the OTU -> OVATUS direction. Each case trains on OTU,
then tests on OVATUS immediately (before the next case overwrites
the shared weight).
# Old scope (commented out): ONLY the OVATUS -> OTU direction. Each case
# trained on OVATUS, then tested on OTU immediately.

Weight policy (save disk):
  - ONE shared file, overwritten every run:
        weights/dsr_sam/vit_h_dsr_sam_overwrite_best.pth
    (see DG/DSR_SAM/train2.py). No per-case weight files are kept.
Results policy:
  - results.json gets a DIFFERENT name per case, e.g.
      dsr_sam_abl_case03_emaT_truncF_ltsdT_frzF_sOTU_tOVATUS
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
from utils.sequential_training import SequentialDomainRunner, cleanup_resources
from utils.metrics import save_results
from DG.DSR_SAM.data import get_dsr_sam_loaders
from DG.DSR_SAM.model import build_dsr_sam
from DG.DSR_SAM.train2 import (
    train_dsr_sam,
    OVERWRITE_DIR,
    OVERWRITE_PREFIX,
    OVERWRITE_BEST,
)
from DG.DSR_SAM.test import test_dsr_sam_on_target

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
FREEZE_N_E = 2


def get_base_config():
    """Base configuration for all 16 cases."""
    return {
        'image_size': 256,
        'batch_size': 8,
        'num_workers': 4,
        'pin_memory': True,
        'device': 'cuda:1' if torch.cuda.is_available() else 'cpu',

        'n_epochs': 50,
        'base_lr': 0.0005,
        'warmup_period': 250,

        'num_classes': 1,
        'rank': 64,
        'lora_A_init': 'orthogonal',  # 'kaiming' (default) or 'orthogonal'; env DSR_SAM_LORA_A_INIT overrides
        'ema_mode': False,   # overwritten per case by ema_lora
        'ema_rate': 0.999,
        'kd_weight': 1e-7,
        'beta1': 0.0,        # overwritten per case by l_tsd
        'truncation': False,  # overwritten per case
        'truncation_size': 96,
        'truncation_period': 4,
        'dash_warm': 300,
        'freeze_A_after_N_epoch': 2,  # overwritten per case by freeze_a
        'N_e': 2,
        'is_freeze_A_after_N_epoch': False,  # overwritten per case
        'compute_l_tsd': False,  # overwritten per case by l_tsd
        'patience': 10,

        'experiment_process_save_epochs': False,
        'save_last_epoch': False,  # never keep a 'last' copy in ablation mode

        'phase': 'train',
        'model_type': 'vit_h',
        'checkpoint': 'weights/sam/sam_vit_h_4b8939.pth',
    }


def flag_tag(value):
    """Compact True/False tag for results names: True->'T', False->'F'."""
    return 'T' if value else 'F'


def case_base_name(case_idx, ema_lora, truncation, l_tsd, freeze_a):
    """Unique results.json base name for one ablation case."""
    return (f'dsr_sam_abl_case{case_idx:02d}_'
            f'ema{flag_tag(ema_lora)}_'
            f'trunc{flag_tag(truncation)}_'
            f'ltsd{flag_tag(l_tsd)}_'
            f'frz{flag_tag(freeze_a)}')


def run_one_case(case_idx, ema_lora, truncation, l_tsd, freeze_a, config, runner):
    """Train ONE ablation case (OTU) into the shared weight, then test OVATUS.
    # Old direction (commented out): Train ONE ablation case (OVATUS) into the
    # shared weight, then test OTU.

    - Trains into OVERWRITE_BEST (overwritten every call).
    - Right after training, loads that weight and tests OVATUS:
        * prints metrics to log.txt immediately,
        * appends `f'{base}_sOTU_tOVATUS'` to results.json immediately
          (before the next case overwrites the weight).
    # Old naming (commented out): `f'{base}_sOVATUS_tOTU'`.
    """
    os.makedirs(OVERWRITE_DIR, exist_ok=True)

    config = dict(config)
    # Constant prefix -> train.py always saves to the SAME single file.
    config['model_dir'] = OVERWRITE_DIR
    config['prefix'] = OVERWRITE_PREFIX
    config['best_weight_path'] = OVERWRITE_BEST
    best_path = OVERWRITE_BEST

    base_name = case_base_name(case_idx, ema_lora, truncation, l_tsd, freeze_a)

    log_print(f'\n{"="*70}')
    log_print(f'ABLATION CASE {case_idx:02d}/16: {base_name}')
    log_print(f'Source: {SOURCE} | Target: {TARGET}')
    log_print(f'Config: ema_lora={ema_lora}, truncation={truncation}, '
              f'l_tsd={l_tsd} (beta1={config["beta1"]}), '
              f'freeze_a={freeze_a} (N_e={config.get("N_e")}, '
              f'threshold={config["freeze_A_after_N_epoch"]}), '
              f'lora_A_init={config.get("lora_A_init", "kaiming")}')
    log_print(f'Weight file (shared, overwritten): {best_path}')
    log_print(f'{"="*70}')

    with runner.domain_context():
        device = config['device']
        model = build_dsr_sam(
            checkpoint=config['checkpoint'], model_type=config['model_type'],
            image_size=config['image_size'], num_classes=config['num_classes'],
            rank=config['rank'], ema_mode=config['ema_mode'],
            truncation_size=config['truncation_size'],
            lora_A_init=config.get('lora_A_init', 'kaiming'),
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
    lora_A_init = os.environ.get('DSR_SAM_LORA_A_INIT', 'kaiming')
    assert lora_A_init in ('kaiming', 'orthogonal'), \
        f"DSR_SAM_LORA_A_INIT must be 'kaiming' or 'orthogonal', got {lora_A_init!r}"

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    runner = SequentialDomainRunner(device=device)

    # Remove stale per-study weights; only the single shared file is kept.
    try:
        os.makedirs(OVERWRITE_DIR, exist_ok=True)
        for f in os.listdir(OVERWRITE_DIR):
            fp = os.path.join(OVERWRITE_DIR, f)
            if (fp != OVERWRITE_BEST and f.endswith('.pth')
                    and os.path.isfile(fp)):
                os.remove(fp)
                log_print(f'Removed stale per-study weight: {fp}')
    except OSError:
        pass

    # ============================================================
    # 16-case grid: all True/False combos of
    # (ema_lora, truncation, l_tsd, freeze_a), OTU -> OVATUS only.
    # Old direction (commented out): OVATUS -> OTU only.
    # ============================================================
    log_print(f'\n{"#"*70}')
    log_print('# DSR-SAM 16-CASE ABLATION: OTU -> OVATUS')
    # Old log (commented out): '# DSR-SAM 16-CASE ABLATION: OVATUS -> OTU'
    log_print(f'# l_tsd=True -> beta={L_TSD_BETA} | freeze_a=True -> N_e={FREEZE_N_E} | lora_A_init={lora_A_init}')
    log_print(f'# Shared weight (overwritten): {OVERWRITE_BEST}')
    log_print(f'{"#"*70}')

    grid = list(itertools.product([False, True], repeat=4))

    for case_idx, (ema_lora, truncation, l_tsd, freeze_a) in enumerate(grid, start=1):
        cleanup_resources(device)
        config = get_base_config()
        config.update({
            'n_epochs': n_epochs,
            'lora_A_init': lora_A_init,
            # 1) ema_lora
            'ema_mode': bool(ema_lora),
            # 2) truncation
            'truncation': bool(truncation),
            # 3) l_tsd (hardcoded beta when True)
            'compute_l_tsd': bool(l_tsd),
            'beta1': L_TSD_BETA if l_tsd else 0.0,
            # 4) freeze_a (hardcoded N_e=2 when True)
            'is_freeze_A_after_N_epoch': bool(freeze_a),
            'N_e': FREEZE_N_E if freeze_a else 0,
            'freeze_A_after_N_epoch': max(0, n_epochs - FREEZE_N_E) if freeze_a else 0,
        })
        run_one_case(case_idx, ema_lora, truncation, l_tsd, freeze_a, config, runner)

    log_print(f'\n{"="*70}')
    log_print('ALL 16 ABLATION CASES COMPLETED! (OTU -> OVATUS)')
    # Old log (commented out): 'ALL 16 ABLATION CASES COMPLETED! (OVATUS -> OTU)'
    log_print('Results: results.json (unique name per case)')
    log_print(f'Weights: single overwritten file {OVERWRITE_BEST}')
    log_print(f'{"="*70}')


if __name__ == '__main__':
    main()
