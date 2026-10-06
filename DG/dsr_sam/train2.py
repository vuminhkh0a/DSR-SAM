"""
Train module for DSR-SAM 8-case ablation (main2.py).

Direction: OVATUS (train) -> OTU (test).
# Old direction (commented out): OTU (train) -> OVATUS (test).

Memory constraint:
  - ALL 8 runs share ONE weight file (per backbone, default vit_b):
        weights/dsr_sam/vit_b_dsr_sam_overwrite_best.pth
    Every iteration strictly overwrites this exact file. No per-case
    checkpoint files are kept (no `*_last.pth`, no experiment snapshots,
    stale `*_best.pth` files are deleted).
  - Results identity is kept ONLY in results.json via a unique
    `weight_tag` per case, never in the weight filename.

Ablation mapping (set by main2.py per case):
  - ema_lora  -> cfg['ema_mode']
  - truncation-> cfg['truncation']
  - l_tsd=True  -> cfg['compute_l_tsd']=True,  cfg['beta1']=0.1 (hardcoded)
  - l_tsd=False -> cfg['compute_l_tsd']=False, cfg['beta1']=0.0
"""
import os

from DG.dsr_sam.train import train_dsr_sam as _train_dsr_sam_base

# ---- Single overwrite weight (1 file per backbone for ALL 16 cases) ----
OVERWRITE_DIR = 'weights/dsr_sam'


def shared_prefix(model_type='vit_b'):
    """Shared overwrite prefix, carrying the ViT size (e.g. vit_b)."""
    return f'{model_type}_dsr_sam_overwrite'


def shared_best(model_type='vit_b'):
    """Shared overwrite weight path for one backbone."""
    return os.path.join(OVERWRITE_DIR, f'{shared_prefix(model_type)}_best.pth')


# Back-compat default (vit_b).
OVERWRITE_PREFIX = shared_prefix()
OVERWRITE_BEST = shared_best()


def get_overwrite_paths(model_type='vit_b'):
    """Return (model_dir, prefix, best_path) for the single shared weight."""
    return OVERWRITE_DIR, shared_prefix(model_type), shared_best(model_type)


def beta_tag(beta):
    """Compact beta tag for results names: 0.1->'01', 0.5->'05', 1.0->'10'."""
    return str(beta).replace('.', '')


def train_dsr_sam(model, train_loader, val_loader, device, cfg):
    """Train DSR-SAM, always (over)writing the single shared weight file.

    Forces cfg['model_dir']/cfg['prefix'] to the shared overwrite location,
    disables all secondary saves, then delegates the actual loop to
    DG.dsr_sam.train.train_dsr_sam. Old per-study `*_best.pth` / `*_last.pth`
    files are removed so only 1 weight remains on disk.
    """
    cfg = dict(cfg)
    # Strict overwrite: every case writes to the exact same file.
    best_path = shared_best(cfg.get('model_type', 'vit_b'))
    cfg['model_dir'] = OVERWRITE_DIR
    cfg['prefix'] = shared_prefix(cfg.get('model_type', 'vit_b'))
    # Never keep secondary copies in ablation mode (saves disk).
    cfg['save_last_epoch'] = False
    cfg['experiment_process_save_epochs'] = False

    os.makedirs(OVERWRITE_DIR, exist_ok=True)

    best_val_loss = _train_dsr_sam_base(model, train_loader, val_loader, device, cfg)

    # Defensive cleanup: remove any non-shared weight that may have slipped
    # through (e.g. a `_last.pth` written by other code paths).
    try:
        for f in os.listdir(OVERWRITE_DIR):
            fp = os.path.join(OVERWRITE_DIR, f)
            if os.path.isfile(fp) and fp != best_path and f.endswith('.pth'):
                os.remove(fp)
    except OSError:
        pass

    return best_val_loss
