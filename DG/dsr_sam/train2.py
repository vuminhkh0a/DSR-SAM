"""
Shared single-file training for the DSR-SAM analyses (main_a.py / main_h.py).

Memory policy: every case of an analysis overwrites ONE checkpoint, so at most
one weight per analysis exists on disk:

    weights/dsr_sam/analysis/vit_b_ablation_overwrite_best.pth   (main_a)
    weights/dsr_sam/analysis/vit_b_hparam_overwrite_best.pth     (main_h)

No per-case `*_last.pth` files and no experiment snapshots are written, and
any stale sibling of the same run is removed. Per-case identity is kept ONLY
in results.json (unique name per case), never in the weight filename.

The analysis checkpoints live in `weights/dsr_sam/analysis/` so the per-source
run_sota DSR-SAM weights (`weights/dsr_sam/vit_b_dsr_sam_s_<src>_best.pth`)
are never touched.
"""
import os

from DG.dsr_sam.train import train_dsr_sam as _train_dsr_sam_base

# Analysis checkpoints live in a sub-dir (protects the run_sota weights).
OVERWRITE_DIR = os.path.join('weights', 'dsr_sam', 'analysis')
DEFAULT_RUN_NAME = 'analysis_overwrite'


def shared_prefix(model_type='vit_b', run_name=DEFAULT_RUN_NAME):
    """Shared overwrite prefix, carrying the backbone and the analysis name."""
    return f'{model_type}_{run_name}'


def shared_best(model_type='vit_b', run_name=DEFAULT_RUN_NAME,
                overwrite_dir=OVERWRITE_DIR):
    """Shared overwrite weight path for one backbone + analysis."""
    return os.path.join(overwrite_dir, f'{shared_prefix(model_type, run_name)}_best.pth')


def train_dsr_sam(model, train_loader, val_loader, device, cfg):
    """Train DSR-SAM, always (over)writing ONE shared analysis weight file.

    Forces cfg['model_dir']/cfg['prefix'] to the shared overwrite location,
    disables all secondary saves, then delegates the actual loop to
    DG.dsr_sam.train.train_dsr_sam. Stale siblings of the same run are removed
    so at most 1 weight of this run remains on disk.
    """
    cfg = dict(cfg)
    model_type = cfg.get('model_type', 'vit_b')
    run_name = cfg.get('overwrite_name', DEFAULT_RUN_NAME)
    overwrite_dir = cfg.get('overwrite_dir', OVERWRITE_DIR)
    prefix = shared_prefix(model_type, run_name)
    best_path = shared_best(model_type, run_name, overwrite_dir)

    # Strict overwrite: every case writes to the exact same file.
    cfg['model_dir'] = overwrite_dir
    cfg['prefix'] = prefix
    # Never keep secondary copies (saves disk).
    cfg['save_last_epoch'] = False
    cfg['experiment_process_save_epochs'] = False

    os.makedirs(overwrite_dir, exist_ok=True)

    best_val_loss = _train_dsr_sam_base(model, train_loader, val_loader, device, cfg)

    # Defensive cleanup: keep only this run's single weight.
    try:
        for f in os.listdir(overwrite_dir):
            fp = os.path.join(overwrite_dir, f)
            if (os.path.isfile(fp) and f.endswith('.pth')
                    and f.startswith(prefix) and fp != best_path):
                os.remove(fp)
    except OSError:
        pass

    return best_val_loss
