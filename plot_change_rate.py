"""Change-rate distribution projected onto the PRETRAINED SAM weight (vit_b).

Standalone version of the `demo.ipynb` cells (works without editing the notebook).

Projects each configuration's LoRA delta onto the SVD of the **pretrained** SAM
`qkv.weight` and plots the sorted change rates per block:

  - DSR-SAM (compute_l_tsd True / False): student LoRA  deltaW = B @ A
  - SR-SAM:                              EMA LoRA      deltaW = B_ema @ A_ema

  change_rate_i = |u_i^T deltaW v_i| / (sigma_i + eps)

Checkpoints are the per-epoch experiment snapshots written by
`experiment_process_save_epochs` (even epochs 2..20):

  weights/dsr_sam/vit_b_ltsd_s_<src>_experiment_epoch_<N>.pth
  weights/dsr_sam/vit_b_noLtsd_s_<src>_experiment_epoch_<N>.pth
  weights/sr_sam/vit_b_sr_sam_s_<src>_experiment_epoch_<N>.pth

Usage:
  python plot_change_rate.py                 # source OTU, blocks 1,3,5,7,9,11
  SOURCE=OVATUS python plot_change_rate.py
"""
import gc
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch

# ---------------------------------------------------------------------------
MODEL_TYPE = 'vit_b'
SOURCE = os.environ.get('SOURCE', 'OTU')
EPS = 1e-8
TOP_K = 100
BLOCKS = [1, 3, 5, 7, 9, 11]             # vit_b has 12 encoder blocks (0-11)
EPOCHS_TO_CHECK = list(range(2, 21, 2))  # experiment snapshots: 2,4,...,20
PRETRAINED_CKPT = 'weights/sam/sam_vit_b_01ec64.pth'
SAVE_DIR = 'saved_plots'

try:
    from utils.env import get_device
    DEVICE = get_device()
except Exception as exc:  # CUDA_DEVICE / CUDA_VISIBLE_DEVICES mismatch, etc.
    print(f'[warn] get_device() failed ({exc}); falling back to CPU.')
    DEVICE = 'cpu'


def main():
    print(f'Device: {DEVICE}')
    pretrained_sd = torch.load(PRETRAINED_CKPT, map_location='cpu', weights_only=True)
    print('Loaded pretrained:', PRETRAINED_CKPT)

    svd_cache = {}

    def pretrained_svd(block_id, branch):
        key = (block_id, branch)
        if key not in svd_cache:
            W = pretrained_sd[f'image_encoder.blocks.{block_id}.attn.qkv.weight'].float()
            dim = W.shape[1]
            rows = slice(0, dim) if branch == 'q' else slice(-dim, None)
            U, S, Vh = torch.linalg.svd(W[rows].to(DEVICE), full_matrices=False)
            svd_cache[key] = (U, S, Vh)
        return svd_cache[key]

    def change_rates(ckpt_sd, block_id, branch, kind='student'):
        """Sorted (descending) change rates on the PRETRAINED-weight SVD."""
        U, S, Vh = pretrained_svd(block_id, branch)
        prefix = f'sam.image_encoder.blocks.{block_id}.attn.qkv.'
        tag = 'ema_' if kind == 'ema' else ''
        A = ckpt_sd[prefix + f'{tag}linear_a_{branch}.weight'].float().to(DEVICE)
        B = ckpt_sd[prefix + f'{tag}linear_b_{branch}.weight'].float().to(DEVICE)
        proj = torch.diag(U.T @ (B @ A) @ Vh.T)
        cr = (proj.abs() / (S.abs() + EPS)).cpu().numpy()
        return np.sort(cr)[::-1]

    configs = [
        ('DSR-SAM L_tsd=True',
         f'weights/dsr_sam/vit_b_ltsd_s_{SOURCE}_experiment_epoch_{{ep}}.pth', 'student'),
        ('DSR-SAM L_tsd=False',
         f'weights/dsr_sam/vit_b_noLtsd_s_{SOURCE}_experiment_epoch_{{ep}}.pth', 'student'),
        ('SR-SAM (EMA LoRA)',
         f'weights/sr_sam/vit_b_sr_sam_s_{SOURCE}_experiment_epoch_{{ep}}.pth', 'ema'),
    ]

    config_epochs = []
    for label, path_tpl, kind in configs:
        eps_ok = [ep for ep in EPOCHS_TO_CHECK if os.path.exists(path_tpl.format(ep=ep))]
        config_epochs.append(eps_ok)
        print(f'{label:22s} [{kind:7s}]: {len(eps_ok):2d} snapshots -> {eps_ok}')

    colors = {'q': 'tab:blue', 'v': 'tab:orange'}
    os.makedirs(SAVE_DIR, exist_ok=True)

    for b in BLOCKS:
        n_cols = max((len(e) for e in config_epochs), default=0)
        if n_cols == 0:
            print(f'Block {b}: no experiment snapshots found; run DG.sota.run_sota first.')
            continue

        fig, axes = plt.subplots(len(configs), n_cols,
                                 figsize=(3.2 * n_cols, 2.8 * len(configs)),
                                 squeeze=False)
        for r, (label, path_tpl, kind) in enumerate(configs):
            eps_ok = config_epochs[r]
            for c in range(n_cols):
                ax = axes[r, c]
                if c >= len(eps_ok):
                    ax.axis('off')
                    continue
                ep = eps_ok[c]
                ckpt_sd = torch.load(path_tpl.format(ep=ep),
                                     map_location='cpu', weights_only=True)
                for branch in ('q', 'v'):
                    cr = change_rates(ckpt_sd, b, branch, kind)[:TOP_K]
                    ax.bar(np.arange(len(cr)), cr, width=1.0, color=colors[branch],
                           alpha=0.6, edgecolor='none', label=branch.upper())
                del ckpt_sd
                if 'cuda' in str(DEVICE):
                    torch.cuda.empty_cache()
                ax.set_title(f'{label} | epoch {ep}', fontsize=9)
                ax.set_xlabel('Rank (sorted)', fontsize=8)
                ax.set_ylabel('Change rate', fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.3, axis='y')

        legend_handles = [Patch(facecolor=colors['q'], alpha=0.6, label='Q'),
                          Patch(facecolor=colors['v'], alpha=0.6, label='V')]
        fig.legend(handles=legend_handles, loc='upper center', ncol=2,
                   bbox_to_anchor=(0.5, 1.0), fontsize=10)
        fig.suptitle(
            f'Change-rate distribution (pretrained-weight projection) | {MODEL_TYPE} | '
            f'block {b} | source {SOURCE} | top {TOP_K}',
            fontsize=13, fontweight='bold', y=1.02)

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        save_path = os.path.join(
            SAVE_DIR,
            f'ChangeRate_pretrained_{MODEL_TYPE}_{SOURCE}_block{b}_Top{TOP_K}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print('Saved:', save_path)
        plt.close(fig)
        gc.collect()


if __name__ == '__main__':
    main()
