"""DSR-SAM2 entry point.

Dual-branch DSR-SAM: unchanged SAM ViT encoder plus an ImageNet1k-v1
ResNet18 CNN branch (layer4-8 reach 16x16x768), with 8 Grassmann-weighted
Cross-Branch Attention blocks coupling ViT blocks 8-11 to ResBlocks 5-8.
LoRA/EMA/subspace truncation and the L_tsd loss are inherited from DSR-SAM.

Single-source DG: trains on ONE source domain, evaluates on the two
held-out domains (leave-one-out benchmark, one source per run).
Results go to results.json as ``*_dsr_sam2_*`` and weights to
``weights/dsr_sam2/``.
"""
import os
import sys
import torch

from utils.seed import set_seed
from utils.env import get_device
from utils.sequential_training import SequentialDomainRunner, cleanup_resources
from DG.dsr_sam2.data import get_dsr_sam2_loaders
from DG.dsr_sam2.model import build_dsr_sam2
from DG.dsr_sam2.train import train_dsr_sam2
from DG.dsr_sam2.test import test_dsr_sam2_on_target

set_seed()



def run_single_source(source, targets, cfg, runner):
    cfg = dict(cfg)
    # Phase 2 (reproducibility): reseed per source so a source's run no longer
    # depends on RNG consumed by previously run sources/cases.
    set_seed(cfg.get('seed', 42))
    cfg['model_dir'] = 'weights/dsr_sam2/'
    # `run_name` lets variants (e.g. dsr_sam2_noLtsd) save to distinct weights.
    cfg['prefix'] = f'{cfg["model_type"]}_{cfg.get("run_name", "dsr_sam2")}_s_{source}'
    cfg['source'] = source

    print(f'\n{"="*70}')
    print('TRAINING')
    print(f'Backbone: {cfg["model_type"]}')
    print(f'Source: {source}')
    print(f'Targets: {", ".join(targets)}')
    print(f'{"="*70}')
    sys.stdout.flush()

    with runner.domain_context():
        device = cfg['device']

        model = build_dsr_sam2(
            checkpoint=cfg['checkpoint'], model_type=cfg['model_type'],
            image_size=cfg['image_size'], num_classes=cfg['num_classes'],
            rank=cfg['rank'], ema_mode=cfg['ema_mode'],
            truncation_size=cfg['truncation_size'],
            top_s_change_rates=cfg['top_s_change_rates'],
            d_delta=cfg['d_delta'],
            alpha=cfg.get('alpha', 0.1),
            cnn_pretrained=cfg.get('cnn_pretrained', True),
            cba_last_n=cfg.get('cba_last_n', 4),
            # grassmann_grad=False (default) -> detach the Grassmann gate.
            grassmann_detach=not cfg.get('grassmann_grad', False),
            truncate_before_compute_grassmann=cfg.get(
                'truncate_before_compute_grassmann', 0),
            truncate_ref=cfg.get('truncate_ref', False),
        ).to(device)

        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'\nModel params: {params/1e6:.3f}M (trainable: {trainable/1e6:.3f}M)')
        sys.stdout.flush()

        if cfg['phase'] == 'train':
            train_loader, val_loader = get_dsr_sam2_loaders(
                source, image_size=cfg['image_size'],
                batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                pin_memory=cfg['pin_memory'],
            )
            runner.register_loaders(train_loader)
            runner.register_loaders(val_loader)

            train_dsr_sam2(model, train_loader, val_loader, device, cfg)
            runner.destroy_loaders()

        # Load best and last checkpoints and evaluate both on the targets.
        best_path = os.path.join(cfg['model_dir'], f'{cfg["prefix"]}_best.pth')
        last_path = os.path.join(cfg['model_dir'], f'{cfg["prefix"]}_last.pth')

        def evaluate(weight_path, weight_tag):
            if not os.path.exists(weight_path):
                print(f'\nNo {weight_tag} checkpoint found at {weight_path}')
                return
            print(f'\n{weight_tag.capitalize()} weight is loaded: {weight_path}')
            sys.stdout.flush()
            model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
            for target in targets:
                print(f'\n{"="*70}')
                print('TESTING')
                print(f'Source: {source}')
                print(f'Target: {target}')
                print(f'Weight: {weight_tag}')
                test_dsr_sam2_on_target(
                    model, target, device,
                    image_size=cfg['image_size'], batch_size=cfg['batch_size'],
                    num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'],
                    source_name=source, model_type=cfg['model_type'],
                    write_results=True, weight_tag=weight_tag,
                    result_tag=cfg.get('result_tag'),
                )
            print(f'{"="*70}')

        evaluate(best_path, 'best')
        if cfg.get('save_last_epoch', False):
            evaluate(last_path, 'last')


CONFIG = {
    'image_size': 256,
    'batch_size': 8,
    'num_workers': 4,
    'pin_memory': True,
    'device': get_device(),  # CUDA_DEVICE from y_DG/.env

    'seed': 42,                 # RNG seed (reset per source for reproducibility)

    'n_epochs': 100,
    'base_lr': 0.0005,          # paper Sec. 3 (initial learning rate)
    'warmup_period': 250,       # paper Sec. 3 (warm-up iterations)
    'patience': 20,              # early stopping on val loss (0 = disabled)

    'num_classes': 1,
    'rank': 64,                 # paper Sec. 3 (LoRA rank)
    'ema_mode': False,
    'ema_rate': 0.999,          # paper Sec. 2.3 (EMA rate alpha)
    'kd_weight': 1e-7,          # paper Sec. 3 (lambda, polyp) / repo --kd_weight
    'beta1': 0.1,                # L_tsd coefficient
    'truncation': False,
    'truncation_size': 96,      # paper Sec. 3 (s, Table 4)
    'top_s_change_rates': 96,   # number of top change rates contributing to L_tsd
    'truncation_period': 4,     # paper Sec. 3 (every 4 epochs)
    'dash_warm': 300,           # repo run_CVC-ClinicDB.sh --Dash_warm 300
    'compute_l_tsd': False,      # whether to compute L_tsd loss
    'alpha': 0.1,              # coefficient of the L_tsd log-squash term
    'd_delta': False,           # L_tsd numerator: keep only change rates that
                                # grew since the previous iteration (dDelta)

    'experiment_process_save_epochs': False,  # off here; run_sota enables it (even epochs 2,4,...,20)
    'save_last_epoch': False,  # whether to save weights at the last epoch

    'phase': 'train',

    'model_type': 'vit_b',
    'checkpoint': 'weights/sam/sam_vit_b_01ec64.pth',

    # ---- DSR-SAM2 dual-branch architecture ----
    'cnn_pretrained': True,   # ResNet18 IMAGENET1K_V1 through layer3
    'cba_last_n': 4,          # last 4 ViT blocks x 2 CBA each = 8 CBA blocks
    'grassmann_grad': False,  # False -> G_score detached (fixed per-channel gate)

    # s = number of smallest singular values/vectors to remove before the
    # Grassmann score (e.g. 96 -> zero the 96 lowest sigma). 0 = disabled.
    'truncate_before_compute_grassmann': 0,
    # Only valid when truncate_before_compute_grassmann > 0:
    # True  -> truncated features are only the G_score reference (fusion
    #          keeps the original f_vit/f_cnn);
    # False -> the truncated features are also fused.
    'truncate_ref': False,
}

DATASETS = ['OTU', 'OVATUS', 'USOVA']

SINGLE_SOURCE_RUNS = [
    ('OTU', ['OVATUS', 'USOVA']),
    ('OVATUS', ['OTU', 'USOVA']),
    ('USOVA', ['OTU', 'OVATUS']),
]


if __name__ == '__main__':
    runs = SINGLE_SOURCE_RUNS
    if os.environ.get('DSR_SAM_RUNS'):
        runs = [r for r in SINGLE_SOURCE_RUNS
                if r[0] in os.environ['DSR_SAM_RUNS'].split(',')]

    n_epochs = CONFIG['n_epochs']
    if os.environ.get('DSR_SAM_EPOCHS'):
        n_epochs = int(os.environ['DSR_SAM_EPOCHS'])
    CONFIG['n_epochs'] = n_epochs

    device = torch.device(CONFIG['device'])
    runner = SequentialDomainRunner(device=device)

    for source, targets in runs:
        cleanup_resources(device)
        run_single_source(source, targets, CONFIG, runner)
        
