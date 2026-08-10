"""
SR-SAM: Subspace Regularization for Domain Generalization of Segment
Anything Model

Paper: "Subspace Regularization for Domain Generalization of Segment
       Anything Model" (MICCAI 2025)
Repo:  https://github.com/xjiangmed/SR-SAM

Single-source DG: trains on ONE source domain, evaluates on the two
held-out domains (leave-one-out benchmark, one source per run).
"""
import os
import sys
import torch

from utils.seed import set_seed
from utils.sequential_training import SequentialDomainRunner, cleanup_resources
from DG.SR_SAM.data import get_sr_sam_loaders
from DG.SR_SAM.model import build_sr_sam
from DG.SR_SAM.train import train_sr_sam
from DG.SR_SAM.test import test_sr_sam_on_target

set_seed()

DATASETS = ['OTU', 'OVATUS', 'USOVA']


def run_single_source(source, targets, cfg, runner):
    cfg = dict(cfg)
    cfg['model_dir'] = 'weights/sr_sam/'
    cfg['prefix'] = f'vit_b_sr_sam_s_{source}'

    print(f'\n{"="*70}')
    print('TRAINING')
    print(f'Source: {source}')
    print(f'Targets: {", ".join(targets)}')
    print(f'{"="*70}')
    sys.stdout.flush()

    with runner.domain_context():
        device = cfg['device']

        model = build_sr_sam(
            checkpoint=cfg['checkpoint'], model_type='vit_b',
            image_size=cfg['image_size'], num_classes=cfg['num_classes'],
            rank=cfg['rank'], ema_mode=cfg['ema_mode'],
            truncation_size=cfg['truncation_size'],
            truncation=cfg['truncation'],
        ).to(device)

        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'\nModel params: {params/1e6:.3f}M (trainable: {trainable/1e6:.3f}M)')
        sys.stdout.flush()

        if cfg['phase'] == 'train':
            train_loader, val_loader = get_sr_sam_loaders(
                source, image_size=cfg['image_size'],
                batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                pin_memory=cfg['pin_memory'],
            )
            runner.register_loaders(train_loader)
            runner.register_loaders(val_loader)

            train_sr_sam(model, train_loader, val_loader, device, cfg)
            runner.destroy_loaders()

        # Load best checkpoint
        best_path = os.path.join(cfg['model_dir'], f'{cfg["prefix"]}_best.pth')
        if os.path.exists(best_path):
            print(f'\nBest weight is loaded: {best_path}')
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        else:
            print(f'\nNo best checkpoint found at {best_path}')
            return

        # Target evaluation (the held-out domains)
        for target in targets:
            print(f'\n{"="*70}')
            print('TESTING')
            print(f'Source: {source}')
            print(f'Target: {target}')
            test_sr_sam_on_target(
                model, target, device,
                image_size=cfg['image_size'], batch_size=cfg['batch_size'],
                num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'],
                source_name=source, write_results=True,
            )
        print(f'{"="*70}')


CONFIG = {
    'image_size': 256,
    'batch_size': 8,
    'num_workers': 4,
    'pin_memory': True,
    'device': 'cuda:1' if torch.cuda.is_available() else 'cpu',  # GPU0 busy; change if needed

    'n_epochs': 160,
    'base_lr': 0.0005,          # paper Sec. 3 (initial learning rate)
    'warmup_period': 250,       # paper Sec. 3 (warm-up iterations)

    'num_classes': 1,
    'rank': 64,                 # paper Sec. 3 (LoRA rank)
    'ema_mode': True,
    'ema_rate': 0.999,          # paper Sec. 2.3 (EMA rate alpha)
    'kd_weight': 1e-7,          # paper Sec. 3 (lambda, polyp) / repo --kd_weight
    'truncation': True,
    'truncation_size': 96,      # paper Sec. 3 (s, Table 4)
    'truncation_period': 4,     # paper Sec. 3 (every 4 epochs)
    'dash_warm': 300,           # repo run_CVC-ClinicDB.sh --Dash_warm 300

    'phase': 'train',

    'checkpoint': 'weights/sam/sam_vit_b_01ec64.pth',
}

SINGLE_SOURCE_RUNS = [
    ('OTU', ['OVATUS', 'USOVA']),
    ('OVATUS', ['OTU', 'USOVA']),
    ('USOVA', ['OTU', 'OVATUS']),
]


if __name__ == '__main__':
    runs = SINGLE_SOURCE_RUNS
    if os.environ.get('SR_SAM_RUNS'):
        runs = [r for r in SINGLE_SOURCE_RUNS
                if r[0] in os.environ['SR_SAM_RUNS'].split(',')]

    n_epochs = CONFIG['n_epochs']
    if os.environ.get('SR_SAM_EPOCHS'):
        n_epochs = int(os.environ['SR_SAM_EPOCHS'])
    CONFIG['n_epochs'] = n_epochs

    device = torch.device(CONFIG['device'])
    runner = SequentialDomainRunner(device=device)

    for source, targets in runs:
        cleanup_resources(device)
        run_single_source(source, targets, CONFIG, runner)
