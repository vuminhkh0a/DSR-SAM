"""
MA-SAM: Modality-agnostic SAM Adaptation for 3D Medical Image Segmentation

Paper: "Modality-Agnostic SAM Adaptation for 3D Medical Image Segmentation"
       (arXiv:2309.08842)
Repo:  https://github.com/cchen-cc/MA-SAM

Multi-source DG: trains on TWO source domains, evaluates on the third
held-out domain (leave-one-out benchmark, one target per run).
"""
import os
import sys
import torch

from utils.seed import set_seed
from utils.sequential_training import SequentialDomainRunner, cleanup_resources
from DG.MASAM.data import get_masam_loaders
from DG.MASAM.model import build_masam
from DG.MASAM.train import train_masam
from DG.MASAM.test import test_masam_on_target

set_seed()

DATASETS = ['OTU', 'OVATUS', 'USOVA']


def run_multi_source(sources, target, cfg, runner):
    cfg = dict(cfg)
    cfg['model_dir'] = 'weights/masam/'
    cfg['prefix'] = f'vit_h_masam_s_{"_".join(sources)}'

    print(f'\n{"="*70}')
    print('TRAINING')
    print(f'Sources: {", ".join(sources)}')
    print(f'Target: {target}')
    print(f'{"="*70}')
    sys.stdout.flush()

    with runner.domain_context():
        device = cfg['device']

        model = build_masam(
            checkpoint=cfg['checkpoint'], model_type='vit_h',
            image_size=cfg['image_size'], num_classes=cfg['num_classes'],
            rank=cfg['rank'], scale=cfg['scale'],
        ).to(device)

        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'\nModel params: {params/1e6:.3f}M (trainable: {trainable/1e6:.3f}M)')
        sys.stdout.flush()

        if cfg['phase'] == 'train':
            train_loader, val_loader = get_masam_loaders(
                sources, image_size=cfg['image_size'],
                batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                pin_memory=cfg['pin_memory'],
            )
            runner.register_loaders(train_loader)
            runner.register_loaders(val_loader)

            train_masam(model, train_loader, val_loader, device, cfg)
            runner.destroy_loaders()

        # Load best checkpoint
        best_path = os.path.join(cfg['model_dir'], f'{cfg["prefix"]}_best.pth')
        if os.path.exists(best_path):
            print(f'\nBest weight is loaded: {best_path}')
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        else:
            print(f'\nNo best checkpoint found at {best_path}')
            return

        # Target evaluation (the held-out domain)
        print(f'\n{"="*70}')
        print('TESTING')
        print(f'Sources: {", ".join(sources)}')
        print(f'Target: {target}')
        test_masam_on_target(
            model, target, device,
            image_size=cfg['image_size'], batch_size=cfg['batch_size'],
            num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'],
            source_names=sources, write_results=True,
        )
        print(f'{"="*70}')


CONFIG = {
    'image_size': 256,
    'batch_size': 4,
    'num_workers': 4,
    'pin_memory': True,
    'device': 'cuda:1' if torch.cuda.is_available() else 'cpu',  # GPU0 busy; change if needed

    'n_epochs': 60,
    'base_lr': 0.0008,          # repo train.py --base_lr
    'warmup': True,
    'warmup_period': 250,       # repo trainer_bbox.py warmup iterations
    'lr_exp': 7,                # repo poly decay exponent

    'num_classes': 1,
    'rank': 32,                 # FacT rank r (repo Fact_tt_Sam default)
    'scale': 1.0,               # FacT scale s (paper/repo Fact_tt_Sam)

    'use_amp': True,
    'phase': 'train',

    'checkpoint': 'weights/sam/sam_vit_h_4b8939.pth',
}

MULTI_SOURCE_RUNS = [
    (['OTU', 'OVATUS'], 'USOVA'),
    (['OTU', 'USOVA'], 'OVATUS'),
    (['OVATUS', 'USOVA'], 'OTU'),
]


if __name__ == '__main__':
    runs = MULTI_SOURCE_RUNS
    if os.environ.get('MASAM_RUNS'):
        runs = [r for r in MULTI_SOURCE_RUNS
                if r[1] in os.environ['MASAM_RUNS'].split(',')]

    n_epochs = CONFIG['n_epochs']
    if os.environ.get('MASAM_EPOCHS'):
        n_epochs = int(os.environ['MASAM_EPOCHS'])
    CONFIG['n_epochs'] = n_epochs

    device = torch.device(CONFIG['device'])
    runner = SequentialDomainRunner(device=device)

    for sources, target in runs:
        cleanup_resources(device)
        run_multi_source(sources, target, CONFIG, runner)
