"""
MA-SAM: Modality-agnostic SAM Adaptation for 3D Medical Image Segmentation

Paper: https://arxiv.org/pdf/2309.08842
       "Modality-Agnostic SAM Adaptation for 3D Medical Image Segmentation"
Repo:  https://github.com/cchen-cc/MA-SAM

Single-source DG: trains on ONE source domain, evaluates on the two
held-out target domains (leave-one-out benchmark, 3 sources x 2 targets
= 6 cases), so the numbers are comparable with the other DG methods.
Note: the original paper trains on multiple source modalities; this
benchmark uses the shared single-source protocol.
"""
import os
import sys
import torch

from utils.seed import set_seed
from utils.env import get_device
from utils.sequential_training import SequentialDomainRunner, cleanup_resources
from utils.checkpoint import load_trainable
from DG.masam.data import get_masam_loaders
from DG.masam.model import build_masam
from DG.masam.train import train_masam
from DG.masam.test import test_masam_on_target

set_seed()

DATASETS = ['OTU', 'OVATUS', 'USOVA']


def run_single_source(source, targets, cfg, runner):
    sources = [source]
    cfg = dict(cfg)
    cfg['model_dir'] = 'weights/masam/'
    cfg['prefix'] = f'{cfg["model_type"]}_masam_s_{source}'
    if os.environ.get('MASAM_SMOKE') == '1':
        cfg['prefix'] += '_smoke'

    print(f'\n{"="*70}')
    print('TRAINING')
    print(f'Source: {source}')
    print(f'Targets: {", ".join(targets)}')
    print(f'{"="*70}')
    sys.stdout.flush()

    with runner.domain_context():
        device = cfg['device']

        model = build_masam(
            checkpoint=cfg['checkpoint'], model_type=cfg['model_type'],
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

        # Load best/last checkpoints and evaluate both on the targets.
        best_path = os.path.join(cfg['model_dir'], f'{cfg["prefix"]}_best.pth')
        last_path = os.path.join(cfg['model_dir'], f'{cfg["prefix"]}_last.pth')

        def evaluate(weight_path, weight_tag):
            if not os.path.exists(weight_path):
                print(f'\nNo {weight_tag} checkpoint found at {weight_path}')
                return
            print(f'\n{weight_tag.capitalize()} weight is loaded: {weight_path}')
            sys.stdout.flush()
            load_trainable(model, weight_path, device)
            for target in targets:
                print(f'\n{"="*70}')
                print('TESTING')
                print(f'Source: {source}')
                print(f'Target: {target}')
                print(f'Weight: {weight_tag}')
                test_masam_on_target(
                    model, target, device,
                    image_size=cfg['image_size'], batch_size=cfg['batch_size'],
                    num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'],
                    source_names=sources, write_results=cfg['write_results'],
                    model_type=cfg['model_type'], weight_tag=weight_tag,
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

    'n_epochs': 50,
    'patience': 5,              # early stopping on val loss (0 = disabled)
    'base_lr': 0.0008,          # repo train.py --base_lr
    'warmup': True,
    'warmup_period': 250,       # repo trainer_bbox.py warmup iterations
    'lr_exp': 7,                # repo poly decay exponent

    'num_classes': 1,
    'rank': 32,                 # FacT rank r (repo Fact_tt_Sam default)
    'scale': 1.0,               # FacT scale s (paper/repo Fact_tt_Sam)

    'use_amp': True,
    'save_last_epoch': False,    # keep the last-epoch snapshot besides best
    'write_results': True,       # False during smoke runs
    'phase': 'train',

    'model_type': 'vit_b',
    'checkpoint': 'weights/sam/sam_vit_b_01ec64.pth',
}

SINGLE_SOURCE_RUNS = [
    ('OTU', ['OVATUS', 'USOVA']),
    ('OVATUS', ['OTU', 'USOVA']),
    ('USOVA', ['OTU', 'OVATUS']),
]


if __name__ == '__main__':
    smoke = os.environ.get('MASAM_SMOKE') == '1'
    if smoke:
        runs = [('OTU', ['OVATUS'])]
        CONFIG['n_epochs'] = 1
        CONFIG['write_results'] = False
    else:
        runs = SINGLE_SOURCE_RUNS
        if os.environ.get('MASAM_RUNS'):
            runs = [r for r in SINGLE_SOURCE_RUNS
                    if r[0] in os.environ['MASAM_RUNS'].split(',')]
        if os.environ.get('MASAM_EPOCHS'):
            CONFIG['n_epochs'] = int(os.environ['MASAM_EPOCHS'])

    device = torch.device(CONFIG['device'])
    runner = SequentialDomainRunner(device=device)

    for source, targets in runs:
        cleanup_resources(device)
        run_single_source(source, targets, CONFIG, runner)
