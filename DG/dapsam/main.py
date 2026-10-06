"""
DAPSAM: Prompting Segment Anything Model with Domain-Adaptive Prototype
for Generalizable Medical Image Segmentation (MICCAI 2024,
arXiv:2409.12522).

Paper: https://arxiv.org/pdf/2409.12522
       "Prompting Segment Anything Model with Domain-Adaptive Prototype
        for Generalizable Medical Image Segmentation"
Repo:  https://github.com/wkklavis/DAPSAM (prostate/ branch)

Single-source DG: trains on ONE source domain, evaluates on the two
held-out target domains (leave-one-out benchmark, 3 sources x 2 targets
= 6 cases). No manual prompts: the prototype prompt generator (memory
bank N=256) creates the dense prompt automatically.

Note: the paper trains every run with ViT-B (Sec. 3.1), so the default
backbone here is vit_b (switchable to vit_h via model_type + checkpoint,
like the other DG methods). Image size follows the y_DG benchmark (256;
the repo uses 384 for prostate).
"""
import os
import sys
import torch

from utils.seed import set_seed
from utils.env import get_device
from utils.sequential_training import SequentialDomainRunner, cleanup_resources
from utils.checkpoint import load_trainable
from DG.dapsam.data import get_dapsam_loaders
from DG.dapsam.model import build_dapsam
from DG.dapsam.train import train_dapsam
from DG.dapsam.test import test_dapsam_on_target

set_seed()

DATASETS = ['OTU', 'OVATUS', 'USOVA']


def run_single_source(source, targets, cfg, runner):
    cfg = dict(cfg)
    cfg['model_dir'] = 'weights/dapsam/'
    cfg['prefix'] = f'{cfg["model_type"]}_dapsam_s_{source}'
    if os.environ.get('DAPSAM_SMOKE') == '1':
        cfg['prefix'] += '_smoke'

    print(f'\n{"="*70}')
    print('TRAINING')
    print(f'Backbone: {cfg["model_type"]}')
    print(f'Source: {source}')
    print(f'Targets: {", ".join(targets)}')
    print(f'{"="*70}')
    sys.stdout.flush()

    with runner.domain_context():
        device = cfg['device']

        model = build_dapsam(
            checkpoint=cfg['checkpoint'], model_type=cfg['model_type'],
            image_size=cfg['image_size'], num_classes=cfg['num_classes'],
            mem_dim=cfg['mem_dim']).to(device)

        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'\nModel params: {params/1e6:.3f}M (trainable: {trainable/1e6:.3f}M)')
        sys.stdout.flush()

        if cfg['phase'] == 'train':
            train_loader, val_loader = get_dapsam_loaders(
                source, image_size=cfg['image_size'],
                batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                pin_memory=cfg['pin_memory'],
            )
            runner.register_loaders(train_loader)
            runner.register_loaders(val_loader)

            train_dapsam(model, train_loader, val_loader, device, cfg)
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
            load_trainable(model, weight_path, device)
            for target in targets:
                print(f'\n{"="*70}')
                print('TESTING')
                print(f'Source: {source}')
                print(f'Target: {target}')
                print(f'Weight: {weight_tag}')
                test_dapsam_on_target(
                    model, target, device,
                    image_size=cfg['image_size'], batch_size=cfg['batch_size'],
                    num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'],
                    source_name=source, model_type=cfg['model_type'],
                    write_results=cfg['write_results'], weight_tag=weight_tag,
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
    'base_lr': 0.0005,          # repo train.py --base_lr
    'warmup': True,             # repo train.py --warmup
    'warmup_period': 250,       # repo train.py --warmup_period
    'lr_exp': 0.9,              # repo trainer.py poly decay exponent
    'dice_weight': 0.8,         # repo train.py --dice_param (paper Eq. 8)
    'patience': 5,             # early stopping on val loss (0 = disabled)

    'num_classes': 1,
    'mem_dim': 256,             # paper Table 6 (memory bank size N)

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
    smoke = os.environ.get('DAPSAM_SMOKE') == '1'
    if smoke:
        runs = [('OTU', ['OVATUS'])]
        CONFIG['n_epochs'] = 1
        CONFIG['write_results'] = False
    else:
        runs = SINGLE_SOURCE_RUNS
        if os.environ.get('DAPSAM_DOMAINS'):
            runs = [(s, [t for t in ts if t in DATASETS])
                    for s, ts in SINGLE_SOURCE_RUNS
                    if s in os.environ['DAPSAM_DOMAINS'].split(',')]
        if os.environ.get('DAPSAM_EPOCHS'):
            CONFIG['n_epochs'] = int(os.environ['DAPSAM_EPOCHS'])

    device = torch.device(CONFIG['device'])
    runner = SequentialDomainRunner(device=device)

    for source, targets in runs:
        cleanup_resources(device)
        run_single_source(source, targets, CONFIG, runner)
