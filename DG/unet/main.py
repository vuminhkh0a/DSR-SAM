"""
UNet (Ronneberger et al., MICCAI 2015, arXiv:1505.04597).

Paper: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
Repo:  https://github.com/milesial/pytorch-unet (plain UNet baseline,
       distinct from SL's VGG16BN variant)

Single-source DG: trains on ONE source domain, evaluates on the two
held-out target domains (leave-one-out benchmark, 3 sources x 2 targets
= 6 cases).
"""
import os
import sys
import torch

from utils.seed import set_seed
from utils.env import get_device
from utils.sequential_training import SequentialDomainRunner, cleanup_resources
from DG.unet.data import get_unet_loaders
from DG.unet.model import build_unet
from DG.unet.train import train_unet
from DG.unet.test import test_unet_on_target

set_seed()

DATASETS = ['OTU', 'OVATUS', 'USOVA']


def run_single_source(source, targets, cfg, runner):
    cfg = dict(cfg)
    cfg['model_dir'] = 'weights/unet/'
    cfg['prefix'] = f'unet_s_{source}'

    print(f'\n{"="*70}')
    print('TRAINING')
    print(f'Source: {source}')
    print(f'Targets: {", ".join(targets)}')
    print(f'{"="*70}')
    sys.stdout.flush()

    with runner.domain_context():
        device = cfg['device']

        model = build_unet(n_channels=3, n_classes=1,
                           bilinear=cfg['bilinear']).to(device)

        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'\nModel params: {params/1e6:.3f}M (trainable: {trainable/1e6:.3f}M)')
        sys.stdout.flush()

        if cfg['phase'] == 'train':
            train_loader, val_loader = get_unet_loaders(
                source, image_size=cfg['image_size'],
                batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                pin_memory=cfg['pin_memory'],
            )
            runner.register_loaders(train_loader)
            runner.register_loaders(val_loader)

            train_unet(model, train_loader, val_loader, device, cfg)
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
                test_unet_on_target(
                    model, target, device,
                    image_size=cfg['image_size'], batch_size=cfg['batch_size'],
                    num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'],
                    source_name=source,
                    write_results=True, weight_tag=weight_tag,
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
    'base_lr': 1e-5,            # repo train.py --learning-rate
    'weight_decay': 1e-8,       # repo train.py default
    'momentum': 0.999,          # repo train.py default
    'plateau_patience': 5,      # repo ReduceLROnPlateau patience (max Dice)
    'patience': 5,             # early stopping on val loss (0 = disabled)
    'bilinear': False,          # repo default (transposed-conv upsampling)

    'save_last_epoch': False,    # keep the last-epoch snapshot besides best

    'phase': 'train',
}

SINGLE_SOURCE_RUNS = [
    ('OTU', ['OVATUS', 'USOVA']),
    ('OVATUS', ['OTU', 'USOVA']),
    ('USOVA', ['OTU', 'OVATUS']),
]


if __name__ == '__main__':
    runs = SINGLE_SOURCE_RUNS
    if os.environ.get('UNET_DOMAINS'):
        runs = [(s, [t for t in ts if t in DATASETS])
                for s, ts in SINGLE_SOURCE_RUNS
                if s in os.environ['UNET_DOMAINS'].split(',')]

    n_epochs = CONFIG['n_epochs']
    if os.environ.get('UNET_EPOCHS'):
        n_epochs = int(os.environ['UNET_EPOCHS'])
    CONFIG['n_epochs'] = n_epochs

    device = torch.device(CONFIG['device'])
    runner = SequentialDomainRunner(device=device)

    for source, targets in runs:
        cleanup_resources(device)
        run_single_source(source, targets, CONFIG, runner)
