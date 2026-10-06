"""
MaxStyle: Adversarial Style Composition for Robust Medical Image
Segmentation (MICCAI 2022, arXiv:2206.01737).

Paper: "MaxStyle: Adversarial Style Composition for Robust Medical Image
       Segmentation"
Repo:  https://github.com/cherise215/MaxStyle

Single-source DG: trains on ONE source domain, evaluates on the two
held-out target domains (leave-one-out benchmark, 3 sources x 2 targets
= 6 cases). A binary UNet carries MaxStyle layers on its 4 encoder
stages; style params are adversarially optimized (repo config
MICCAI2022_MaxStyle.json: n_iter=5, style lr=0.1) while the net trains
with AdamW lr 1e-4 (repo learning lr/optimizer_type).
"""
import os
import sys
import torch

from utils.seed import set_seed
from utils.env import get_device
from utils.sequential_training import SequentialDomainRunner, cleanup_resources
from DG.maxstyle.data import get_maxstyle_loaders
from DG.maxstyle.model import build_maxstyle_unet
from DG.maxstyle.train import train_maxstyle
from DG.maxstyle.test import test_maxstyle_on_target

set_seed()

DATASETS = ['OTU', 'OVATUS', 'USOVA']


def run_single_source(source, targets, cfg, runner):
    cfg = dict(cfg)
    cfg['model_dir'] = 'weights/maxstyle/'
    cfg['prefix'] = f'maxstyle_s_{source}'

    print(f'\n{"="*70}')
    print('TRAINING')
    print(f'Source: {source}')
    print(f'Targets: {", ".join(targets)}')
    print(f'{"="*70}')
    sys.stdout.flush()

    with runner.domain_context():
        device = cfg['device']

        model = build_maxstyle_unet(in_ch=3, num_classes=1).to(device)

        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'\nModel params: {params/1e6:.3f}M (trainable: {trainable/1e6:.3f}M)')
        sys.stdout.flush()

        if cfg['phase'] == 'train':
            train_loader, val_loader = get_maxstyle_loaders(
                source, image_size=cfg['image_size'],
                batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                pin_memory=cfg['pin_memory'],
            )
            runner.register_loaders(train_loader)
            runner.register_loaders(val_loader)

            train_maxstyle(model, train_loader, val_loader, device, cfg)
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
                test_maxstyle_on_target(
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
    'base_lr': 0.0001,          # repo config learning lr
    'patience': 5,             # early stopping on val loss (0 = disabled)

    'style_n_iter': 5,          # repo max_style n_iter
    'style_lr': 0.1,            # repo max_style lr

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
    if os.environ.get('MAXSTYLE_DOMAINS'):
        runs = [(s, [t for t in ts if t in DATASETS])
                for s, ts in SINGLE_SOURCE_RUNS
                if s in os.environ['MAXSTYLE_DOMAINS'].split(',')]

    n_epochs = CONFIG['n_epochs']
    if os.environ.get('MAXSTYLE_EPOCHS'):
        n_epochs = int(os.environ['MAXSTYLE_EPOCHS'])
    CONFIG['n_epochs'] = n_epochs

    device = torch.device(CONFIG['device'])
    runner = SequentialDomainRunner(device=device)

    for source, targets in runs:
        cleanup_resources(device)
        run_single_source(source, targets, CONFIG, runner)
