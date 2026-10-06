"""
CoSAM: Self-Correcting SAM for Domain Generalization in 2D Medical Image
Segmentation (arXiv:2411.10136).

Paper: https://arxiv.org/pdf/2411.10136
       "CoSAM: Self-Correcting SAM for Domain Generalization in 2D
        Medical Image Segmentation"
Repo:  no public code was released with the request; this implementation
       follows the paper (Alg. 1/2, Eq. 1-9, Sec. 5.2).

Single-source DG: trains on ONE source domain, evaluates on the two
held-out target domains (leave-one-out benchmark, 3 sources x 2 targets
= 6 cases). Hyper-parameters below are the paper's prostate settings
(Sec. 5.2: Adam 1e-4 poly^0.9, K=64, lr/g weights 1.0/0.1, T=4);
alpha is the paper's perturbation probability (Eq. 3).

Note: the paper trains SAM-B (ViT-B), so the default backbone here is
vit_b (switchable to vit_h via model_type + checkpoint, like the other
DG methods). Image size follows the y_DG benchmark (256).
"""
import os
import sys
import torch

from utils.seed import set_seed
from utils.env import get_device
from utils.sequential_training import SequentialDomainRunner, cleanup_resources
from utils.checkpoint import load_trainable
from DG.cosam.data import get_cosam_loaders
from DG.cosam.model import build_cosam
from DG.cosam.train import train_cosam
from DG.cosam.test import test_cosam_on_target

set_seed()

DATASETS = ['OTU', 'OVATUS', 'USOVA']


def run_single_source(source, targets, cfg, runner):
    cfg = dict(cfg)
    cfg['model_dir'] = 'weights/cosam/'
    cfg['prefix'] = f'{cfg["model_type"]}_cosam_s_{source}'
    if os.environ.get('COSAM_SMOKE') == '1':
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

        model = build_cosam(
            checkpoint=cfg['checkpoint'], model_type=cfg['model_type'],
            image_size=cfg['image_size'],
            num_classes=cfg['num_classes']).to(device)

        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'\nModel params: {params/1e6:.3f}M (trainable: {trainable/1e6:.3f}M)')
        sys.stdout.flush()

        if cfg['phase'] == 'train':
            train_loader, val_loader = get_cosam_loaders(
                source, image_size=cfg['image_size'],
                batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                pin_memory=cfg['pin_memory'],
            )
            runner.register_loaders(train_loader)
            runner.register_loaders(val_loader)

            train_cosam(model, train_loader, val_loader, device, cfg)
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
                test_cosam_on_target(
                    model, target, device,
                    image_size=cfg['image_size'], batch_size=cfg['batch_size'],
                    num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'],
                    source_name=source, model_type=cfg['model_type'],
                    write_results=cfg['write_results'], weight_tag=weight_tag, cfg=cfg,
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
    'base_lr': 0.0001,          # paper Sec. 5.2 (prostate)
    'lr_exp': 0.9,              # paper Sec. 5.2 (polynomial decay power)
    'patience': 5,             # early stopping on val loss (0 = disabled)

    'num_classes': 1,
    'num_points': 64,           # paper Sec. 5.2 (K, prostate)
    'num_iters': 4,             # paper Sec. 5.2 (T, prostate)
    'perturb_prob': 0.2,        # paper Sec. 5.2 (alpha, prostate)
    'lr_weight': 1.0,           # paper Sec. 5.2 (lambda_r)
    'lg_weight': 0.1,           # paper Sec. 5.2 (lambda_g)

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
    smoke = os.environ.get('COSAM_SMOKE') == '1'
    if smoke:
        runs = [('OTU', ['OVATUS'])]
        CONFIG['n_epochs'] = 1
        CONFIG['write_results'] = False
    else:
        runs = SINGLE_SOURCE_RUNS
        if os.environ.get('COSAM_DOMAINS'):
            runs = [(s, [t for t in ts if t in DATASETS])
                    for s, ts in SINGLE_SOURCE_RUNS
                    if s in os.environ['COSAM_DOMAINS'].split(',')]
        if os.environ.get('COSAM_EPOCHS'):
            CONFIG['n_epochs'] = int(os.environ['COSAM_EPOCHS'])

    device = torch.device(CONFIG['device'])
    runner = SequentialDomainRunner(device=device)

    for source, targets in runs:
        cleanup_resources(device)
        run_single_source(source, targets, CONFIG, runner)
