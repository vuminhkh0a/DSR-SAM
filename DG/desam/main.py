"""
DeSAM: Decoupled Segment Anything Model for Generalizable Medical Image
Segmentation (MICCAI 2024).

Paper: https://arxiv.org/pdf/2306.00499
Repo:  https://github.com/yifangao112/DeSAM

Single-source domain generalization benchmark (leave-one-out):
train on one source domain (OTU / OVATUS / USOVA) and test on the other two.
The original SAM ViT-B image encoder and the prompt encoder are frozen;
PRIM + PDMM (the decoupled mask decoder) are fine-tuned with point prompts
(grid-points mode, DeSAM-P). (The paper/repo default is ViT-H; this
benchmark uses the original SAM ViT-B backbone, model_type='vit_b'.)
"""
import os
import sys

import torch

from utils.seed import set_seed
from utils.env import get_device
from utils.sequential_training import SequentialDomainRunner, cleanup_resources
from utils.checkpoint import load_trainable

from DG.desam.model import build_desam
from DG.desam.data import get_desam_loaders
from DG.desam.train import train_desam
from DG.desam.test import test_desam

set_seed()

DATASETS = ['OTU', 'OVATUS', 'USOVA']

CHECKPOINT = 'weights/sam/sam_vit_b_01ec64.pth'


def run_single_source(source, targets, cfg, device, runner):
    cfg = dict(cfg)
    cfg['model_dir'] = 'weights/desam/'
    cfg['prefix'] = f'{cfg["model_type"]}_desam_s_{source}'
    if os.environ.get('DESAM_SMOKE') == '1':
        cfg['prefix'] += '_smoke'

    print(f'\n{"="*70}')
    print('TRAINING')
    print(f'Source: {source}')
    print(f'Targets: {", ".join(targets)}')
    print(f'{"="*70}')
    sys.stdout.flush()

    with runner.domain_context():
        model = build_desam(checkpoint=cfg['checkpoint'], model_type=cfg['model_type']).to(device)
        for p in model.image_encoder.parameters():
            p.requires_grad = False
        for p in model.prompt_encoder.parameters():
            p.requires_grad = False

        trainable = sum(p.numel() for p in model.mask_decoder.parameters())
        total = sum(p.numel() for p in model.parameters())
        print(f'\nModel params: {total/1e6:.2f}M total | {trainable/1e6:.2f}M trainable')
        sys.stdout.flush()

        if cfg['phase'] == 'train':
            train_loader, val_loader = get_desam_loaders(
                source, cfg['image_size'], cfg['batch_size'],
                cfg['num_workers'], cfg['pin_memory'], cfg['neg_points'],
            )
            runner.register_loaders(train_loader)
            runner.register_loaders(val_loader)

            optimizer = torch.optim.Adam(
                model.mask_decoder.parameters(), lr=cfg['lr'], weight_decay=0,
            )
            train_desam(
                model=model, train_loader=train_loader, val_loader=val_loader,
                device=device, optimizer=optimizer, n_epochs=cfg['n_epochs'],
                lr=cfg['lr'], model_dir=cfg['model_dir'], prefix=cfg['prefix'],
                patience=cfg.get('patience', 5),
                save_last=cfg.get('save_last_epoch', False),
            )
            runner.destroy_loaders()

        best_path = os.path.join(cfg['model_dir'], f'{cfg["prefix"]}_best.pth')
        last_path = os.path.join(cfg['model_dir'], f'{cfg["prefix"]}_last.pth')

        def evaluate(weight_path, weight_tag):
            if not os.path.exists(weight_path):
                print(f'\nNo {weight_tag} checkpoint found at {weight_path}')
                return
            print(f'\n{weight_tag.capitalize()} weight is loaded: {weight_path}')
            load_trainable(model, weight_path, device)
            for tgt in targets:
                print(f'\n{"="*70}')
                print('TESTING')
                print(f'Source: {source}')
                print(f'Target: {tgt}')
                print(f'Weight: {weight_tag}')
                test_desam(
                    model=model, source_name=source, target_name=tgt, device=device,
                    image_size=cfg['image_size'], batch_size=cfg['batch_size'],
                    num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'],
                    grid=cfg['grid'], iou_thresh=cfg['iou_thresh'],
                    model_type=cfg['model_type'],
                    write_results=cfg['write_results'], weight_tag=weight_tag,
                )

        evaluate(best_path, 'best')
        if cfg.get('save_last_epoch', False):
            evaluate(last_path, 'last')


CONFIG = {
    'device': get_device(),  # CUDA_DEVICE from y_DG/.env
    'model_type': 'vit_b',
    'checkpoint': CHECKPOINT,
    'n_epochs': 50,
    'patience': 5,              # early stopping on val loss (0 = disabled)
    'batch_size': 8,
    'lr': 0.0001,
    'neg_points': 1,
    'grid': 9,
    'iou_thresh': 0.5,
    'image_size': 256,
    'num_workers': 4,
    'pin_memory': True,
    'model_dir': 'weights/desam',
    'save_last_epoch': False,    # keep the last-epoch snapshot besides best
    'write_results': True,       # False during smoke runs
    'phase': 'train',
}


if __name__ == '__main__':
    smoke = os.environ.get('DESAM_SMOKE') == '1'
    if smoke:
        runs = [('OTU', ['OVATUS'])]
        CONFIG['n_epochs'] = 1
        CONFIG['write_results'] = False
    else:
        domains = DATASETS
        if os.environ.get('DESAM_DOMAINS'):
            domains = [d for d in DATASETS if d in os.environ['DESAM_DOMAINS'].split(',')]
        if os.environ.get('DESAM_EPOCHS'):
            CONFIG['n_epochs'] = int(os.environ['DESAM_EPOCHS'])
        runs = [(src, [d for d in DATASETS if d != src]) for src in domains]

    device = torch.device(CONFIG['device'])
    runner = SequentialDomainRunner(device=device)

    for src, target in runs:
        cleanup_resources(device)
        run_single_source(src, target, CONFIG, device, runner)