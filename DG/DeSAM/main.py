"""
DeSAM: Decoupled Segment Anything Model for Generalizable Medical Image
Segmentation (MICCAI 2024, arXiv:2306.00499, repo yifangao112/DeSAM).

Single-source domain generalization benchmark (leave-one-out):
train on one source domain (OTU / OVATUS / USOVA) and test on the other two.
The SAM ViT-H image encoder and the prompt encoder are frozen; PRIM + PDMM
(the decoupled mask decoder) are fine-tuned with point prompts (grid-points
mode, DeSAM-P).
"""
import os
import sys

import torch

from utils.seed import set_seed
from utils.sequential_training import SequentialDomainRunner, cleanup_resources

from DG.DeSAM.model import build_desam
from DG.DeSAM.data import get_desam_loaders
from DG.DeSAM.train import train_desam
from DG.DeSAM.test import test_desam

set_seed()

DATASETS = ['OTU', 'OVATUS', 'USOVA']

CHECKPOINT = 'weights/sam/sam_vit_h_4b8939.pth'


def run_single_source(source, targets, cfg, device, runner):
    cfg = dict(cfg)
    cfg['model_dir'] = 'weights/desam/'
    cfg['prefix'] = f'vit_h_desam_s_{source}'

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
            )
            runner.destroy_loaders()

        best_path = os.path.join(cfg['model_dir'], f'{cfg["prefix"]}_best.pth')
        if os.path.exists(best_path):
            print(f'\nBest weight is loaded: {best_path}')
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        else:
            print(f'\nNo best checkpoint found at {best_path}')
            return

        for tgt in targets:
            print(f'\n{"="*70}')
            print('TESTING')
            print(f'Source: {source}')
            print(f'Target: {tgt}')
            test_desam(
                model=model, source_name=source, target_name=tgt, device=device,
                image_size=cfg['image_size'], num_workers=cfg['num_workers'],
                pin_memory=cfg['pin_memory'], grid=cfg['grid'],
                iou_thresh=cfg['iou_thresh'],
            )


CONFIG = {
    'device': 'cuda:1' if torch.cuda.is_available() else 'cpu',  # GPU0 busy; change if needed
    'model_type': 'vit_h',
    'checkpoint': CHECKPOINT,
    'n_epochs': 50,
    'batch_size': 4,  # frozen ViT-H encode is memory-bound; fp16 encode peaks ~11GB at bs4
    'lr': 0.0001,
    'neg_points': 1,
    'grid': 9,
    'iou_thresh': 0.5,
    'image_size': 256,
    'num_workers': 4,
    'pin_memory': True,
    'model_dir': 'weights/desam',
    'phase': 'train',
}


if __name__ == '__main__':
    domains = DATASETS
    if os.environ.get('DESAM_DOMAINS'):
        domains = [d for d in DATASETS if d in os.environ['DESAM_DOMAINS'].split(',')]

    n_epochs = CONFIG['n_epochs']
    if os.environ.get('DESAM_EPOCHS'):
        n_epochs = int(os.environ['DESAM_EPOCHS'])
    CONFIG['n_epochs'] = n_epochs

    device = torch.device(CONFIG['device'])
    runner = SequentialDomainRunner(device=device)

    for src in domains:
        target = [d for d in DATASETS if d != src]
        cleanup_resources(device)
        run_single_source(src, target, CONFIG, device, runner)