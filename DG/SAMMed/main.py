"""
SAMMed: Leveraging SAM for Single-Source Domain Generalization in Medical
Image Segmentation (arXiv:2401.02076, repo SARIHUST/SAMMed).

Single-source DG benchmark (leave-one-out): train on ONE source domain
(OTU / OVATUS / USOVA) and test on the other two.

Pipeline (paper Fig. 1 / README):
  1. Train the DeepLabV3-ResNet50 segmentation backbone on the source
     domain (repo resnet_prostate.py) to predict coarse masks.
  2. Mask-filtering module: threshold theta_1 = 0.75, keep the largest
     connected component (BFS), and save the refined bounding boxes
     (repo save_resnet_bbox.py) for the source train/val splits and the
     target test splits.
  3. Fine-tune SAM ViT-B (mask decoder only, image/prompt encoders frozen
     via no_grad as in the repo training loop; paper Sec. 3.4) with the
     refined bounding boxes (merging strategy, repo
     sam_4_preprocessed_bbox.py), Adam lr = 1e-4, wd = 1e-3, 200 epochs.
  4. Test on the held-out domains with theta_2 = 0.5.
"""
import os
import sys
import torch

from utils.seed import set_seed
from utils.sequential_training import SequentialDomainRunner, cleanup_resources

from DG.SAMMed.model import build_sammed_sam, build_resnet_seg
from DG.SAMMed.data import get_sammed_loaders, get_resnet_loaders, box_save_dir
from DG.SAMMed.bbox_gen import ensure_bboxes
from DG.SAMMed.train import train_resnet, train_sam
from DG.SAMMed.test import test_sammed

set_seed()

DATASETS = ['OTU', 'OVATUS', 'USOVA']

CHECKPOINT = 'weights/sam/sam_vit_b_01ec64.pth'


def run_single_source(source, targets, cfg, device, runner):
    cfg = dict(cfg)
    cfg['model_dir'] = 'weights/sammed/'
    cfg['prefix'] = f'vit_b_sammed_s_{source}'
    cfg['resnet_prefix'] = f'sammed_resnet_s_{source}'

    print(f'\n{"="*70}')
    print('TRAINING (Stage 1: Segmentation Backbone)')
    print(f'Source: {source}')
    print(f'Targets: {", ".join(targets)}')
    print(f'{"="*70}')
    sys.stdout.flush()

    with runner.domain_context():
        device = cfg['device']

        # ---------------- Stage 1: DeepLabV3-ResNet50 backbone ----------------
        resnet = build_resnet_seg(num_classes=1).to(device)
        params = sum(p.numel() for p in resnet.parameters())
        print(f'\nResnet params: {params/1e6:.3f}M')
        sys.stdout.flush()

        if cfg['phase'] == 'train':
            train_loader, val_loader = get_resnet_loaders(
                source, image_size=cfg['image_size'],
                batch_size=cfg['resnet_batch_size'], num_workers=cfg['num_workers'],
                pin_memory=cfg['pin_memory'],
            )
            runner.register_loaders(train_loader)
            runner.register_loaders(val_loader)

            train_resnet(
                model=resnet, train_loader=train_loader, val_loader=val_loader,
                device=device, n_epochs=cfg['resnet_epochs'], lr=cfg['resnet_lr'],
                model_dir=cfg['model_dir'], prefix=cfg['resnet_prefix'],
            )
            runner.destroy_loaders()

        resnet_best = os.path.join(cfg['model_dir'], f'{cfg["resnet_prefix"]}_best.pth')
        if os.path.exists(resnet_best):
            print(f'\nBest resnet weight is loaded: {resnet_best}')
            resnet.load_state_dict(torch.load(resnet_best, map_location=device, weights_only=True))
        else:
            print(f'\nNo best resnet checkpoint found at {resnet_best}')
            return

        # ------- Stage 2: refined bboxes for source train/val + targets -------
        print(f'\n{"="*70}')
        print('TRAINING (Stage 2: Mask-Filtering -> Refined Bounding Boxes)')
        src_box_dir = box_save_dir(source, source, cfg['thresh_1'])
        ensure_bboxes(resnet, source, 'train', src_box_dir, image_size=cfg['image_size'],
                      thresh=cfg['thresh_1'], batch_size=cfg['resnet_batch_size'],
                      num_workers=cfg['num_workers'], device=device)
        ensure_bboxes(resnet, source, 'val', src_box_dir, image_size=cfg['image_size'],
                      thresh=cfg['thresh_1'], batch_size=cfg['resnet_batch_size'],
                      num_workers=cfg['num_workers'], device=device)
        for tgt in targets:
            tgt_box_dir = box_save_dir(source, tgt, cfg['thresh_1'])
            ensure_bboxes(resnet, tgt, 'test', tgt_box_dir, image_size=cfg['image_size'],
                          thresh=cfg['thresh_1'], batch_size=cfg['resnet_batch_size'],
                          num_workers=cfg['num_workers'], device=device)

        # ---------------- Stage 3: fine-tune SAM with bbox prompts ----------------
        print(f'\n{"="*70}')
        print('TRAINING (Stage 3: SAM fine-tuning with refined bboxes)')
        model = build_sammed_sam(checkpoint=cfg['checkpoint'], model_type='vit_b').to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f'\nModel params: {params/1e6:.3f}M (mask decoder fine-tuned per repo/paper; '
              f'image + prompt encoders frozen via no_grad)')
        sys.stdout.flush()

        if cfg['phase'] == 'train':
            train_loader, val_loader = get_sammed_loaders(
                source, box_dir=src_box_dir, image_size=cfg['image_size'],
                batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                pin_memory=cfg['pin_memory'],
            )
            runner.register_loaders(train_loader)
            runner.register_loaders(val_loader)

            train_sam(
                model=model, train_loader=train_loader, val_loader=val_loader,
                device=device, n_epochs=cfg['n_epochs'], lr=cfg['lr'],
                model_dir=cfg['model_dir'], prefix=cfg['prefix'],
            )
            runner.destroy_loaders()

        best_path = os.path.join(cfg['model_dir'], f'{cfg["prefix"]}_best.pth')
        if os.path.exists(best_path):
            print(f'\nBest weight is loaded: {best_path}')
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        else:
            print(f'\nNo best checkpoint found at {best_path}')
            return

        # ---------------- Stage 4: evaluate on the held-out targets ----------------
        for tgt in targets:
            print(f'\n{"="*70}')
            print('TESTING')
            print(f'Source: {source}')
            print(f'Target: {tgt}')
            tgt_box_dir = box_save_dir(source, tgt, cfg['thresh_1'])
            test_sammed(
                model=model, source_name=source, target_name=tgt, device=device,
                image_size=cfg['image_size'], batch_size=cfg['batch_size'],
                num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'],
                box_dir=tgt_box_dir, thresh=cfg['thresh_2'],
            )
        print(f'{"="*70}')


CONFIG = {
    'image_size': 512,          # per-patch resolution (repo prostate loads at 512)
    'num_workers': 4,
    'pin_memory': True,
    'device': 'cuda:1' if torch.cuda.is_available() else 'cpu',  # GPU0 busy; change if needed

    # Stage 1: DeepLabV3-ResNet50 backbone (repo resnet_prostate.py)
    'resnet_epochs': 100,
    'resnet_batch_size': 8,
    'resnet_lr': 0.001,         # repo --lr

    # Mask filtering (repo save_resnet_bbox.py)
    'thresh_1': 0.75,           # theta_1: coarse-mask confidence threshold

    # Stage 3: SAM fine-tuning (repo sam_4_preprocessed_bbox.py)
    'n_epochs': 200,            # repo --epoch
    'batch_size': 16,           # repo --batch_size (16 patches -> 4 merged 1024^2)
    'lr': 0.0001,               # repo --lr
    'thresh_2': 0.5,            # theta_2: SAM prediction threshold (paper Sec. 4.1)

    'phase': 'train',

    'checkpoint': CHECKPOINT,
}

SINGLE_SOURCE_RUNS = [
    ('OTU', ['OVATUS', 'USOVA']),
    ('OVATUS', ['OTU', 'USOVA']),
    ('USOVA', ['OTU', 'OVATUS']),
]


if __name__ == '__main__':
    runs = SINGLE_SOURCE_RUNS
    if os.environ.get('SAMMED_RUNS'):
        runs = [r for r in SINGLE_SOURCE_RUNS if r[0] in os.environ['SAMMED_RUNS'].split(',')]

    n_epochs = CONFIG['n_epochs']
    if os.environ.get('SAMMED_EPOCHS'):
        n_epochs = int(os.environ['SAMMED_EPOCHS'])
    CONFIG['n_epochs'] = n_epochs

    resnet_epochs = CONFIG['resnet_epochs']
    if os.environ.get('SAMMED_RESNET_EPOCHS'):
        resnet_epochs = int(os.environ['SAMMED_RESNET_EPOCHS'])
    CONFIG['resnet_epochs'] = resnet_epochs

    device = torch.device(CONFIG['device'])
    runner = SequentialDomainRunner(device=device)

    for source, targets in runs:
        cleanup_resources(device)
        run_single_source(source, targets, CONFIG, device, runner)
