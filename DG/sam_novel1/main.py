"""
SAM + LoRA (rank 8): plain single-source DG baseline.

Architecture: frozen SAM ViT-B image encoder + prompt encoder, LoRA
adapters on the q/v projections of every transformer block, and a
trainable mask decoder.  No EMA, no subspace regularization, no auxiliary
loss, no dual branch: just SAM + LoRA.

Single-source DG: trains on ONE source domain, evaluates on the two
held-out domains (leave-one-out benchmark, one source per run).
"""
import os
import sys
import torch

from utils.seed import set_seed
from utils.env import get_device
from utils.sequential_training import SequentialDomainRunner, cleanup_resources
from utils.checkpoint import load_trainable
from DG.sam_novel1.data import get_sam_novel1_loaders
from DG.sam_novel1.model import build_sam_novel1
from DG.sam_novel1.train import train_sam_novel1
from DG.sam_novel1.test import test_sam_novel1_on_target

set_seed()

DATASETS = ['OTU', 'OVATUS', 'USOVA']


def run_single_source(source, targets, cfg, runner):
    cfg = dict(cfg)
    # Reseed per source so a case does not depend on RNG consumed by earlier
    # sources (same reproducibility policy as the other DG methods).
    set_seed(cfg.get('seed', 42))
    cfg['model_dir'] = 'weights/sam_novel1/'
    cfg['prefix'] = f'{cfg["model_type"]}_sam_novel1_s_{source}'

    print(f'\n{"="*70}')
    print('TRAINING')
    print(f'Backbone: {cfg["model_type"]}')
    print(f'Adapter: LoRA (rank {cfg["rank"]})')
    print(f'Source: {source}')
    print(f'Targets: {", ".join(targets)}')
    print(f'{"="*70}')
    sys.stdout.flush()

    with runner.domain_context():
        device = cfg['device']

        model = build_sam_novel1(
            checkpoint=cfg['checkpoint'], model_type=cfg['model_type'],
            image_size=cfg['image_size'], num_classes=cfg['num_classes'],
            rank=cfg['rank'],
        ).to(device)

        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'\nModel params: {params/1e6:.3f}M '
              f'(trainable: {trainable/1e6:.3f}M)')
        sys.stdout.flush()

        if cfg['phase'] == 'train':
            train_loader, val_loader = get_sam_novel1_loaders(
                source, image_size=cfg['image_size'],
                batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                pin_memory=cfg['pin_memory'],
                prompt_method=cfg['prompt_method'],
                yolo_size=cfg['prompt_yolo_size'],
            )
            runner.register_loaders(train_loader)
            runner.register_loaders(val_loader)

            train_sam_novel1(model, train_loader, val_loader, device, cfg)
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
                test_sam_novel1_on_target(
                    model, target, device,
                    image_size=cfg['image_size'], batch_size=cfg['batch_size'],
                    num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'],
                    source_name=source, model_type=cfg['model_type'],
                    prompt_method=cfg['prompt_method'],
                    yolo_size=cfg['prompt_yolo_size'],
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
    'seed': 42,                 # RNG seed (reset per source for reproducibility)

    'n_epochs': 100,
    'base_lr': 0.0005,          # standard AdamW learning rate for SAM LoRA
    'patience': 20,             # early stopping on val loss (0 = disabled)

    'num_classes': 1,
    'rank': 8,                  # LoRA rank (requested baseline)

    # Prompt boxes: 'gt' -> box_coords.json (usual benchmark);
    # 'yolov8' -> box_coords_yolov8_<prompt_yolo_size>.json
    'prompt_method': os.environ.get('PROMPT_METHOD', 'gt'),
    'prompt_yolo_size': os.environ.get('PROMPT_YOLO_SIZE', 'm'),
    'phase': 'train',

    'save_last_epoch': False,   # keep the best checkpoint only

    'model_type': 'vit_b',
    'checkpoint': 'weights/sam/sam_vit_b_01ec64.pth',
}

SINGLE_SOURCE_RUNS = [
    ('OTU', ['OVATUS', 'USOVA']),
    ('OVATUS', ['OTU', 'USOVA']),
    ('USOVA', ['OTU', 'OVATUS']),
]


if __name__ == '__main__':
    runs = SINGLE_SOURCE_RUNS
    if os.environ.get('SAM_NOVEL1_RUNS'):
        runs = [r for r in SINGLE_SOURCE_RUNS
                if r[0] in os.environ['SAM_NOVEL1_RUNS'].split(',')]

    n_epochs = CONFIG['n_epochs']
    if os.environ.get('SAM_NOVEL1_EPOCHS'):
        n_epochs = int(os.environ['SAM_NOVEL1_EPOCHS'])
    CONFIG['n_epochs'] = n_epochs

    device = torch.device(CONFIG['device'])
    runner = SequentialDomainRunner(device=device)

    for source, targets in runs:
        cleanup_resources(device)
        run_single_source(source, targets, CONFIG, runner)
