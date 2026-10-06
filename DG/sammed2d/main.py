"""
SAM-Med2D zero-shot evaluation (OpenGVLab/SAM-Med2D,
"SAM-Med2D: Segment Anything in 2D Medical Images").

Paper: https://arxiv.org/pdf/2308.16184 (SAM-Med2D)
Repo:  https://github.com/uni-medical/SAM-Med2D
       (official code is maintained at https://github.com/OpenGVLab/SAM-Med2D)

The official sam-med2d_b.pth checkpoint (ViT-B + adapters, native 256)
is evaluated with box prompts from box_coords.json (upstream test.py,
boxes_prompt=True). Inference only: no training is performed, and each
benchmark dataset is tested once (3 cases: t_OTU, t_OVATUS, t_USOVA).
"""
import os
import sys
import torch

from utils.seed import set_seed
from utils.env import get_device
from utils.sequential_training import SequentialDomainRunner, cleanup_resources
from DG.sammed2d.model import build_sammed2d
from DG.sammed2d.test import test_sammed2d_on_target

set_seed()

DATASETS = ['OTU', 'OVATUS', 'USOVA']


def run_testonly(targets, cfg, runner):
    cfg = dict(cfg)
    ckpt = cfg['test_checkpoint']
    print(f'\n{"="*70}')
    print('TEST-ONLY (official SAM-Med2D, no training)')
    print(f'Backbone: {cfg["model_type"]} + adapter')
    print(f'Checkpoint: {ckpt}')
    print(f'Targets: {", ".join(targets)}')
    print(f'{"="*70}')
    sys.stdout.flush()

    if not os.path.exists(ckpt):
        print(f'\nNo test checkpoint found at {ckpt}, skipping.')
        return

    with runner.domain_context():
        device = cfg['device']
        model = build_sammed2d(
            checkpoint=ckpt, model_type=cfg['model_type'],
            image_size=cfg['image_size']).to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f'\nModel params: {params/1e6:.3f}M (frozen, test-only)')
        sys.stdout.flush()
        for target in targets:
            print(f'\n{"="*70}')
            print('TESTING')
            print(f'Target: {target}')
            print(f'Weight: testonly ({ckpt})')
            test_sammed2d_on_target(
                model, target, device,
                image_size=cfg['image_size'], batch_size=cfg['batch_size'],
                num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'],
                model_type=cfg['model_type'],
                write_results=True, weight_tag='testonly',
            )
        print(f'{"="*70}')


CONFIG = {
    'image_size': 256,            # native SAM-Med2D resolution
    'batch_size': 8,
    'num_workers': 4,
    'pin_memory': True,
    'device': get_device(),  # CUDA_DEVICE from y_DG/.env

    'test_checkpoint': 'weights/sam_med2d/sam-med2d_b.pth',

    'model_type': 'vit_b',
}

if __name__ == '__main__':
    targets = DATASETS
    if os.environ.get('SAMMED2D_TEST_DOMAINS'):
        targets = [t for t in DATASETS
                   if t in os.environ['SAMMED2D_TEST_DOMAINS'].split(',')]

    device = torch.device(CONFIG['device'])
    runner = SequentialDomainRunner(device=device)

    cleanup_resources(device)
    run_testonly(targets, CONFIG, runner)
