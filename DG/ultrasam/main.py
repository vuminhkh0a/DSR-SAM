"""
UltraSAM zero-shot evaluation (CAMMA-public/UltraSam,
"UltraSam: A Foundation Model for Ultrasound").

Paper: https://arxiv.org/pdf/2411.16222 (UltraSam)
Repo:  https://github.com/CAMMA-public/UltraSam

The official UltraSam.pth (mmdet checkpoint, converted on the fly) is
evaluated with box prompts from box_coords.json at the native 1024
resolution. Inference only: no training is performed, and each
benchmark dataset is tested once (3 cases: t_OTU, t_OVATUS, t_USOVA).
"""
import os
import sys
import torch

from utils.seed import set_seed
from utils.env import get_device
from utils.sequential_training import SequentialDomainRunner, cleanup_resources
from DG.ultrasam.model import build_ultrasam
from DG.ultrasam.test import test_ultrasam_on_target

set_seed()

DATASETS = ['OTU', 'OVATUS', 'USOVA']


def run_testonly(targets, cfg, runner):
    cfg = dict(cfg)
    ckpt = cfg['test_checkpoint']
    print(f'\n{"="*70}')
    print('TEST-ONLY (official UltraSAM, no training)')
    print(f'Backbone: {cfg["model_type"]}')
    print(f'Checkpoint: {ckpt}')
    print(f'Targets: {", ".join(targets)}')
    print(f'{"="*70}')
    sys.stdout.flush()

    if not os.path.exists(ckpt):
        print(f'\nNo test checkpoint found at {ckpt}, skipping.')
        return

    with runner.domain_context():
        device = cfg['device']
        model = build_ultrasam(checkpoint=ckpt).to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f'\nModel params: {params/1e6:.3f}M (frozen, test-only)')
        sys.stdout.flush()
        for target in targets:
            print(f'\n{"="*70}')
            print('TESTING')
            print(f'Target: {target}')
            print(f'Weight: testonly ({ckpt})')
            test_ultrasam_on_target(
                model, target, device,
                image_size=cfg['image_size'], batch_size=cfg['batch_size'],
                num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'],
                model_type=cfg['model_type'],
                write_results=True, weight_tag='testonly',
            )
        print(f'{"="*70}')


CONFIG = {
    'image_size': 256,          # benchmark size; forward runs at 1024 native
    'batch_size': 8,
    'num_workers': 4,
    'pin_memory': True,
    'device': get_device(),  # CUDA_DEVICE from y_DG/.env

    'test_checkpoint': 'weights/ultrasam/ultrasam_vit_b.pth',

    'model_type': 'vit_b',
}

if __name__ == '__main__':
    targets = DATASETS
    if os.environ.get('ULTRASAM_TEST_DOMAINS'):
        targets = [t for t in DATASETS
                   if t in os.environ['ULTRASAM_TEST_DOMAINS'].split(',')]

    device = torch.device(CONFIG['device'])
    runner = SequentialDomainRunner(device=device)

    cleanup_resources(device)
    run_testonly(targets, CONFIG, runner)
