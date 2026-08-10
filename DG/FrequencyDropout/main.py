"""
Frequency Dropout (FD): Feature-Level Regularization via Randomized Filtering

Paper:  "Frequency Dropout: Feature-Level Regularization via Randomized
         Filtering" (ECCV 2022 MCV Workshop)
Repo:   https://github.com/mobarakol/Frequency_Dropout
Baselin: https://github.com/pairlab/CBS

Single-source DG: trains on ONE source domain, evaluates on the other two
held-out domains (leave-one-out benchmark).
"""
import os
import sys
import torch
from utils.seed import set_seed
from utils.models import VGG16BN_Unet
from utils.sequential_training import SequentialDomainRunner, cleanup_resources
from utils.data import get_dataloaders, get_target_loader
from DG.FrequencyDropout.train import train_fd
from DG.FrequencyDropout.test import test_fd_on_target
from DG.FrequencyDropout.model import VGG16BN_Unet_FD

set_seed()

DATASETS = ['OTU', 'OVATUS', 'USOVA']


def run_single_source(source, targets, cfg, runner):
    cfg = dict(cfg)
    cfg['model_dir'] = 'weights/frequency_dropout/'
    cfg['prefix'] = f'fd_s_{source}'

    print(f'\n{"="*70}')
    print(f'TRAINING')
    print(f'Source: {source}')
    print(f'Targets: {", ".join(targets)}')
    print(f'{"="*70}')
    sys.stdout.flush()

    with runner.domain_context():
        device = cfg['device']

        if cfg['use_fd']:
            model = VGG16BN_Unet_FD(with_vgg16bn=True, fd_cfg=cfg['fd_cfg']).to(device)
        else:
            model = VGG16BN_Unet(with_vgg16bn=True).to(device)

        params = sum(p.numel() for p in model.parameters())
        print(f'\nModel params: {params/1e6:.3f}M')
        sys.stdout.flush()

        if cfg['phase'] == 'train':
            train_loader, val_loader, _ = get_dataloaders(
                name=source, image_size=cfg['image_size'], transform=None,
                batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                pin_memory=cfg['pin_memory'],
            )
            runner.register_loaders(train_loader)
            runner.register_loaders(val_loader)

            train_fd(model, train_loader, val_loader, device, cfg)
            runner.destroy_loaders()

        # Load best checkpoint
        best_path = os.path.join(cfg['model_dir'], f'{cfg["prefix"]}_best.pth')
        if os.path.exists(best_path):
            print(f'\nBest weight is loaded: {best_path}')
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        else:
            print(f'\nNo best checkpoint found at {best_path}')
            return

        # Source validation
        print(f'\nSOURCE VALIDATION')
        print(f'Source: {source}')
        _, val_loader_src, _ = get_dataloaders(
            name=source, image_size=cfg['image_size'], transform=None,
            batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
            pin_memory=cfg['pin_memory'],
        )
        runner.register_loaders(val_loader_src)
        from utils.eval import evaluate
        _, src_dice, src_iou, src_prec, src_rec, src_hd95 = evaluate(
            model=model, model_name='FD', device=device,
            loader=val_loader_src, with_loss=False, with_hd95=True,
            print_results=True, write_results=False,
        )
        runner.destroy_loaders()

        # Target evaluation (on each held-out domain)
        for tgt in targets:
            target_loader = get_target_loader(
                tgt, cfg['image_size'], cfg['batch_size'],
                cfg['num_workers'], cfg['pin_memory'], split='test',
            )
            runner.register_loaders(target_loader)

            print(f'\n{"="*70}')
            print(f'TESTING')
            print(f'Source: {source}')
            print(f'Target: {tgt}')
            test_fd_on_target(
                model, target_loader, device,
                source, tgt, write_results=True,
            )


CONFIG = {
    'image_size': 256,
    'batch_size': 4,
    'num_workers': 4,
    'pin_memory': True,
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',

    'n_epochs': 60,
    'lr': 0.001,

    'use_fd': True,
    'phase': 'train',

    # Frequency Dropout configuration
    'fd_cfg': {
        'kernel_size': 3,

        'use_rf': True,
        # Filter param ranges: [Gaussian, LoG, Gabor]
        'freq_min_all': [0.2, 0.2, 0.0],
        'freq_max_all': [1.0, 3.0, 1.0],
        'dropout_p_all': [0.4, 0.5, 0.8],
    },
}


if __name__ == '__main__':
    device = torch.device(CONFIG['device'])
    runner = SequentialDomainRunner(device=device)

    for src in DATASETS:
        targets = [d for d in DATASETS if d != src]
        cleanup_resources(device)
        run_single_source(src, targets, CONFIG, runner)
