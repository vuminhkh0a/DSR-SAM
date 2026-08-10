"""
DoFE: Domain-oriented Feature Embedding

Paper:  DoFE: Domain-Oriented Feature Embedding for Generalizable
        Fundus Image Segmentation on Unseen Datasets (TMI 2020)
Repo:   https://github.com/emma-sjwang/Dofe

Multi-source DG: trains on multiple source domains simultaneously,
evaluates on held-out target domain.
"""
import os
import sys
import torch
import torch.optim as optim
from utils.seed import set_seed
from utils.models import VGG16BN_Unet
from utils.sequential_training import SequentialDomainRunner, cleanup_resources
from DG.Dofe.data import (
    get_dofe_train_loader, get_pretrain_loader,
    get_dofe_val_loader, get_dofe_target_loader,
)
from DG.Dofe.train import pretrain, extract_features_for_centroids, train_dofe, validate_dofe
from DG.Dofe.test import test_dofe_on_target
from DG.Dofe.model import VGG16BN_DoFE

set_seed()

DATASETS = ['OTU', 'OVATUS', 'USOVA']


def run_multi_source(sources, target, cfg, runner):
    cfg = dict(cfg)
    source_label = '_'.join(sources)
    cfg['model_dir'] = 'weights/dofe/'
    cfg['prefix'] = f'dofe_s_{source_label}'

    print(f'\n{"="*70}')
    print(f'TRAINING')
    print(f'Source: {source_label}')
    print(f'{"="*70}')
    sys.stdout.flush()

    with runner.domain_context():
        if cfg['phase'] == 'train':
            num_domains = len(sources)
            device = cfg['device']

            # ---- Phase 1: Pretrain vanilla network ----
            pretrain_loader = get_pretrain_loader(
                sources, cfg['image_size'], cfg['batch_size'],
                cfg['num_workers'], cfg['pin_memory'],
            )
            runner.register_loaders(pretrain_loader)

            val_loader = get_dofe_val_loader(
                sources[0], cfg['image_size'], cfg['batch_size'],
                cfg['num_workers'], cfg['pin_memory'],
            )
            runner.register_loaders(val_loader)

            pretrain_model = VGG16BN_Unet(with_vgg16bn=True).to(device)
            params = sum(p.numel() for p in pretrain_model.parameters())
            print(f'\nModel params: {params/1e6:.3f}M')
            sys.stdout.flush()

            n_pretrain = cfg.get('n_pretrain_epochs', cfg['n_epochs'] // 3)
            pt_optimizer = optim.Adam(pretrain_model.parameters(), lr=cfg['lr'], betas=(0.9, 0.999))
            pt_scheduler = optim.lr_scheduler.ExponentialLR(pt_optimizer, gamma=0.99)

            print(f'\n[Phase 1] Pretraining ({n_pretrain} epochs)')
            pretrain_path = pretrain(
                pretrain_model, pretrain_loader, val_loader, device,
                pt_optimizer, pt_scheduler, n_pretrain,
                cfg['model_dir'], cfg['prefix'],
            )
            runner.destroy_loaders()

            # ---- Phase 2: Initialize centroids ----
            print(f'\n[Phase 2] Initializing Domain Knowledge Pool')
            dofe_model, features_dict = extract_features_for_centroids(
                pretrain_path, sources, cfg['image_size'], device, num_domains,
            )
            dofe_model.init_centroids_from_features(features_dict)

            # ---- Phase 3: Full DOFE training ----
            train_loader = get_dofe_train_loader(
                sources, cfg['image_size'], cfg['batch_size'],
                cfg['num_workers'], cfg['pin_memory'],
            )
            runner.register_loaders(train_loader)

            val_loader2 = get_dofe_val_loader(
                sources[0], cfg['image_size'], cfg['batch_size'],
                cfg['num_workers'], cfg['pin_memory'],
            )
            runner.register_loaders(val_loader2)

            n_dofe = cfg['n_epochs'] - n_pretrain
            d_optimizer = optim.Adam(dofe_model.parameters(), lr=cfg['lr'], betas=(0.9, 0.999))
            d_scheduler = optim.lr_scheduler.StepLR(d_optimizer, step_size=n_dofe // 2, gamma=0.2)

            print(f'\n[Phase 3] DOFE Training ({n_dofe} epochs)')
            train_dofe(
                dofe_model, train_loader, val_loader2, device,
                d_optimizer, d_scheduler, n_dofe,
                cfg['model_dir'], cfg['prefix'],
                alpha=cfg['alpha'], lam=cfg['lam'],
            )
            runner.destroy_loaders()

            # ---- Load best checkpoint ----
            best_path = os.path.join(cfg['model_dir'], f'{cfg["prefix"]}_best.pth')
            if os.path.exists(best_path):
                print(f'Best weight is loaded: {best_path}')
                dofe_model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
            else:
                print(f'\nNo best checkpoint found at {best_path}')

            # ---- Source validation ----
            print(f'\nSOURCE VALIDATION')
            print(f'Source: {source_label}')
            val_loader3 = get_dofe_val_loader(
                sources[0], cfg['image_size'], cfg['batch_size'],
                cfg['num_workers'], cfg['pin_memory'],
            )
            runner.register_loaders(val_loader3)
            validate_dofe(dofe_model, val_loader3, device)
            runner.destroy_loaders()

        else:
            # Test-only mode
            feat_hw = cfg['image_size'] // 16
            dofe_model = VGG16BN_DoFE(num_domains=len(sources), with_vgg16bn=True, feat_hw=feat_hw).to(cfg['device'])
            best_path = os.path.join(cfg['model_dir'], f'{cfg["prefix"]}_best.pth')
            if not os.path.exists(best_path):
                print(f'Checkpoint not found: {best_path}')
                return
            dofe_model.load_state_dict(torch.load(best_path, map_location=cfg['device'], weights_only=True))
            print(f'\nLoaded checkpoint: {best_path}')

        # ---- Target evaluation ----
        target_loader = get_dofe_target_loader(
            target, cfg['image_size'], cfg['batch_size'],
            cfg['num_workers'], cfg['pin_memory'], split='test',
        )
        runner.register_loaders(target_loader)

        print(f'\n{"="*70}')
        print(f'TESTING')
        print(f'Source: {source_label}')
        print(f'Target: {target}')
        test_dofe_on_target(
            dofe_model, target_loader, cfg['device'],
            source_label, target, write_results=True,
        )


CONFIG = {
    'image_size': 256,
    'batch_size': 4,
    'num_workers': 4,
    'pin_memory': True,
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',

    'n_epochs': 60,
    'n_pretrain_epochs': 20,
    'lr': 0.001,
    'alpha': 0.1,
    'lam': 0.9,

    'mode': 'multi_source',
    'phase': 'train',
}


if __name__ == '__main__':
    mode = CONFIG['mode']
    device = torch.device(CONFIG['device'])
    runner = SequentialDomainRunner(device=device)

    if mode == 'multi_source':
        for tgt in DATASETS:
            sources = [d for d in DATASETS if d != tgt]
            cleanup_resources(device)
            run_multi_source(sources, tgt, CONFIG, runner)
    else:
        raise ValueError(f"DoFE requires multi_source mode. Got: {mode}")
