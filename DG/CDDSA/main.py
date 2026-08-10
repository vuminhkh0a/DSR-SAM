"""
CDDSA: Contrastive Domain Disentanglement and Style Augmentation

Paper:  CDDSA: Contrastive Domain Disentanglement and Style Augmentation
        for Generalizable Medical Image Segmentation (arXiv 2211.12081)
Repo:   https://github.com/HiLab-git/DAG4MIA

Multi-source DG: trains on multiple source domains simultaneously,
evaluates on held-out target domain.
"""
import os
import sys
import torch
from utils.seed import set_seed
from utils.sequential_training import SequentialDomainRunner, cleanup_resources
from DG.CDDSA.data import (
    get_cddsa_train_loader, get_cddsa_val_loader, get_cddsa_target_loader,
)
from DG.CDDSA.train import train_cddsa, validate_cddsa
from DG.CDDSA.test import test_cddsa_on_target
from DG.CDDSA.train import CDDSAModel

set_seed()

DATASETS = ['OTU', 'OVATUS', 'USOVA']


def run_multi_source(sources, target, cfg, runner):
    cfg = dict(cfg)
    source_label = '_'.join(sources)
    cfg['model_dir'] = 'weights/cddsa/'
    cfg['prefix'] = f'cddsa_s_{source_label}'

    print(f'\n{"="*70}')
    print(f'TRAINING')
    print(f'Source: {source_label}')
    print(f'Target: {target}')
    print(f'{"="*70}')
    sys.stdout.flush()

    with runner.domain_context():
        num_domains = len(sources)
        device = cfg['device']

        model = CDDSAModel(
            z_length=cfg['z_length'],
            in_channel=cfg['in_channel'],
            img_size=cfg['image_size'],
            anatomy_channel=cfg['anatomy_channel'],
            num_classes=cfg['num_classes'],
        ).to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f'\nModel params: {params/1e6:.3f}M')
        sys.stdout.flush()

        if cfg['phase'] == 'train':
            train_loader = get_cddsa_train_loader(
                sources, cfg['image_size'], cfg['batch_size'],
                cfg['num_workers'], cfg['pin_memory'],
            )
            runner.register_loaders(train_loader)

            val_loader = get_cddsa_val_loader(
                sources[0], cfg['image_size'], cfg['batch_size'],
                cfg['num_workers'], cfg['pin_memory'],
            )
            runner.register_loaders(val_loader)

            train_cddsa(model, train_loader, val_loader, device, cfg)
            runner.destroy_loaders()

        best_path = os.path.join(cfg['model_dir'], f'{cfg["prefix"]}_best.pth')
        if os.path.exists(best_path):
            print(f'\nBest weight is loaded: {best_path}')
            ckpt = torch.load(best_path, map_location=device, weights_only=True)
            model.m_encoder.load_state_dict(ckpt['m_encoder'])
            model.a_encoder.load_state_dict(ckpt['a_encoder'])
            model.segmentor.load_state_dict(ckpt['segmentor'])
            model.decoder.load_state_dict(ckpt['decoder'])
        else:
            print(f'\nNo best checkpoint found at {best_path}')
            return

        print(f'\nSOURCE VALIDATION')
        print(f'Source: {source_label}')
        val_loader2 = get_cddsa_val_loader(
            sources[0], cfg['image_size'], cfg['batch_size'],
            cfg['num_workers'], cfg['pin_memory'],
        )
        runner.register_loaders(val_loader2)
        validate_cddsa(model, val_loader2, device)
        runner.destroy_loaders()

        target_loader = get_cddsa_target_loader(
            target, cfg['image_size'], cfg['batch_size'],
            cfg['num_workers'], cfg['pin_memory'], split='test',
        )
        runner.register_loaders(target_loader)

        print(f'\n{"="*70}')
        print(f'TESTING')
        print(f'Source: {source_label}')
        print(f'Target: {target}')
        test_cddsa_on_target(
            model, target_loader, device,
            source_label, target, write_results=True,
        )


CONFIG = {
    'image_size': 256,
    'batch_size': 4,
    'num_workers': 4,
    'pin_memory': True,
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',

    'n_epochs': 120,
    'lr': 0.001,
    'z_length': 16,
    'anatomy_channel': 8,
    'in_channel': 3,
    'num_classes': 1,

    'kl_w': 0.001,
    'seg_w': 1.0,
    'reco_w': 1.0,
    'recoz_w': 1.0,
    'style_w': 0.2,
    'cont_w': 0.2,
    'tau': 0.1,
    'n_minibatch': 8,

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
        raise ValueError(f"CDDSA requires multi_source mode. Got: {mode}")