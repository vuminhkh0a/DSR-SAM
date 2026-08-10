import os
import sys
import torch
import torch.optim as optim
from utils.seed import set_seed
from utils.models import Unet2D_DN
from DG.DualNormalization.data import (
    get_source_loaders, get_target_loader, get_source_val_loader,
)
from utils.data import generator as dn_generator
from utils.seed import worker_init_fn as dn_worker_init_fn
from DG.DualNormalization.train import train_dn, validate_dn
from DG.DualNormalization.test import test_dn_on_target, get_bn_stats_from_model
from utils.sequential_training import SequentialDomainRunner, cleanup_resources

set_seed()


DATASETS = ['OTU', 'OVATUS', 'USOVA']


def run_single_source(src, targets, cfg, runner):
    domain_cfg = dict(cfg)
    domain_cfg['source_domains'] = [src]
    domain_cfg['num_domains'] = 2
    domain_cfg['model_dir'] = 'weights/dn/'
    domain_cfg['prefix'] = f'dn_s_{src}'

    with runner.domain_context():
        print(f'\n{"="*70}')
        print(f'TRAINING')
        print(f'Source: {src}')
        print(f'{"="*70}')
        sys.stdout.flush()

        if domain_cfg['phase'] == 'train':
            source_loaders = get_source_loaders(
                source_names=[src],
                image_size=domain_cfg['image_size'],
                batch_size=domain_cfg['batch_size'],
                num_workers=domain_cfg['num_workers'],
                pin_memory=domain_cfg['pin_memory'],
                mode='single_source',
                apply_style_aug=True,
            )
            runner.register_loaders(source_loaders)

            val_loader = get_source_val_loader(
                source_name=src,
                image_size=domain_cfg['image_size'],
                batch_size=domain_cfg['batch_size'],
                num_workers=domain_cfg['num_workers'],
                pin_memory=domain_cfg['pin_memory'],
            )
            runner.register_loaders(val_loader)

            model = Unet2D_DN(
                in_channels=3, n=16, num_classes=1,
                num_domains=2, momentum=0.1,
            ).to(domain_cfg['device'])

            params = sum(p.numel() for p in model.parameters())
            print(f'\nModel params: {params/1e6:.3f}M')
            sys.stdout.flush()

            optimizer = optim.Adam(model.parameters(), lr=domain_cfg['lr'], betas=(0.9, 0.999))
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

            train_dn(
                model=model, loaders=source_loaders,
                val_loader=val_loader,
                device=domain_cfg['device'],
                optimizer=optimizer, scheduler=scheduler,
                n_epochs=domain_cfg['n_epochs'],
                model_dir=domain_cfg['model_dir'],
                prefix=domain_cfg['prefix'],
            )

            runner.destroy_loaders()

            best_path = os.path.join(domain_cfg['model_dir'], f'{domain_cfg["prefix"]}_best.pth')
            if os.path.exists(best_path):
                model.load_state_dict(torch.load(best_path, map_location=domain_cfg['device']))

            print(f'\nSOURCE VALIDATION')
            print(f'Source: {src}')
            val_loader_2 = get_source_val_loader(
                source_name=src,
                image_size=domain_cfg['image_size'],
                batch_size=domain_cfg['batch_size'],
                num_workers=domain_cfg['num_workers'],
                pin_memory=domain_cfg['pin_memory'],
            )
            runner.register_loaders(val_loader_2)
            validate_dn(model, val_loader_2, domain_cfg['device'], domain_id=0)
            runner.destroy_loaders()
        else:
            model = Unet2D_DN(
                in_channels=3, n=16, num_classes=1,
                num_domains=2, momentum=0.1,
            ).to(domain_cfg['device'])
            best_path = os.path.join(domain_cfg['model_dir'], f'{domain_cfg["prefix"]}_best.pth')
            if not os.path.exists(best_path):
                print(f'Checkpoint not found: {best_path}')
                return
            model.load_state_dict(torch.load(best_path, map_location=domain_cfg['device']))
            print(f'\nLoaded checkpoint: {best_path}')

        stored_means, stored_vars = get_bn_stats_from_model(model, 2)

        for tgt in targets:
            target_loader = get_target_loader(
                target_name=tgt,
                image_size=domain_cfg['image_size'],
                batch_size=domain_cfg['batch_size'],
                num_workers=domain_cfg['num_workers'],
                pin_memory=domain_cfg['pin_memory'],
                split='test',
            )
            runner.register_loaders(target_loader)

            print(f'\n{"="*70}')
            print(f'TESTING')
            print(f'Source: {src}')
            print(f'Target: {tgt}')
            print(f'{"="*70}')
            test_dn_on_target(
                model=model, target_loader=target_loader,
                device=domain_cfg['device'], num_domains=2,
                stored_means=stored_means, stored_vars=stored_vars,
                source_label=src, target_name=tgt, write_results=True,
            )
            runner.destroy_loaders()


def get_multi_source_val_loader(source_names, image_size, batch_size, num_workers, pin_memory):
    from DG.DualNormalization.data import _SourceValDataset, _make_loader
    from torch.utils.data import ConcatDataset
    datasets = []
    for name in source_names:
        ds = _SourceValDataset(name, image_size)
        datasets.append(ds)
    combined = ConcatDataset(datasets)
    return _make_loader(
        combined, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )


def run_multi_source(sources, target, cfg, runner):
    cfg = dict(cfg)
    cfg['source_domains'] = sources
    cfg['num_domains'] = len(sources)
    cfg['model_dir'] = 'weights/dn/'
    cfg['prefix'] = f'dn_s_{"_".join(sources)}'

    source_label = '_'.join(sources)
    print(f'\n{"="*70}')
    print(f'TRAINING')
    print(f'Source: {source_label}')
    print(f'{"="*70}')
    sys.stdout.flush()

    with runner.domain_context():
        if cfg['phase'] == 'train':
            source_loaders = get_source_loaders(
                source_names=sources,
                image_size=cfg['image_size'],
                batch_size=cfg['batch_size'],
                num_workers=cfg['num_workers'],
                pin_memory=cfg['pin_memory'],
                mode='multi_source',
                apply_style_aug=True,
            )
            runner.register_loaders(source_loaders)

            val_loader = get_multi_source_val_loader(
                sources,
                image_size=cfg['image_size'],
                batch_size=cfg['batch_size'],
                num_workers=cfg['num_workers'],
                pin_memory=cfg['pin_memory'],
            )
            runner.register_loaders(val_loader)

            model = Unet2D_DN(
                in_channels=3, n=16, num_classes=1,
                num_domains=cfg['num_domains'], momentum=0.1,
            ).to(cfg['device'])

            params = sum(p.numel() for p in model.parameters())
            print(f'\nModel params: {params/1e6:.3f}M')
            sys.stdout.flush()

            optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], betas=(0.9, 0.999))
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

            train_dn(
                model=model, loaders=source_loaders,
                val_loader=val_loader,
                device=cfg['device'],
                optimizer=optimizer, scheduler=scheduler,
                n_epochs=cfg['n_epochs'],
                model_dir=cfg['model_dir'],
                prefix=cfg['prefix'],
            )

            runner.destroy_loaders()

            best_path = os.path.join(cfg['model_dir'], f'{cfg["prefix"]}_best.pth')
            if os.path.exists(best_path):
                print(f'Best weight is loaded: {best_path}')
                model.load_state_dict(torch.load(best_path, map_location=cfg['device']))
        else:
            model = Unet2D_DN(
                in_channels=3, n=16, num_classes=1,
                num_domains=cfg['num_domains'], momentum=0.1,
            ).to(cfg['device'])
            best_path = os.path.join(cfg['model_dir'], f'{cfg["prefix"]}_best.pth')
            if not os.path.exists(best_path):
                print(f'Checkpoint not found: {best_path}')
                return
            model.load_state_dict(torch.load(best_path, map_location=cfg['device']))
            print(f'\nLoaded checkpoint: {best_path}')

        target_loader = get_target_loader(
            target_name=target,
            image_size=cfg['image_size'],
            batch_size=cfg['batch_size'],
            num_workers=cfg['num_workers'],
            pin_memory=cfg['pin_memory'],
            split='test',
        )
        runner.register_loaders(target_loader)

        print(f'\n{"="*70}')
        print(f'TESTING')
        print(f'Source: {source_label}')
        print(f'Target: {target}')
        stored_means, stored_vars = get_bn_stats_from_model(model, cfg['num_domains'])
        test_dn_on_target(
            model=model, target_loader=target_loader,
            device=cfg['device'], num_domains=cfg['num_domains'],
            stored_means=stored_means, stored_vars=stored_vars,
            source_label=source_label, target_name=target, write_results=True,
        )


CONFIG = {
    'image_size': 256,
    'batch_size': 4,
    'num_workers': 4,
    'pin_memory': True,
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',

    'n_epochs': 50,
    'lr': 0.001,

    'mode': 'single_source',
    'phase': 'train',
}


if __name__ == '__main__':
    mode = CONFIG['mode']
    device = torch.device(CONFIG['device'])

    runner = SequentialDomainRunner(
        device=device,
        generator=dn_generator,
        worker_init_fn=dn_worker_init_fn,
    )

    if mode == 'multi_source':
        for tgt in DATASETS:
            sources = [d for d in DATASETS if d != tgt]
            cleanup_resources(device)
            run_multi_source(sources, tgt, CONFIG, runner)

    elif mode == 'single_source':
        for src in DATASETS:
            targets = [d for d in DATASETS if d != src]
            cleanup_resources(device)
            run_single_source(src, targets, CONFIG, runner)
    else:
        raise ValueError(f"Unknown mode: {mode}")
