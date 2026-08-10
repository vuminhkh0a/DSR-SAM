import os
import sys
import torch
import torch.optim as optim
from utils.seed import set_seed
from utils.models import VGG16BN_Unet
from DG.AADG.data import (
    get_source_loader, get_multi_source_loader,
    get_target_loader, get_source_val_loader,
)
from DG.AADG.train import train_aadg, validate_aadg
from DG.AADG.test import test_aadg_on_target
from utils.sequential_training import SequentialDomainRunner, cleanup_resources

set_seed()

DATASETS = ['OTU', 'OVATUS', 'USOVA']


def run_single_source(src, targets, cfg, runner):
    domain_cfg = dict(cfg)
    domain_cfg['source_domains'] = [src]
    domain_cfg['model_dir'] = 'weights/aadg/'
    domain_cfg['prefix'] = f'aadg_s_{src}'

    with runner.domain_context():
        print(f'\n{"="*70}')
        print(f'TRAINING')
        print(f'Source: {src}')
        print(f'{"="*70}')
        sys.stdout.flush()

        if domain_cfg['phase'] == 'train':
            train_loader = get_source_loader(
                source_name=src,
                image_size=domain_cfg['image_size'],
                batch_size=domain_cfg['batch_size'],
                num_workers=domain_cfg['num_workers'],
                pin_memory=domain_cfg['pin_memory'],
            )
            runner.register_loaders(train_loader)

            val_loader = get_source_val_loader(
                source_name=src,
                image_size=domain_cfg['image_size'],
                batch_size=domain_cfg['batch_size'],
                num_workers=domain_cfg['num_workers'],
                pin_memory=domain_cfg['pin_memory'],
            )
            runner.register_loaders(val_loader)

            model = VGG16BN_Unet(with_vgg16bn=True).to(domain_cfg['device'])

            params = sum(p.numel() for p in model.parameters())
            print(f'\nModel params: {params/1e6:.3f}M')
            sys.stdout.flush()

            optimizer = optim.Adam(model.parameters(), lr=domain_cfg['lr'], betas=(0.9, 0.999))
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

            train_aadg(
                model=model, train_loader=train_loader,
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
            validate_aadg(model, val_loader_2, domain_cfg['device'])
            runner.destroy_loaders()
        else:
            model = VGG16BN_Unet(with_vgg16bn=True).to(domain_cfg['device'])
            best_path = os.path.join(domain_cfg['model_dir'], f'{domain_cfg["prefix"]}_best.pth')
            if not os.path.exists(best_path):
                print(f'Checkpoint not found: {best_path}')
                return
            model.load_state_dict(torch.load(best_path, map_location=domain_cfg['device']))
            print(f'\nLoaded checkpoint: {best_path}')

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
            test_aadg_on_target(
                model=model, target_loader=target_loader,
                device=domain_cfg['device'],
                source_label=src, target_name=tgt, write_results=True,
            )
            runner.destroy_loaders()


def run_multi_source(sources, target, cfg, runner):
    cfg = dict(cfg)
    cfg['source_domains'] = sources
    cfg['model_dir'] = 'weights/aadg/'
    cfg['prefix'] = f'aadg_s_{"_".join(sources)}'

    source_label = '_'.join(sources)
    print(f'\n{"="*70}')
    print(f'TRAINING')
    print(f'Source: {source_label}')
    print(f'{"="*70}')
    sys.stdout.flush()

    with runner.domain_context():
        if cfg['phase'] == 'train':
            train_loader = get_multi_source_loader(
                source_names=sources,
                image_size=cfg['image_size'],
                batch_size=cfg['batch_size'],
                num_workers=cfg['num_workers'],
                pin_memory=cfg['pin_memory'],
            )
            runner.register_loaders(train_loader)

            val_loader = get_source_val_loader(
                source_name=sources[0],
                image_size=cfg['image_size'],
                batch_size=cfg['batch_size'],
                num_workers=cfg['num_workers'],
                pin_memory=cfg['pin_memory'],
            )
            runner.register_loaders(val_loader)

            model = VGG16BN_Unet(with_vgg16bn=True).to(cfg['device'])

            params = sum(p.numel() for p in model.parameters())
            print(f'\nModel params: {params/1e6:.3f}M')
            sys.stdout.flush()

            optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], betas=(0.9, 0.999))
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

            train_aadg(
                model=model, train_loader=train_loader,
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
            model = VGG16BN_Unet(with_vgg16bn=True).to(cfg['device'])
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
        test_aadg_on_target(
            model=model, target_loader=target_loader,
            device=cfg['device'],
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

    runner = SequentialDomainRunner(device=device)

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
