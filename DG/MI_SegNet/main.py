import os
import sys
import torch
import torch.optim as optim
from utils.seed import set_seed
from DG.MI_SegNet.model import SegEncoder, SegDecoder, ReconEncoder, ReconDecoder, Mine_Conv
from DG.MI_SegNet.data import get_source_loader
from utils.data import get_source_val_loader, get_target_loader
from DG.MI_SegNet.train import train_mi_segnet, validate_epoch
from DG.MI_SegNet.test import test_mi_segnet_on_target
from utils.sequential_training import SequentialDomainRunner, cleanup_resources

set_seed()

DATASETS = ['OTU', 'OVATUS', 'USOVA']


def run_single_source(src, targets, cfg, runner):
    domain_cfg = dict(cfg)
    domain_cfg['source_domains'] = [src]
    domain_cfg['model_dir'] = 'weights/mi_segnet/'
    domain_cfg['prefix'] = f'mi_segnet_s_{src}'

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

            seg_encoder = SegEncoder(
                in_channels=domain_cfg['in_channels'],
                init_features=domain_cfg['seg_init_features'],
                num_blocks=domain_cfg['num_blocks'],
            ).to(domain_cfg['device'])
            seg_decoder = SegDecoder(
                out_channels=domain_cfg['out_channels'],
                init_features=domain_cfg['seg_init_features'],
                num_blocks=domain_cfg['num_blocks'],
            ).to(domain_cfg['device'])
            rec_encoder = ReconEncoder(
                in_channels=domain_cfg['in_channels'],
                init_features=domain_cfg['rec_init_features'],
            ).to(domain_cfg['device'])
            rec_decoder = ReconDecoder(
                in_channels_a=16 * domain_cfg['seg_init_features'],
                in_channels_d=16 * domain_cfg['rec_init_features'],
                out_channels=domain_cfg['in_channels'],
                init_features=domain_cfg['rec_init_features'],
            ).to(domain_cfg['device'])
            mine = Mine_Conv(
                in_channels_x=16 * domain_cfg['seg_init_features'],
                in_channels_y=16 * domain_cfg['rec_init_features'],
                inter_channels=domain_cfg['mine_inter_channels'],
            ).to(domain_cfg['device'])

            seg_params = sum(p.numel() for p in seg_encoder.parameters()) + sum(p.numel() for p in seg_decoder.parameters())
            rec_params = sum(p.numel() for p in rec_encoder.parameters()) + sum(p.numel() for p in rec_decoder.parameters())
            mine_params = sum(p.numel() for p in mine.parameters())
            print(f'\nSeg params: {seg_params / 1e6:.3f}M | Rec params: {rec_params / 1e6:.3f}M | Mine params: {mine_params / 1e3:.1f}K')
            sys.stdout.flush()

            opt_seg_en = optim.Adam(seg_encoder.parameters(), lr=domain_cfg['lr'], betas=(0.9, 0.999))
            opt_seg_de = optim.Adam(seg_decoder.parameters(), lr=domain_cfg['lr'], betas=(0.9, 0.999))
            opt_rec_en = optim.Adam(rec_encoder.parameters(), lr=domain_cfg['lr'], betas=(0.9, 0.999))
            opt_rec_de = optim.Adam(rec_decoder.parameters(), lr=domain_cfg['lr'], betas=(0.9, 0.999))
            opt_mine = optim.Adam(mine.parameters(), lr=domain_cfg['lr'], betas=(0.9, 0.999))

            print(f'\n{"="*70}')
            print(f'TRAINING')
            print(f'Source: {src}')
            print(f'{"="*70}')
            train_mi_segnet(
                seg_encoder=seg_encoder, seg_decoder=seg_decoder,
                rec_encoder=rec_encoder, rec_decoder=rec_decoder,
                mine=mine,
                train_loader=train_loader, val_loader=val_loader,
                device=domain_cfg['device'],
                opt_seg_en=opt_seg_en, opt_seg_de=opt_seg_de,
                opt_rec_en=opt_rec_en, opt_rec_de=opt_rec_de,
                opt_mine=opt_mine,
                n_epochs=domain_cfg['n_epochs'],
                model_dir=domain_cfg['model_dir'],
                prefix=domain_cfg['prefix'],
            )

            runner.destroy_loaders()

            best_path = os.path.join(domain_cfg['model_dir'], f'{domain_cfg["prefix"]}_best.pth')
            if os.path.exists(best_path):
                ckpt = torch.load(best_path, map_location=domain_cfg['device'])
                seg_encoder.load_state_dict(ckpt['seg_encoder'])
                seg_decoder.load_state_dict(ckpt['seg_decoder'])

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
            validate_epoch(seg_encoder, seg_decoder, val_loader_2, domain_cfg['device'])
            runner.destroy_loaders()
        else:
            seg_encoder = SegEncoder(
                in_channels=domain_cfg['in_channels'],
                init_features=domain_cfg['seg_init_features'],
                num_blocks=domain_cfg['num_blocks'],
            ).to(domain_cfg['device'])
            seg_decoder = SegDecoder(
                out_channels=domain_cfg['out_channels'],
                init_features=domain_cfg['seg_init_features'],
                num_blocks=domain_cfg['num_blocks'],
            ).to(domain_cfg['device'])
            best_path = os.path.join(domain_cfg['model_dir'], f'{domain_cfg["prefix"]}_best.pth')
            if not os.path.exists(best_path):
                print(f'Checkpoint not found: {best_path}')
                return
            ckpt = torch.load(best_path, map_location=domain_cfg['device'])
            seg_encoder.load_state_dict(ckpt['seg_encoder'])
            seg_decoder.load_state_dict(ckpt['seg_decoder'])
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
            test_mi_segnet_on_target(
                seg_encoder=seg_encoder, seg_decoder=seg_decoder,
                target_loader=target_loader,
                device=domain_cfg['device'],
                source_label=src, target_name=tgt, write_results=True,
            )
            runner.destroy_loaders()


CONFIG = {
    'image_size': 256,
    'batch_size': 4,
    'num_workers': 4,
    'pin_memory': True,
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',

    'n_epochs': 200,
    'lr': 1e-4,

    'in_channels': 3,
    'out_channels': 1,
    'seg_init_features': 64,
    'rec_init_features': 16,
    'num_blocks': 2,
    'mine_inter_channels': 64,

    'mode': 'single_source',
    'phase': 'train',
}


if __name__ == '__main__':
    mode = CONFIG['mode']
    device = torch.device(CONFIG['device'])

    runner = SequentialDomainRunner(device=device)

    if mode == 'single_source':
        for src in DATASETS:
            targets = [d for d in DATASETS if d != src]
            cleanup_resources(device)
            run_single_source(src, targets, CONFIG, runner)
    else:
        raise ValueError(f"Unknown mode: {mode}")
