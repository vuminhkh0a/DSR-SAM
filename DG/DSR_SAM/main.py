"""
SR-SAM: Subspace Regularization for Domain Generalization of Segment
Anything Model

Paper: "Subspace Regularization for Domain Generalization of Segment
       Anything Model" (MICCAI 2025)
Repo:  https://github.com/xjiangmed/SR-SAM

Single-source DG: trains on ONE source domain, evaluates on the two
held-out domains (leave-one-out benchmark, one source per run).
"""
import os
import sys
import torch

from utils.seed import set_seed
from utils.sequential_training import SequentialDomainRunner, cleanup_resources
from DG.DSR_SAM.data import get_dsr_sam_loaders
from DG.DSR_SAM.model import build_dsr_sam
from DG.DSR_SAM.train import train_dsr_sam
from DG.DSR_SAM.test import test_dsr_sam_on_target

set_seed()



def run_single_source(source, targets, cfg, runner):
    cfg = dict(cfg)
    cfg['model_dir'] = 'weights/dsr_sam/'
    cfg['prefix'] = f'{cfg["model_type"]}_dsr_sam_s_{source}'
    cfg['source'] = source

    print(f'\n{"="*70}')
    print('TRAINING')
    print(f'Backbone: {cfg["model_type"]}')
    print(f'Source: {source}')
    print(f'Targets: {", ".join(targets)}')
    print(f'{"="*70}')
    sys.stdout.flush()

    with runner.domain_context():
        device = cfg['device']

        model = build_dsr_sam(
            checkpoint=cfg['checkpoint'], model_type=cfg['model_type'],
            image_size=cfg['image_size'], num_classes=cfg['num_classes'],
            rank=cfg['rank'], ema_mode=cfg['ema_mode'],
            truncation_size=cfg['truncation_size'],
            lora_A_init=cfg.get('lora_A_init', 'kaiming'),
        ).to(device)

        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'\nModel params: {params/1e6:.3f}M (trainable: {trainable/1e6:.3f}M)')
        sys.stdout.flush()

        if cfg['phase'] == 'train':
            train_loader, val_loader = get_dsr_sam_loaders(
                source, image_size=cfg['image_size'],
                batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                pin_memory=cfg['pin_memory'],
            )
            runner.register_loaders(train_loader)
            runner.register_loaders(val_loader)

            train_dsr_sam(model, train_loader, val_loader, device, cfg)
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
            model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
            for target in targets:
                print(f'\n{"="*70}')
                print('TESTING')
                print(f'Source: {source}')
                print(f'Target: {target}')
                print(f'Weight: {weight_tag}')
                test_dsr_sam_on_target(
                    model, target, device,
                    image_size=cfg['image_size'], batch_size=cfg['batch_size'],
                    num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'],
                    source_name=source, model_type=cfg['model_type'],
                    write_results=True, weight_tag=weight_tag,
                )
            print(f'{"="*70}')

        evaluate(best_path, 'best')
        evaluate(last_path, 'last')


CONFIG = {
    'image_size': 256,
    'batch_size': 8,
    'num_workers': 4,
    'pin_memory': True,
    'device': 'cuda:1' if torch.cuda.is_available() else 'cpu',  # GPU0 busy; change if needed

    'n_epochs': 50,
    'base_lr': 0.0005,          # paper Sec. 3 (initial learning rate)
    'warmup_period': 250,       # paper Sec. 3 (warm-up iterations)

    'num_classes': 1,
    'rank': 64,                 # paper Sec. 3 (LoRA rank)
    'lora_A_init': 'orthogonal',   # 'kaiming' (default, current) or 'orthogonal'
    'ema_mode': False,
    'ema_rate': 0.999,          # paper Sec. 2.3 (EMA rate alpha)
    'kd_weight': 1e-7,          # paper Sec. 3 (lambda, polyp) / repo --kd_weight
    'beta1': 0.1,                # L_tsd coefficient
    'truncation': False,
    'truncation_size': 96,      # paper Sec. 3 (s, Table 4)
    'truncation_period': 4,     # paper Sec. 3 (every 4 epochs)
    'dash_warm': 300,           # repo run_CVC-ClinicDB.sh --Dash_warm 300
    'freeze_A_after_N_epoch': 0,  # freeze matrix A of student LoRA and EMA LoRA after N epochs
    'is_freeze_A_after_N_epoch': True,  # whether to apply freeze_A_after_N_epoch
    'compute_l_tsd': False,      # whether to compute L_tsd loss

    'experiment_process_save_epochs': False,  # save weights at epochs 1-20 and 60,100,140,...
    'save_last_epoch': False,  # whether to save weights at the last epoch

    'phase': 'train',

    'model_type': 'vit_h',
    'checkpoint': 'weights/sam/sam_vit_h_4b8939.pth',
}

SINGLE_SOURCE_RUNS = [
    ('OTU', ['OTU', 'OVATUS']),
    # ('OVATUS', ['OTU', 'OVATUS']),
]


if __name__ == '__main__':
    runs = SINGLE_SOURCE_RUNS
    if os.environ.get('DSR_SAM_RUNS'):
        runs = [r for r in SINGLE_SOURCE_RUNS
                if r[0] in os.environ['DSR_SAM_RUNS'].split(',')]

    n_epochs = CONFIG['n_epochs']
    if os.environ.get('DSR_SAM_EPOCHS'):
        n_epochs = int(os.environ['DSR_SAM_EPOCHS'])
    CONFIG['n_epochs'] = n_epochs
    if os.environ.get('DSR_SAM_LORA_A_INIT') in ('kaiming', 'orthogonal'):
        CONFIG['lora_A_init'] = os.environ['DSR_SAM_LORA_A_INIT']

    device = torch.device(CONFIG['device'])
    runner = SequentialDomainRunner(device=device)

    for source, targets in runs:
        cleanup_resources(device)
        run_single_source(source, targets, CONFIG, runner)
        
