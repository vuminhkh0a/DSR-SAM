"""DSR-SAM2 entry point.

Dual-branch DSR-SAM: unchanged SAM ViT encoder (all blocks kept) plus an
ImageNet1k-v1 ResNet18 CNN branch. ``cba_last_n = n`` appends n extra
768->768 CNN ResBlocks after the always-kept layer4 and enables 2n
CKA-scaled Cross-Branch Attention blocks in the last n ViT blocks
(no ViT block is added). Each CBA uses
``f_gated_cnn = AttentionGate(f_cnn, f_vit)`` (Oktay et al. 2018) and
``f_vit + ((1 - C_weight) * f_cnn + f_gated_cnn) / 2``, where ``C_weight``
is the per-channel linear CKA. The final fusion is plain
``x = x + cnn_final`` (no CKA, no gate).
Training adds ``cka_loss_weight * L_CKA`` (default 0.1): ``L_CKA`` is the
normalized sum of all CBA ``C_weight`` terms.
LoRA/EMA are inherited from DSR-SAM.

Single-source DG: trains on ONE source domain, evaluates on the two
held-out domains (leave-one-out benchmark, one source per run).
Results go to results.json as ``*_dsr_sam2_cka_*`` and weights to
``weights/dsr_sam2/``.
"""
import os
import sys
import torch

from utils.seed import set_seed
from utils.env import get_device
from utils.sequential_training import SequentialDomainRunner, cleanup_resources
from DG.dsr_sam2.data import get_dsr_sam2_loaders
from DG.dsr_sam2.model import build_dsr_sam2
from DG.dsr_sam2.train import train_dsr_sam2
from DG.dsr_sam2.test import test_dsr_sam2_on_target

set_seed()



def run_single_source(source, targets, cfg, runner):
    cfg = dict(cfg)
    # Phase 2 (reproducibility): reseed per source so a source's run no longer
    # depends on RNG consumed by previously run sources/cases.
    set_seed(cfg.get('seed', 42))
    cfg['model_dir'] = 'weights/dsr_sam2/'
    # `weight_name_tag` lets variants save to distinct weights; ablations pass
    # the same tag so their checkpoints overwrite one set of files.
    cfg['prefix'] = f'{cfg["model_type"]}_{cfg.get("weight_name_tag", "dsr_sam2_cka")}_s_{source}'
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

        model = build_dsr_sam2(
            checkpoint=cfg['checkpoint'], model_type=cfg['model_type'],
            image_size=cfg['image_size'], num_classes=cfg['num_classes'],
            rank=cfg['rank'], ema_mode=cfg['ema_mode'],
            cnn_pretrained=cfg.get('cnn_pretrained', True),
            cba_last_n=cfg.get('cba_last_n', 4),
            cnn_branch=cfg.get('cnn_branch', True),
            cka_fusion=cfg.get('cka_fusion', True),
            use_attention_gate=cfg.get('use_attention_gate', True),
            use_attention_gate_cba=cfg.get('use_attention_gate_cba', True),
            attention_gate_channels=cfg.get('attention_gate_channels', None),
        ).to(device)

        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'\nModel params: {params/1e6:.3f}M (trainable: {trainable/1e6:.3f}M)')
        sys.stdout.flush()

        if cfg['phase'] == 'train':
            train_loader, val_loader = get_dsr_sam2_loaders(
                source, image_size=cfg['image_size'],
                batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                pin_memory=cfg['pin_memory'],
            )
            runner.register_loaders(train_loader)
            runner.register_loaders(val_loader)

            train_dsr_sam2(model, train_loader, val_loader, device, cfg)
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
                test_dsr_sam2_on_target(
                    model, target, device,
                    image_size=cfg['image_size'], batch_size=cfg['batch_size'],
                    num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'],
                    source_name=source, model_type=cfg['model_type'],
                    write_results=True, weight_tag=weight_tag,
                    result_tag=cfg.get('result_name_tag'),
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
    'base_lr': 0.0005,          # paper Sec. 3 (initial learning rate)
    'warmup_period': 250,       # paper Sec. 3 (warm-up iterations)
    'patience': 20,              # early stopping on val loss (0 = disabled)

    'num_classes': 1,
    'rank': 64,                 # paper Sec. 3 (LoRA rank)
    'ema_mode': False,
    'ema_rate': 0.999,          # paper Sec. 2.3 (EMA rate alpha)
    'kd_weight': 1e-7,          # paper Sec. 3 (lambda, polyp) / repo --kd_weight

    'experiment_process_save_epochs': False,  # off here; run_sota enables it (even epochs 2,4,...,20)
    'save_last_epoch': False,  # whether to save weights at the last epoch

    'phase': 'train',

    'model_type': 'vit_b',
    'checkpoint': 'weights/sam/sam_vit_b_01ec64.pth',

    # ---- DSR-SAM2 dual-branch architecture ----
    'cnn_branch': True,       # False -> plain SAM + LoRA (no CNN, no CBA)
    'cnn_pretrained': True,   # ResNet18 IMAGENET1K_V1 through layer3
    # n -> last n ViT blocks x 2 CBA each = 2*n CBA blocks, plus n additional
    # 768->768 CNN ResBlocks (layer5 ... layer(4+n)). 0..12 for vit_b.
    # default: 8
    'cba_last_n': 8,

    # CBA fusion: f_vit + ((1 - C_weight)*f_cnn + f_gated_cnn) / 2, where
    # C_weight is the per-channel linear CKA between same-index channels of
    # f_vit and f_cnn, and f_gated_cnn = AttentionGate(f_cnn, f_vit) uses
    # f_vit as the gating signal. C_weight in [0, 1] (1 = aligned,
    # 0 = uncorrelated), so the CNN branch contributes most where the two
    # branches are least aligned. Off-diagonal channel pairs are not used.
    # True  -> CKA + attention-gated fusion (DSR-SAM2)
    # False -> f_vit + f_cnn (ablation: gate bypassed)
    'cka_fusion': True,

    # Paper-style additive AttentionGate (Oktay et al. 2018) at the 2n CBA
    # fusion points.  The gate is
    # sigmoid(psi(ReLU(W_x(f_cnn) + W_g(f_vit)))) applied to f_cnn, with 1x1
    # convs and biases, no BatchNorm.  CKA and all gate params are always
    # differentiable (no detach toggle).
    'use_attention_gate': True,
    'use_attention_gate_cba': True,
    'attention_gate_channels': 768,  # F_int; 768 = C for vit_b

    # Final ViT/CNN fusion (after the last ViT block, before the SAM neck):
    # plain x = x + cnn_final (no CKA, no attention gate).

    # L_CKA = normalized sum of every CBA C_weight term, added to the
    # training loss as cka_loss_weight * L_CKA.  The model exposes it as
    # outputs['cka_loss'].
    'compute_cka_loss': True,
    'cka_loss_weight': 0.1,

    # Saved artifacts: weights/dsr_sam2/<model_type>_<weight_name_tag>_s_<src>*
    # and results.json entries tagged with <result_name_tag>.
    'weight_name_tag': 'dsr_sam2_cka',
    'result_name_tag': 'dsr_sam2_cka',
}

DATASETS = ['OTU', 'OVATUS', 'USOVA']

SINGLE_SOURCE_RUNS = [
    ('OTU', ['OVATUS', 'USOVA']),
    ('OVATUS', ['OTU', 'USOVA']),
    ('USOVA', ['OTU', 'OVATUS']),
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

    device = torch.device(CONFIG['device'])
    runner = SequentialDomainRunner(device=device)

    for source, targets in runs:
        cleanup_resources(device)
        run_single_source(source, targets, CONFIG, runner)
        
