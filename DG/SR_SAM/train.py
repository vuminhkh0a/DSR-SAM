"""
SR-SAM training (paper Sec. 2.2-2.3 + repo sam_lora_image_encoder.py).

  * Loss: L = L_seg + lambda * L_distill, with L_seg = CE + Dice (paper
    Sec. 2.3, "combination of cross entropy loss and dice loss") computed
    on the 256-res sigmoid predictions, and L_distill = mean binary KL
    divergence between the EMA teacher and the LoRA student (Eq. 4).
  * Optimizer: AdamW (filtered to trainable params, weight_decay 0.1),
    base_lr = 5e-4 with a linear warm-up of 250 iterations (paper Sec. 3).
  * EMA LoRA: updated every iteration with rate alpha = 0.999 (paper).
  * Subspace regularization: every `truncation_period` epochs (first after
    `dash_warm` iterations, repo run_CVC-ClinicDB.sh --Dash_warm 300),
    TSD identification + truncation with truncation size s = 96 (paper
    Sec. 3, Table 4).
"""
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

from DG.SR_SAM.model import ema_update
from utils.metrics import loss_ce, loss_dice, metric_dice_iou_prec_rec_hd95


class SegLoss(nn.Module):
    """L_seg = CE + Dice on sigmoid probabilities."""

    def __init__(self):
        super(SegLoss, self).__init__()

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        ce = loss_ce(probs, targets)
        dice = loss_dice(probs, targets)
        return ce + dice, ce, dice


def kd_loss(student_logits, teacher_logits):
    """Binary KL divergence KL(p_t || p_s) averaged over all pixels (Eq. 4)."""
    eps = 1e-7
    p_s = torch.sigmoid(student_logits)
    p_t = torch.sigmoid(teacher_logits).detach()
    kl = (p_t * torch.log((p_t + eps) / (p_s + eps))
          + (1 - p_t) * torch.log((1 - p_t + eps) / (1 - p_s + eps)))
    return kl.mean()


@torch.no_grad()
def compute_teacher_logits(model, images, bbox, device):
    outputs = model(images, multimask_output=False, image_size=images.shape[-1],
                    bbox_input=bbox, ema=True)
    return outputs['low_res_logits']


def format_duration(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def train_one_epoch(model, loader, device, optimizer, seg_loss, cfg, iter_num):
    model.train()
    running_loss = 0.0
    running_ce = 0.0
    running_dice = 0.0
    running_kd = 0.0
    lr_ = cfg['base_lr']

    for images, masks, bbox in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        bbox = bbox.to(device, non_blocking=True)

        if iter_num < cfg['warmup_period']:
            lr_ = cfg['base_lr'] * ((iter_num + 1) / cfg['warmup_period'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_
        iter_num += 1

        optimizer.zero_grad()
        outputs = model(images, multimask_output=False, image_size=images.shape[-1],
                        bbox_input=bbox, ema=False)
        loss_seg, loss_ce_, loss_dice_ = seg_loss(outputs['masks'], masks)

        loss = loss_seg
        loss_kd = torch.tensor(0.0, device=device)
        if cfg['kd_weight'] > 0:
            teacher_logits = compute_teacher_logits(model, images, bbox, device)
            loss_kd = kd_loss(outputs['low_res_logits'], teacher_logits)
            loss = loss_seg + cfg['kd_weight'] * loss_kd

        loss.backward()
        optimizer.step()

        if cfg['ema_mode']:
            ema_update(model, cfg['ema_rate'])

        running_loss += loss.item()
        running_ce += loss_ce_.item()
        running_dice += loss_dice_.item()
        running_kd += loss_kd.item()

    n = max(len(loader), 1)
    return (running_loss / n, running_ce / n, running_dice / n,
            running_kd / n, lr_, iter_num)


@torch.no_grad()
def validate_epoch(model, loader, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_precision = 0.0
    running_recall = 0.0
    running_hd95 = 0.0

    for images, masks, bbox in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        bbox = bbox.to(device, non_blocking=True)

        outputs = model(images, multimask_output=False, image_size=images.shape[-1],
                        bbox_input=bbox, ema=False)
        probs = torch.sigmoid(outputs['masks'].float())
        # Validation loss is not defined in the paper/repo (train.py is not
        # released); use the benchmark convention (bce + dice) / 2.
        running_loss += ((loss_ce(probs, masks) + loss_dice(probs, masks)) / 2).item()

        results = metric_dice_iou_prec_rec_hd95(y_pred=probs, y_true=masks,
                                                with_hd95=True, threshold=0.5)
        running_dice += results['dice']
        running_iou += results['iou']
        running_precision += results['precision']
        running_recall += results['recall']
        running_hd95 += results['hd95']

    n = len(loader)
    return (running_loss / n, running_dice / n * 100, running_iou / n * 100,
            running_precision / n * 100, running_recall / n * 100, running_hd95 / n)


@torch.no_grad()
def truncate_tsds(model):
    """TSD identification + truncation on every LoRA qkv module."""
    truncated = 0
    for block in model.sam.image_encoder.blocks:
        qkv = block.attn.qkv
        if hasattr(qkv, 'update_base_weight'):
            top_q, top_v = qkv.update_base_weight(ema=True)
            truncated += top_q.numel() + top_v.numel()
    return truncated


def train_sr_sam(model, train_loader, val_loader, device, cfg):
    model_dir = cfg['model_dir']
    prefix = cfg['prefix']
    n_epochs = cfg['n_epochs']
    base_lr = cfg['base_lr']

    os.makedirs(model_dir, exist_ok=True)
    best_path = os.path.join(model_dir, f'{prefix}_best.pth')

    seg_loss = SegLoss()

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=base_lr, betas=(0.9, 0.999), weight_decay=0.1)

    iter_num = 0
    best_val_loss = float('inf')
    # Subspace regularization schedule: first truncation once `dash_warm`
    # iterations have elapsed (repo --Dash_warm 300), then every
    # `truncation_period` epochs (paper Sec. 3).
    first_truncation_epoch = math.ceil(cfg['dash_warm'] / len(train_loader))

    start_time = time.time()
    epoch_times = []

    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        print(f'Epoch [{epoch}/{n_epochs}]')
        sys.stdout.flush()

        (train_loss, train_ce, train_dice, train_kd,
         lr_now, iter_num) = train_one_epoch(
            model, train_loader, device, optimizer, seg_loss, cfg, iter_num)
        (val_epoch_loss, val_dice, val_iou, val_prec, val_rec,
         val_hd95) = validate_epoch(model, val_loader, device)

        truncated = 0
        if (cfg['truncation'] and epoch >= first_truncation_epoch
                and (epoch - first_truncation_epoch) % cfg['truncation_period'] == 0):
            truncated = truncate_tsds(model)

        is_best = val_epoch_loss < best_val_loss
        prev_str = f'{best_val_loss:.4f}' if best_val_loss != float('inf') else 'N/A'
        if is_best:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), best_path)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        elapsed = time.time() - start_time
        avg_time = sum(epoch_times) / len(epoch_times)
        remaining = avg_time * (n_epochs - epoch)

        print(f'  Train Loss: {train_loss:.4f} | CE: {train_ce:.4f} | Dice: {train_dice:.4f} | '
              f'KD: {train_kd:.6f} | Val Loss: {val_epoch_loss:.4f} | Best Val Loss: {best_val_loss:.4f}')
        print(f'  Dice: {val_dice:.2f}  IoU: {val_iou:.2f}  Prec: {val_prec:.2f}  '
              f'Rec: {val_rec:.2f}  HD95: {val_hd95:.2f}')
        print(f'  Lr: {lr_now:.6f} | Truncated TSDs: {truncated}')
        print(f'  Time: {format_duration(epoch_time)} | Avg: {format_duration(avg_time)} | '
              f'Elapsed: {format_duration(elapsed)} | Remaining: {format_duration(remaining)}')
        if is_best:
            print(f'  >>> New Best Validation Loss | Previous: {prev_str} | '
                  f'Current : {val_epoch_loss:.4f}')
        sys.stdout.flush()

    print(f'\nSR-SAM training complete. Best Val Loss: {best_val_loss:.4f} -> {best_path}')
    sys.stdout.flush()
    return best_path
