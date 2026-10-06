"""
Nora training (repo trainer.py).

  * Loss: L = (1 - lambda) * BCE + lambda * Dice on sigmoid probabilities
    (repo calc_loss, dice_param = 0.8; BCE/dice follow repo utils.py:
    BCELoss and global-batch dice with smooth 1.0).
  * Optimizer: AdamW over trainable params (adapters + prompt generator +
    mask decoder), base_lr = 5e-4, weight_decay 0.1 (repo train script).
  * LR schedule: linear warm-up for 250 iterations, then poly decay
    lr = base_lr * (1 - t/T)^0.9 (repo trainer.py).
"""
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

from utils.metrics import loss_ce, loss_dice, metric_dice_iou_prec_rec_hd95
from utils.checkpoint import save_trainable


def bce_loss(pred, label):
    return nn.BCELoss()(pred, label)


def dice_coeff(pred, label):
    """Repo utils.py dice_coeff (smooth=1.0, global batch sum)."""
    smooth = 1.0
    bs = pred.size(0)
    m1 = pred.contiguous().view(bs, -1)
    m2 = label.contiguous().view(bs, -1)
    intersection = (m1 * m2).sum()
    return 1 - (2.0 * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def calc_loss(outputs, masks, dice_weight=0.8):
    probs = torch.sigmoid(outputs['masks'])
    loss_ce = bce_loss(probs, masks)
    loss_dice = dice_coeff(probs, masks)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice


def format_duration(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def train_one_epoch(model, loader, device, optimizer, cfg, iter_num,
                    max_iterations):
    model.train()
    base_lr = cfg['base_lr']
    warmup = cfg['warmup']
    warmup_period = cfg['warmup_period']
    lr_exp = cfg['lr_exp']
    dice_weight = cfg['dice_weight']
    running_loss = 0.0
    running_ce = 0.0
    running_dice = 0.0
    lr_ = base_lr

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if warmup and iter_num < warmup_period:
            lr_ = base_lr * ((iter_num + 1) / warmup_period)
        else:
            if warmup:
                shift_iter = iter_num - warmup_period
                assert shift_iter >= 0
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** lr_exp
            else:
                shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** lr_exp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_
        iter_num += 1

        optimizer.zero_grad()
        outputs = model(images, multimask_output=False,
                        image_size=images.shape[-1])
        loss, loss_ce, loss_dice = calc_loss(outputs, masks, dice_weight)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_ce += loss_ce.item()
        running_dice += loss_dice.item()

    n = max(len(loader), 1)
    return running_loss / n, running_ce / n, running_dice / n, lr_, iter_num


@torch.no_grad()
def validate_epoch(model, loader, device, cfg):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_precision = 0.0
    running_recall = 0.0
    running_hd95 = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(images, multimask_output=False,
                        image_size=images.shape[-1])
        probs = torch.sigmoid(outputs['masks'].float())

        running_loss += (0.5 * loss_ce(probs, masks) + 0.5 * loss_dice(probs, masks)).item()
        results = metric_dice_iou_prec_rec_hd95(y_pred=probs, y_true=masks,
                                                with_hd95=True, threshold=0.5)
        running_dice += results['dice']
        running_iou += results['iou']
        running_precision += results['precision']
        running_recall += results['recall']
        running_hd95 += results['hd95']

    n = max(len(loader), 1)
    return (running_loss / n, running_dice / n * 100, running_iou / n * 100,
            running_precision / n * 100, running_recall / n * 100,
            running_hd95 / n)


def train_nora(model, train_loader, val_loader, device, cfg):
    model_dir = cfg['model_dir']
    prefix = cfg['prefix']
    n_epochs = cfg['n_epochs']
    base_lr = cfg['base_lr']
    patience = cfg.get('patience', 0)  # 0 = disabled

    os.makedirs(model_dir, exist_ok=True)
    best_path = os.path.join(model_dir, f'{prefix}_best.pth')
    last_path = os.path.join(model_dir, f'{prefix}_last.pth')

    if cfg['warmup']:
        init_lr = base_lr / cfg['warmup_period']
    else:
        init_lr = base_lr
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad,
                                   model.parameters()),
                            lr=init_lr, betas=(0.9, 0.999), weight_decay=0.1)

    max_iterations = n_epochs * len(train_loader)
    iter_num = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0

    start_time = time.time()
    epoch_times = []

    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        print(f'Epoch [{epoch}/{n_epochs}]')
        sys.stdout.flush()

        train_loss, train_ce, train_dice, lr_now, iter_num = train_one_epoch(
            model, train_loader, device, optimizer, cfg, iter_num,
            max_iterations)
        (val_epoch_loss, val_dice, val_iou, val_prec, val_rec,
         val_hd95) = validate_epoch(model, val_loader, device, cfg)

        is_best = val_epoch_loss < best_val_loss
        prev_str = f'{best_val_loss:.4f}' if best_val_loss != float('inf') else 'N/A'
        if is_best:
            best_val_loss = val_epoch_loss
            epochs_no_improve = 0
            save_trainable(model, best_path)
        else:
            epochs_no_improve += 1

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        elapsed = time.time() - start_time
        avg_time = sum(epoch_times) / len(epoch_times)
        remaining = avg_time * (n_epochs - epoch)

        print(f'  Train Loss: {train_loss:.4f} | CE: {train_ce:.4f} | Dice: {train_dice:.4f} | '
              f'Val Loss: {val_epoch_loss:.4f} | Best Val Loss: {best_val_loss:.4f}')
        print(f'  Dice: {val_dice:.2f}  IoU: {val_iou:.2f}  Prec: {val_prec:.2f}  '
              f'Rec: {val_rec:.2f}  HD95: {val_hd95:.2f}')
        print(f'  Lr: {lr_now:.6f}')
        print(f'  Time: {format_duration(epoch_time)} | Avg: {format_duration(avg_time)} | '
              f'Elapsed: {format_duration(elapsed)} | Remaining: {format_duration(remaining)}')
        if is_best:
            print(f'  >>> New Best Validation Loss | Previous: {prev_str} | '
                  f'Current : {val_epoch_loss:.4f}')
        if patience > 0:
            print(f'  Early stopping patience: {epochs_no_improve}/{patience}')
        sys.stdout.flush()

        if patience > 0 and epochs_no_improve > patience:
            print(f'  >>> Early stopping triggered after {epochs_no_improve} epochs without improvement')
            break

    if cfg.get('save_last_epoch', False):
        save_trainable(model, last_path)
        print(f'Last epoch weights saved (filename: {os.path.basename(last_path)}) -> {last_path}')
    print(f'\nNora training complete. Best Val Loss: {best_val_loss:.4f} -> {best_path}')
    sys.stdout.flush()
    return best_val_loss
