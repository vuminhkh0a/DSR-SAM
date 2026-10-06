"""
MaxStyle training (repo config/MICCAI2022_MaxStyle.json + src training).

Per batch:
  1. Standard forward -> seg loss (binary cross entropy, repo seg loss).
  2. Adversarial style search: freeze the net, attach fresh MaxStyle
     layers to the 4 encoder stages and maximize the seg loss w.r.t. the
     style params (n_iter Adam steps, style lr 0.1, repo max_style lr).
  3. Train the net (AdamW, lr 1e-4, repo learning lr/optimizer) on the
     stylized features (style params detached/frozen).
"""
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

from utils.metrics import loss_ce, loss_dice, metric_dice_iou_prec_rec_hd95
from DG.maxstyle.model import make_style_modules


def format_duration(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def seg_bce(logits, masks):
    return nn.BCEWithLogitsLoss()(logits, masks)


def set_net_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def train_one_epoch(model, loader, device, optimizer, cfg):
    model.train()
    style_n_iter = cfg['style_n_iter']
    style_lr = cfg['style_lr']
    running_loss = 0.0
    running_adv = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        B = images.shape[0]

        # 1. Standard forward (also wipes stale grads).
        optimizer.zero_grad()
        logits_clean = model(images)
        loss_clean = seg_bce(logits_clean, masks)

        # 2. Adversarial style search with the net frozen.
        set_net_grad(model, False)
        style_mods = make_style_modules(B, device)
        style_params = []
        for m in style_mods.values():
            style_params += [p for p in m.parameters()
                             if p.requires_grad]
        adv_loss_val = 0.0
        if style_params:
            opt_style = optim.Adam(style_params, lr=style_lr)
            for _ in range(style_n_iter):
                opt_style.zero_grad()
                logits_aug = model(images, style=style_mods)
                loss_adv = -seg_bce(logits_aug, masks)  # maximize seg loss
                loss_adv.backward()
                opt_style.step()
                adv_loss_val = -loss_adv.item()
            for p in style_params:
                p.requires_grad_(False)

        # 3. Train the net on clean + stylized features (repo total =
        # standard_loss + max_style hard-example loss).
        set_net_grad(model, True)
        optimizer.zero_grad()
        logits_styl = model(images, style=style_mods)
        loss = loss_clean + seg_bce(logits_styl, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_adv += adv_loss_val

    n = max(len(loader), 1)
    return running_loss / n, running_adv / n


@torch.no_grad()
def validate_epoch(model, loader, device):
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

        logits = model(images)
        probs = torch.sigmoid(logits.float())

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


def train_maxstyle(model, train_loader, val_loader, device, cfg):
    model_dir = cfg['model_dir']
    prefix = cfg['prefix']
    n_epochs = cfg['n_epochs']
    patience = cfg.get('patience', 0)  # 0 = disabled

    os.makedirs(model_dir, exist_ok=True)
    best_path = os.path.join(model_dir, f'{prefix}_best.pth')
    last_path = os.path.join(model_dir, f'{prefix}_last.pth')

    optimizer = optim.AdamW(model.parameters(), lr=cfg['base_lr'])

    best_val_loss = float('inf')
    epochs_no_improve = 0

    start_time = time.time()
    epoch_times = []

    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        print(f'Epoch [{epoch}/{n_epochs}]')
        sys.stdout.flush()

        train_loss, train_adv = train_one_epoch(model, train_loader, device,
                                                optimizer, cfg)
        (val_epoch_loss, val_dice, val_iou, val_prec, val_rec,
         val_hd95) = validate_epoch(model, val_loader, device)

        is_best = val_epoch_loss < best_val_loss
        prev_str = f'{best_val_loss:.4f}' if best_val_loss != float('inf') else 'N/A'
        if is_best:
            best_val_loss = val_epoch_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_no_improve += 1

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        elapsed = time.time() - start_time
        avg_time = sum(epoch_times) / len(epoch_times)
        remaining = avg_time * (n_epochs - epoch)

        print(f'  Train Loss: {train_loss:.4f} | Adv Seg: {train_adv:.4f} | '
              f'Val Loss: {val_epoch_loss:.4f} | Best Val Loss: {best_val_loss:.4f}')
        print(f'  Dice: {val_dice:.2f}  IoU: {val_iou:.2f}  Prec: {val_prec:.2f}  '
              f'Rec: {val_rec:.2f}  HD95: {val_hd95:.2f}')
        print(f'  Lr: {optimizer.param_groups[0]["lr"]:.6f}')
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
        torch.save(model.state_dict(), last_path)
        print(f'Last epoch weights saved (filename: {os.path.basename(last_path)}) -> {last_path}')
    print(f'\nMaxStyle training complete. Best Val Loss: {best_val_loss:.4f} -> {best_path}')
    sys.stdout.flush()
    return best_val_loss
