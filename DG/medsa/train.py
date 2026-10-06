"""
MedSA training (repo train.py + function.py, 2D sam_adpt mode).

  * Loss: BCEWithLogitsLoss(pos_weight=2) on the decoder logits
    interpolated to the image size (repo function.py, thd=False branch).
  * Optimizer: Adam over trainable params (adapters + full mask decoder +
    prompt encoder), base_lr = 1e-4 (repo cfg.py --lr).
  * Scheduler: StepLR(step_size=10, gamma=0.5), stepped per epoch
    (repo train.py).
"""
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

from utils.metrics import loss_ce, loss_dice, metric_dice_iou_prec_rec_hd95
from utils.checkpoint import save_trainable


def format_duration(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def train_one_epoch(model, loader, device, optimizer, bce):
    model.train()
    running_loss = 0.0

    for images, masks, pt, plabel in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        coords = pt.to(device, non_blocking=True)
        labels = plabel.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images, (coords, labels), images.shape[-1])
        loss = bce(outputs['masks'], masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    n = max(len(loader), 1)
    return running_loss / n


@torch.no_grad()
def validate_epoch(model, loader, device, bce):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_precision = 0.0
    running_recall = 0.0
    running_hd95 = 0.0

    for images, masks, pt, plabel in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        coords = pt.to(device, non_blocking=True)
        labels = plabel.to(device, non_blocking=True)

        outputs = model(images, (coords, labels), images.shape[-1])
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


def train_medsa(model, train_loader, val_loader, device, cfg):
    model_dir = cfg['model_dir']
    prefix = cfg['prefix']
    n_epochs = cfg['n_epochs']
    patience = cfg.get('patience', 0)  # 0 = disabled

    os.makedirs(model_dir, exist_ok=True)
    best_path = os.path.join(model_dir, f'{prefix}_best.pth')
    last_path = os.path.join(model_dir, f'{prefix}_last.pth')

    bce = nn.BCEWithLogitsLoss(pos_weight=torch.ones([1]).to(device) * 2)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                  model.parameters()),
                           lr=cfg['base_lr'], betas=(0.9, 0.999), eps=1e-08,
                           weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=cfg['lr_step'],
                                          gamma=cfg['lr_gamma'])

    best_val_loss = float('inf')
    epochs_no_improve = 0

    start_time = time.time()
    epoch_times = []

    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        print(f'Epoch [{epoch}/{n_epochs}]')
        sys.stdout.flush()

        train_loss = train_one_epoch(model, train_loader, device, optimizer,
                                     bce)
        (val_epoch_loss, val_dice, val_iou, val_prec, val_rec,
         val_hd95) = validate_epoch(model, val_loader, device, bce)
        scheduler.step()

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

        print(f'  Train Loss: {train_loss:.4f} | '
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
        save_trainable(model, last_path)
        print(f'Last epoch weights saved (filename: {os.path.basename(last_path)}) -> {last_path}')
    print(f'\nMedSA training complete. Best Val Loss: {best_val_loss:.4f} -> {best_path}')
    sys.stdout.flush()
    return best_val_loss
