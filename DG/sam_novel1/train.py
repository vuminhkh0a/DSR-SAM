"""
SAM + LoRA (rank 8) training: plain segmentation loss, no auxiliary terms.

  * Loss: L_seg = CE + Dice on the sigmoid predictions (same convention as
    the other DG baselines).
  * Optimizer: AdamW over trainable parameters (LoRA + mask decoder),
    base_lr = 5e-4, weight_decay = 0.1.  No scheduler, no EMA, no KD.
  * Only the best checkpoint (lowest validation loss) is kept by default.
"""
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

from utils.checkpoint import save_trainable
from utils.metrics import loss_ce, loss_dice, metric_dice_iou_prec_rec_hd95


class SegLoss(nn.Module):
    """L_seg = CE + Dice on sigmoid probabilities."""

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        ce = loss_ce(probs, targets)
        dice = loss_dice(probs, targets)
        return ce + dice, ce, dice


def format_duration(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def train_one_epoch(model, loader, device, optimizer, seg_loss):
    model.train()
    running_loss = 0.0
    running_ce = 0.0
    running_dice = 0.0

    for images, masks, bbox in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        bbox = bbox.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images, multimask_output=False,
                        image_size=images.shape[-1], bbox_input=bbox)
        loss, loss_ce_, loss_dice_ = seg_loss(outputs['masks'], masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_ce += loss_ce_.item()
        running_dice += loss_dice_.item()

    n = max(len(loader), 1)
    return running_loss / n, running_ce / n, running_dice / n


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

        outputs = model(images, multimask_output=False,
                        image_size=images.shape[-1], bbox_input=bbox)
        probs = torch.sigmoid(outputs['masks'].float())
        running_loss += ((loss_ce(probs, masks) + loss_dice(probs, masks)) / 2).item()

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


def train_sam_novel1(model, train_loader, val_loader, device, cfg):
    model_dir = cfg['model_dir']
    prefix = cfg['prefix']
    n_epochs = cfg['n_epochs']
    base_lr = cfg['base_lr']

    os.makedirs(model_dir, exist_ok=True)
    best_path = os.path.join(model_dir, f'{prefix}_best.pth')
    last_path = os.path.join(model_dir, f'{prefix}_last.pth')

    seg_loss = SegLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=base_lr, betas=(0.9, 0.999), weight_decay=0.1)

    best_val_loss = float('inf')
    patience = cfg.get('patience', 20)  # 0 = disabled
    epochs_no_improve = 0

    start_time = time.time()
    epoch_times = []

    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        print(f'Epoch [{epoch}/{n_epochs}]')
        sys.stdout.flush()

        train_loss, train_ce, train_dice = train_one_epoch(
            model, train_loader, device, optimizer, seg_loss)
        (val_epoch_loss, val_dice, val_iou, val_prec, val_rec,
         val_hd95) = validate_epoch(model, val_loader, device)

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

        print(f'  Train Loss: {train_loss:.4f} | CE: {train_ce:.4f} | '
              f'Dice: {train_dice:.4f} | Val Loss: {val_epoch_loss:.4f} | '
              f'Best Val Loss: {best_val_loss:.4f}')
        print(f'  Dice: {val_dice:.2f}  IoU: {val_iou:.2f}  '
              f'Prec: {val_prec:.2f}  Rec: {val_rec:.2f}  HD95: {val_hd95:.2f}')
        print(f'  Lr: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'  Time: {format_duration(epoch_time)} | '
              f'Avg: {format_duration(avg_time)} | '
              f'Elapsed: {format_duration(elapsed)} | '
              f'Remaining: {format_duration(remaining)}')
        if is_best:
            print(f'  >>> New Best Validation Loss | Previous: {prev_str} | '
                  f'Current : {val_epoch_loss:.4f}')
        if patience > 0:
            print(f'  Early stopping patience: {epochs_no_improve}/{patience}')
        sys.stdout.flush()

        if patience > 0 and epochs_no_improve >= patience:
            print(f'  >>> Early stopping triggered after {epochs_no_improve} '
                  f'epochs without improvement')
            break

    if cfg.get('save_last_epoch', False):
        save_trainable(model, last_path)
        print(f'Last epoch weights saved (filename: {os.path.basename(last_path)}) '
              f'-> {last_path}')
    print(f'\nSAM + LoRA training complete. Best Val Loss: {best_val_loss:.4f}')
    sys.stdout.flush()
    return best_val_loss
