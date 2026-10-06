"""
SAMUS training (repo train.py + utils/loss_functions/sam_loss.py).

  * Loss: Mask_DC_and_BCE_loss = 0.2 * BCE (pos_weight=2) + 0.8 * Dice
    on the low-res logits (repo get_criterion, SAMUS branch).
  * Optimizer: Adam, base_lr = 5e-4 (repo train.py default), no scheduler
    (warmup=False by default in the repo).
  * The point prompt is a random foreground click per slice (see data.py).
"""
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.metrics import loss_ce, loss_dice, metric_dice_iou_prec_rec_hd95
from utils.checkpoint import save_trainable


class _MaskDiceLoss(nn.Module):
    """Repo MaskDiceLoss (sigmoid single-channel dice)."""

    def forward(self, net_output, target, sigmoid=False):
        if sigmoid:
            net_output = torch.sigmoid(net_output)
        assert net_output.size() == target.size()
        smooth = 1e-5
        intersect = torch.sum(net_output[:, 0] * target[:, 0])
        y_sum = torch.sum(target[:, 0] * target[:, 0])
        z_sum = torch.sum(net_output[:, 0] * net_output[:, 0])
        return 1 - (2 * intersect + smooth) / (z_sum + y_sum + smooth)


class SamusCriterion(nn.Module):
    """Repo Mask_DC_and_BCE_loss for SAMUS (dice_weight=0.8)."""

    def __init__(self, dice_weight=0.8):
        super().__init__()
        self.ce = nn.BCEWithLogitsLoss(pos_weight=torch.ones([1]) * 2)
        self.dc = _MaskDiceLoss()
        self.dice_weight = dice_weight

    def forward(self, net_output, target):
        low_res_logits = net_output['low_res_logits']
        if target.dim() == 4 and target.shape[1] == 1:
            pass
        elif target.dim() == 4:
            target = target[:, :1]
        if target.shape[-2:] != low_res_logits.shape[-2:]:
            target = F.interpolate(target.float(), size=low_res_logits.shape[-2:],
                                   mode='nearest')
        loss_ce = self.ce(low_res_logits, target.float())
        loss_dice = self.dc(low_res_logits, target.float(), sigmoid=True)
        loss = (1 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice
        return loss, loss_ce, loss_dice


def format_duration(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    running_ce = 0.0
    running_dice = 0.0

    for images, masks, pt, plabel in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        coords = pt.to(device, non_blocking=True)
        labels = plabel.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images, (coords, labels))
        loss, loss_ce, loss_dice = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_ce += loss_ce.item()
        running_dice += loss_dice.item()

    n = max(len(loader), 1)
    return running_loss / n, running_ce / n, running_dice / n


@torch.no_grad()
def validate_epoch(model, loader, device, criterion):
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

        outputs = model(images, (coords, labels))
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


def train_samus(model, train_loader, val_loader, device, cfg):
    model_dir = cfg['model_dir']
    prefix = cfg['prefix']
    n_epochs = cfg['n_epochs']
    base_lr = cfg['base_lr']
    patience = cfg.get('patience', 0)  # 0 = disabled

    os.makedirs(model_dir, exist_ok=True)
    best_path = os.path.join(model_dir, f'{prefix}_best.pth')
    last_path = os.path.join(model_dir, f'{prefix}_last.pth')

    criterion = SamusCriterion(dice_weight=0.8)
    criterion.ce.pos_weight = criterion.ce.pos_weight.to(device)

    optimizer = optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0, amsgrad=False)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    start_time = time.time()
    epoch_times = []

    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        print(f'Epoch [{epoch}/{n_epochs}]')
        sys.stdout.flush()

        train_loss, train_ce, train_dice = train_one_epoch(
            model, train_loader, device, optimizer, criterion)
        (val_epoch_loss, val_dice, val_iou, val_prec, val_rec,
         val_hd95) = validate_epoch(model, val_loader, device, criterion)

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
    print(f'\nSAMUS training complete. Best Val Loss: {best_val_loss:.4f} -> {best_path}')
    sys.stdout.flush()
    return best_val_loss
