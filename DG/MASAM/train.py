"""
MA-SAM training (repo MA-SAM/trainer_bbox.py).

  * Loss: hybrid segmentation loss L = 0.2 * CE + 0.8 * Dice
    (paper Sec. 4.2, alpha = 0.2, beta = 0.8; repo calc_loss with
    dice_param = 0.8). Binary adaptation: CE/Dice over the 2 mask-decoder
    channels (background / foreground).
  * Optimizer: AdamW (filtered to trainable params, betas (0.9, 0.999),
    weight_decay 0.1), base_lr = 8e-4 (repo train.py --base_lr).
  * LR schedule: linear warmup for 250 iterations, then exponential
    (poly) decay lr = base_lr * (1 - t/T)^7 (repo trainer_bbox.py).
  * Mixed precision (fp16 autocast + GradScaler) like the repo --use_amp.
"""
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

from utils.metrics import metric_dice_iou_prec_rec_hd95


class DiceLoss(nn.Module):
    """Repo utils.py DiceLoss (multi-class, squared-pred, softmax)."""

    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = 1.0 * (input_tensor == i)
            temp_prob[input_tensor == -100] = -100
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        mask = (target != -100)
        intersect = torch.sum(score * target * mask)
        y_sum = torch.sum(target * target * mask)
        z_sum = torch.sum(score * score * mask)
        loss = 1 - (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size()
        loss = 0.0
        for i in range(self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
        return loss / self.n_classes


def calc_loss(logits, target, ce_loss, dice_loss, dice_weight=0.8):
    """Repo calc_loss: L = (1 - dice_weight) * CE + dice_weight * Dice."""
    loss_ce = ce_loss(logits, target.long())
    loss_dice = dice_loss(logits, target, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice


def format_duration(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def train_one_epoch(model, loader, device, optimizer, scaler, ce_loss, dice_loss,
                    base_lr, warmup, warmup_period, max_iterations, lr_exp, iter_num):
    model.train()
    running_loss = 0.0
    running_ce = 0.0
    running_dice = 0.0
    lr_ = base_lr

    for images, masks, bbox in loader:
        images = images.unsqueeze(1).to(device, non_blocking=True)   # [B, 1, 3, H, W]
        masks = masks.to(device, non_blocking=True)
        bbox = bbox.to(device, non_blocking=True)
        targets = masks[:, 0]

        if warmup and iter_num < warmup_period:
            lr_ = base_lr * ((iter_num + 1) / warmup_period)
        else:
            shift_iter = iter_num - warmup_period if warmup else iter_num
            assert shift_iter >= 0
            lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** lr_exp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_
        iter_num += 1

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(images, multimask_output=False, image_size=images.shape[-1],
                            bbox_input=bbox)
            loss, loss_ce, loss_dice = calc_loss(outputs['low_res_logits'], targets,
                                                 ce_loss, dice_loss)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        running_ce += loss_ce.item()
        running_dice += loss_dice.item()

    n = max(len(loader), 1)
    return running_loss / n, running_ce / n, running_dice / n, lr_, iter_num


@torch.no_grad()
def validate_epoch(model, loader, device, ce_loss, dice_loss):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_precision = 0.0
    running_recall = 0.0
    running_hd95 = 0.0

    for images, masks, bbox in loader:
        images = images.unsqueeze(1).to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        bbox = bbox.to(device, non_blocking=True)
        targets = masks[:, 0]

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(images, multimask_output=False, image_size=images.shape[-1],
                            bbox_input=bbox)
            loss, _, _ = calc_loss(outputs['low_res_logits'], targets, ce_loss, dice_loss)
        running_loss += loss.item()

        probs = torch.softmax(outputs['low_res_logits'].float(), dim=1)[:, 1:2]
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


def train_masam(model, train_loader, val_loader, device, cfg):
    model_dir = cfg['model_dir']
    prefix = cfg['prefix']
    n_epochs = cfg['n_epochs']
    base_lr = cfg['base_lr']
    warmup_period = cfg['warmup_period']
    lr_exp = cfg['lr_exp']

    os.makedirs(model_dir, exist_ok=True)
    best_path = os.path.join(model_dir, f'{prefix}_best.pth')

    ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
    dice_loss = DiceLoss(n_classes=cfg['num_classes'] + 1)

    b_lr = base_lr / warmup_period if cfg['warmup'] else base_lr
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['use_amp'])

    max_iterations = n_epochs * len(train_loader)
    iter_num = 0
    best_val_loss = float('inf')

    start_time = time.time()
    epoch_times = []

    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        print(f'Epoch [{epoch}/{n_epochs}]')
        sys.stdout.flush()

        train_loss, train_ce, train_dice, lr_now, iter_num = train_one_epoch(
            model, train_loader, device, optimizer, scaler, ce_loss, dice_loss,
            base_lr, cfg['warmup'], warmup_period, max_iterations, lr_exp, iter_num)
        (val_epoch_loss, val_dice, val_iou, val_prec, val_rec,
         val_hd95) = validate_epoch(model, val_loader, device, ce_loss, dice_loss)

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
              f'Val Loss: {val_epoch_loss:.4f} | Best Val Loss: {best_val_loss:.4f}')
        print(f'  Dice: {val_dice:.2f}  IoU: {val_iou:.2f}  Prec: {val_prec:.2f}  '
              f'Rec: {val_rec:.2f}  HD95: {val_hd95:.2f}')
        print(f'  Lr: {lr_now:.6f}')
        print(f'  Time: {format_duration(epoch_time)} | Avg: {format_duration(avg_time)} | '
              f'Elapsed: {format_duration(elapsed)} | Remaining: {format_duration(remaining)}')
        if is_best:
            print(f'  >>> New Best Validation Loss | Previous: {prev_str} | '
                  f'Current : {val_epoch_loss:.4f}')
        sys.stdout.flush()

    print(f'\nMA-SAM training complete. Best Val Loss: {best_val_loss:.4f} -> {best_path}')
    sys.stdout.flush()
    return best_path
