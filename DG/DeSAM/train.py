"""
DeSAM training (grid points mode, repo desam/training.py run_training).

The SAM ViT-H image encoder and the prompt encoder are frozen; only the
decoupled mask decoder (PRIM + PDMM) is fine-tuned. Loss per repo defaults:
  * DiceCE loss (MONAI DiceCELoss(sigmoid=True, squared_pred=True) reimpl.)
  * L1 loss on the PRIM IoU prediction
  total = dicece + iou
The learning rate follows the repo poly schedule: lr * (1 - t/T)^0.9.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from utils.metrics import metric_dice_iou_prec_rec_hd95


def dicece_loss(y_pred, y_true, smooth=1e-5):
    """MONAI DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean') reimplementation."""
    p = torch.sigmoid(y_pred)
    p2, t2 = p ** 2, y_true ** 2
    dims = (2, 3)
    intersection = (p2 * t2).sum(dims)
    denominator = p2.sum(dims) + t2.sum(dims)
    dice = 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)
    ce = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='mean')
    return dice.mean() + ce


def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs) ** exponent


def format_duration(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def train_one_epoch(model, loader, device, optimizer, iou_loss):
    model.mask_decoder.train()
    model.image_encoder.eval()
    model.prompt_encoder.eval()

    epoch_diceceloss = 0.0
    epoch_iouloss = 0.0

    for step, (images, masks, in_points, in_labels, iou_label) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.no_grad():
            image_embeddings = model.encode_image(images)
            points_torch = (in_points[:, None, :].to(device, non_blocking=True),
                            in_labels[:, None].to(device, non_blocking=True))
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=points_torch, boxes=None, masks=None,
            )

        mask_pred, iou_predictions = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        diceceloss = dicece_loss(mask_pred, masks)
        iouloss = iou_loss(iou_predictions, iou_label.to(device, non_blocking=True))
        loss = diceceloss + iouloss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_diceceloss += diceceloss.item()
        epoch_iouloss += iouloss.item()

    return epoch_diceceloss / len(loader), epoch_iouloss / len(loader)


def validate_epoch(model, loader, device):
    model.eval()
    val_epoch_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_precision = 0.0
    running_recall = 0.0
    running_hd95 = 0.0

    with torch.no_grad():
        for images, masks, in_points, in_labels, _ in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            image_embeddings = model.encode_image(images)
            points_torch = (in_points[:, None, :].to(device, non_blocking=True),
                            in_labels[:, None].to(device, non_blocking=True))
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=points_torch, boxes=None, masks=None,
            )
            mask_pred, _ = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            val_epoch_loss += dicece_loss(mask_pred, masks).item()

            results = metric_dice_iou_prec_rec_hd95(
                y_pred=torch.sigmoid(mask_pred), y_true=masks,
                with_hd95=True, threshold=0.5,
            )
            running_dice += results['dice']
            running_iou += results['iou']
            running_precision += results['precision']
            running_recall += results['recall']
            running_hd95 += results['hd95']

    n = len(loader)
    return (val_epoch_loss / n,
            running_dice / n * 100, running_iou / n * 100,
            running_precision / n * 100, running_recall / n * 100,
            running_hd95 / n)


def train_desam(model, train_loader, val_loader, device, optimizer, n_epochs, lr,
                model_dir, prefix):
    os.makedirs(model_dir, exist_ok=True)
    best_path = os.path.join(model_dir, f'{prefix}_best.pth')

    iou_loss = torch.nn.L1Loss()
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    start_time = time.time()
    epoch_times = []

    for epoch in range(1, n_epochs + 1):
        optimizer.param_groups[0]['lr'] = poly_lr(epoch - 1, n_epochs, lr, 0.9)

        epoch_start = time.time()
        print(f'Epoch [{epoch}/{n_epochs}]')
        sys.stdout.flush()

        train_epoch_loss, train_iou_loss = train_one_epoch(
            model, train_loader, device, optimizer, iou_loss,
        )
        (val_epoch_loss, val_dice, val_iou, val_prec, val_rec,
         val_hd95) = validate_epoch(model, val_loader, device)

        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)

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

        print(f'  Train Loss: {train_epoch_loss:.4f} | IoU Loss: {train_iou_loss:.4f} | '
              f'Val Loss: {val_epoch_loss:.4f} | Best Val Loss: {best_val_loss:.4f}')
        print(f'  Dice: {val_dice:.2f}  IoU: {val_iou:.2f}  Prec: {val_prec:.2f}  '
              f'Rec: {val_rec:.2f}  HD95: {val_hd95:.2f}')
        print(f'  Lr: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'  Time: {format_duration(epoch_time)} | Avg: {format_duration(avg_time)} | '
              f'Elapsed: {format_duration(elapsed)} | Remaining: {format_duration(remaining)}')
        if is_best:
            print(f'  >>> New Best Validation Loss | Previous: {prev_str} | '
                  f'Current : {val_epoch_loss:.4f}')
        sys.stdout.flush()

    print(f'\nDeSAM training complete. Best Val Loss: {best_val_loss:.4f} -> {best_path}')
    sys.stdout.flush()
    return train_losses, val_losses, best_path
