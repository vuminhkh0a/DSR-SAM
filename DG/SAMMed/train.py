"""
SAMMed training (repo resnet_prostate.py + sam_4_preprocessed_bbox.py).

Stage 1 - segmentation backbone (DeepLabV3-ResNet50, output_stride=8,
ASPP [12, 24, 36], ImageNet-pretrained backbone): trained on the single
source domain with the repo's loss DiceCE(sigmoid=False, squared_pred=True)
+ BCE and Adam (lr = 0.001); coarse masks are thresholded at theta_1 = 0.75
for the mask-filtering module.

Stage 3 - SAM fine-tuning with refined bounding boxes: per the paper
(Sec. 3.4) and the repo training loop, ONLY the SAM ViT-B mask decoder is
fine-tuned - the repo wraps the image encoder and prompt encoder in
torch.no_grad() (Adam over all SAM params, lr = 1e-4, weight_decay = 1e-3,
fp32) with the merging strategy (4x512 -> 1024 composites) and the repo
loss DiceCE(sigmoid=True, squared_pred=True) (MONAI reimpl.).

The validation criterion is (BCE + Dice) / 2 (the paper/repo do not define
a validation loss; it also matches utils/eval.py of the benchmark).
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

from utils.metrics import loss_ce, loss_dice, metric_dice_iou_prec_rec_hd95


def dicece_loss(y_pred, y_true, sigmoid=True, smooth=1e-5):
    """MONAI DiceCELoss(squared_pred=True, reduction='mean') reimplementation."""
    p = torch.sigmoid(y_pred) if sigmoid else y_pred
    p2, t2 = p ** 2, y_true ** 2
    dims = (2, 3)
    intersection = (p2 * t2).sum(dims)
    denominator = p2.sum(dims) + t2.sum(dims)
    dice = 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)
    if sigmoid:
        ce = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='mean')
    else:
        ce = F.binary_cross_entropy(y_pred, y_true, reduction='mean')
    return dice.mean() + ce


def val_loss_bce_dice(y_pred, y_true):
    """(BCE + Dice) / 2 validation criterion."""
    return 0.5 * loss_ce(y_pred, y_true) + 0.5 * loss_dice(y_pred, y_true)


def format_duration(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


# ============================================================
# Stage 1: DeepLabV3-ResNet50 segmentation backbone
# ============================================================

def train_one_epoch_resnet(model, loader, device, optimizer):
    model.train()
    epoch_loss = 0.0
    running_dice = 0.0
    for images, masks, _, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        _, mask_predictions = model(images)
        mask_predictions = torch.sigmoid(mask_predictions)
        loss = dicece_loss(mask_predictions, masks, sigmoid=False) + \
            F.binary_cross_entropy(mask_predictions, masks, reduction='mean')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        results = metric_dice_iou_prec_rec_hd95(
            y_pred=mask_predictions, y_true=masks, with_hd95=False)
        running_dice += results['dice']
    return epoch_loss / len(loader), running_dice / len(loader) * 100


def validate_resnet(model, loader, device):
    model.eval()
    val_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_precision = 0.0
    running_recall = 0.0
    running_hd95 = 0.0
    with torch.no_grad():
        for images, masks, _, _ in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            _, preds = model(images)
            preds = torch.sigmoid(preds)
            val_loss += val_loss_bce_dice(preds, masks).item()
            results = metric_dice_iou_prec_rec_hd95(
                y_pred=preds, y_true=masks, with_hd95=True, threshold=0.5)
            running_dice += results['dice']
            running_iou += results['iou']
            running_precision += results['precision']
            running_recall += results['recall']
            running_hd95 += results['hd95']
    n = len(loader)
    return (val_loss / n,
            running_dice / n * 100, running_iou / n * 100,
            running_precision / n * 100, running_recall / n * 100,
            running_hd95 / n)


def train_resnet(model, train_loader, val_loader, device, n_epochs, lr,
                 model_dir, prefix):
    os.makedirs(model_dir, exist_ok=True)
    best_path = os.path.join(model_dir, f'{prefix}_best.pth')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=[0.9, 0.999])

    best_val_loss = float('inf')
    start_time = time.time()
    epoch_times = []

    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        print(f'Epoch [{epoch}/{n_epochs}]')
        sys.stdout.flush()

        train_loss, train_dice = train_one_epoch_resnet(model, train_loader, device, optimizer)
        (val_loss, val_dice, val_iou, val_prec, val_rec, val_hd95) = \
            validate_resnet(model, val_loader, device)

        is_best = val_loss < best_val_loss
        prev_str = f'{best_val_loss:.4f}' if best_val_loss != float('inf') else 'N/A'
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        elapsed = time.time() - start_time
        avg_time = sum(epoch_times) / len(epoch_times)
        remaining = avg_time * (n_epochs - epoch)

        print(f'  Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.2f} | '
              f'Val Loss: {val_loss:.4f} | Best Val Loss: {best_val_loss:.4f}')
        print(f'  Dice: {val_dice:.2f}  IoU: {val_iou:.2f}  Prec: {val_prec:.2f}  '
              f'Rec: {val_rec:.2f}  HD95: {val_hd95:.2f}')
        print(f'  Lr: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'  Time: {format_duration(epoch_time)} | Avg: {format_duration(avg_time)} | '
              f'Elapsed: {format_duration(elapsed)} | Remaining: {format_duration(remaining)}')
        if is_best:
            print(f'  >>> New Best Validation Loss | Previous: {prev_str} | '
                  f'Current : {val_loss:.4f}')
        sys.stdout.flush()

    print(f'\nSAMMed resnet training complete. Best Val Loss: {best_val_loss:.4f} -> {best_path}')
    sys.stdout.flush()
    return best_path


# ============================================================
# Stage 3: SAM fine-tuning with refined bboxes + merging strategy
# ============================================================

def sam_one_step(model, images, masks, boxes, device, dtype=None):
    """One merged forward/backward unit of the repo training loop:
    merge 4 patches -> 1024x1024 composite, prompt with the shifted boxes,
    decode per box and reconstruct the composite prediction. Like the repo,
    the image encoder and prompt encoder run under torch.no_grad() so only
    the mask decoder is fine-tuned (paper Sec. 3.4)."""
    from DG.SAMMed.data import merge_batch
    merged_images, merged_masks, box_tensor, mbz, row_num, col_num = \
        merge_batch(images, masks, boxes)
    merged_images = merged_images.to(device, non_blocking=True)
    merged_masks = merged_masks.to(device, non_blocking=True)

    with torch.no_grad():
        image_embeddings = model.image_encoder(merged_images)    # (mbz,256,64,64)
        dense_pe = model.prompt_encoder.get_dense_pe()
    predicted_masks = torch.zeros_like(merged_masks)
    for idx in range(mbz):
        cur_embedding = image_embeddings[idx]
        cur_boxes = box_tensor[idx * 4:(idx + 1) * 4].to(device, non_blocking=True)
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=None, boxes=cur_boxes, masks=None)
        low_res, _ = model.mask_decoder(
            image_embeddings=cur_embedding.unsqueeze(0),
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        mask_predictions = model.postprocess_masks(
            low_res, input_size=[1024, 1024], original_size=[1024, 1024])
        for i in range(row_num):
            for j in range(col_num):
                predicted_masks[idx, :, i * 512:(i + 1) * 512,
                                j * 512:(j + 1) * 512] = \
                    mask_predictions[i * col_num + j, :,
                                     i * 512:(i + 1) * 512,
                                     j * 512:(j + 1) * 512]
    return predicted_masks, merged_masks


def train_one_epoch_sam(model, loader, device, optimizer):
    model.train()
    epoch_loss = 0.0
    running_dice = 0.0
    steps = 0
    for images, masks, boxes, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        boxes = boxes.to(device, non_blocking=True)

        predicted_masks, merged_masks = sam_one_step(
            model, images, masks, boxes, device)
        loss = dicece_loss(predicted_masks, merged_masks, sigmoid=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        results = metric_dice_iou_prec_rec_hd95(
            y_pred=torch.sigmoid(predicted_masks.float()), y_true=merged_masks.float(),
            with_hd95=False)
        running_dice += results['dice']
        steps += 1
    return epoch_loss / steps, running_dice / steps * 100


def validate_epoch_sam(model, loader, device):
    model.eval()
    val_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_precision = 0.0
    running_recall = 0.0
    running_hd95 = 0.0
    with torch.no_grad():
        for images, masks, boxes, _ in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            boxes = boxes.to(device, non_blocking=True)
            predicted_masks, merged_masks = sam_one_step(
                model, images, masks, boxes, device)
            val_loss += val_loss_bce_dice(
                torch.sigmoid(predicted_masks.float()), merged_masks.float()).item()
            results = metric_dice_iou_prec_rec_hd95(
                y_pred=torch.sigmoid(predicted_masks.float()), y_true=merged_masks.float(),
                with_hd95=True, threshold=0.5)
            running_dice += results['dice']
            running_iou += results['iou']
            running_precision += results['precision']
            running_recall += results['recall']
            running_hd95 += results['hd95']
    n = len(loader)
    return (val_loss / n,
            running_dice / n * 100, running_iou / n * 100,
            running_precision / n * 100, running_recall / n * 100,
            running_hd95 / n)


def train_sam(model, train_loader, val_loader, device, n_epochs, lr,
              model_dir, prefix):
    os.makedirs(model_dir, exist_ok=True)
    best_path = os.path.join(model_dir, f'{prefix}_best.pth')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)

    best_val_loss = float('inf')
    start_time = time.time()
    epoch_times = []

    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        print(f'Epoch [{epoch}/{n_epochs}]')
        sys.stdout.flush()

        train_loss, train_dice = train_one_epoch_sam(model, train_loader, device, optimizer)
        (val_loss, val_dice, val_iou, val_prec, val_rec, val_hd95) = \
            validate_epoch_sam(model, val_loader, device)

        is_best = val_loss < best_val_loss
        prev_str = f'{best_val_loss:.4f}' if best_val_loss != float('inf') else 'N/A'
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        elapsed = time.time() - start_time
        avg_time = sum(epoch_times) / len(epoch_times)
        remaining = avg_time * (n_epochs - epoch)

        print(f'  Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.2f} | '
              f'Val Loss: {val_loss:.4f} | Best Val Loss: {best_val_loss:.4f}')
        print(f'  Dice: {val_dice:.2f}  IoU: {val_iou:.2f}  Prec: {val_prec:.2f}  '
              f'Rec: {val_rec:.2f}  HD95: {val_hd95:.2f}')
        print(f'  Lr: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'  Time: {format_duration(epoch_time)} | Avg: {format_duration(avg_time)} | '
              f'Elapsed: {format_duration(elapsed)} | Remaining: {format_duration(remaining)}')
        if is_best:
            print(f'  >>> New Best Validation Loss | Previous: {prev_str} | '
                  f'Current : {val_loss:.4f}')
        sys.stdout.flush()

    print(f'\nSAMMed SAM training complete. Best Val Loss: {best_val_loss:.4f} -> {best_path}')
    sys.stdout.flush()
    return best_path
