"""
CoSAM training (paper Alg. 1).

Per batch, four phases share one encoder forward (frozen, no-grad):
  1. Coarse (prompt-free): Ls(y~, y).
  2. Error map (perturbed coarse mask, Eq. 3-4): Le(e~, e) (Eq. 5-6).
  3. Refined (top-K error points + largest-CC box + mask, Eq. 7): Ls(yr, y).
  4. Guided (GT points/box/mask, Eq. 8): Ls(yg, y).
Total (Eq. 9): Ls_coarse + lr_weight * Ls_ref + lg_weight * Ls_guide (+ Le;
M and E have disjoint params, so one Adam step on the sum matches the
paper's separate minima).

  * Ls = Dice + BCE (paper Eq. 2).
  * Optimizer: Adam, base_lr = 1e-4 with poly decay (1 - t/T)^0.9
    (paper Sec. 5.2, prostate setting).
"""
import os
import sys
import time

import torch
import torch.optim as optim

from utils.metrics import loss_ce, loss_dice, metric_dice_iou_prec_rec_hd95
from utils.checkpoint import save_trainable
from DG.cosam.prompts import (binarize, perturb_mask, error_label,
                              error_weight, topk_points, batch_cc_boxes,
                              random_gt_points)


def seg_loss(probs, masks):
    return loss_dice(probs, masks) + loss_ce(probs, masks)


def error_map_loss(err_probs, err, weight):
    eps = 1e-7
    p = err_probs.clamp(eps, 1 - eps)
    return -(weight * err * torch.log(p)
             + (1 - err) * torch.log(1 - p)).mean()


def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs) ** exponent


def format_duration(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def forward_passes(model, images, masks, cfg, train=True, rng=None):
    """Run the four Alg. 1 phases; return losses and the refined probs."""
    K = cfg['num_points']
    alpha = cfg['perturb_prob'] if train else 0.0
    image_size = images.shape[-1]

    with torch.no_grad():
        image_emb = model.encode_image(images)

    # 1. Coarse mask (prompt-free, Eq. 1).
    coarse, _ = model.decode_coarse(image_emb, image_size)
    probs_coarse = torch.sigmoid(coarse)
    loss_coarse = seg_loss(probs_coarse, masks)

    # 2. Error map (Eq. 3-6).
    bin_coarse = (probs_coarse > 0.5).float()
    pert = perturb_mask(bin_coarse, alpha)
    err_pred = model.predict_error(image_emb, pert)
    probs_err = torch.sigmoid(err_pred)
    err = error_label(bin_coarse, masks)
    loss_err = error_map_loss(probs_err, err, error_weight(err))

    # 3. Refined mask (Eq. 7): top-K error points, largest-CC box, mask.
    pt_coords, pt_labels = topk_points(probs_err.detach(), bin_coarse, K)
    box = batch_cc_boxes(bin_coarse, image_size)
    with torch.no_grad():
        sparse_r, _ = model.prompt_encoder(points=(pt_coords, pt_labels),
                                           boxes=box, masks=None)
    low_r, _ = model.mask_decoder(
        image_embeddings=image_emb,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_r,
        dense_prompt_embeddings=model.mask_embedding(bin_coarse),
        multimask_output=False)
    probs_ref = torch.sigmoid(model.postprocess_masks(
        low_r, input_size=(image_size, image_size),
        original_size=(image_size, image_size)))
    loss_ref = seg_loss(probs_ref, masks)

    # 4. Guided mask (Eq. 8): GT points/box/mask.
    g_coords, g_labels = random_gt_points(masks, K, rng=rng)
    g_box = batch_cc_boxes((masks > 0.5).float(), image_size)
    with torch.no_grad():
        sparse_g, _ = model.prompt_encoder(points=(g_coords, g_labels),
                                           boxes=g_box, masks=None)
    low_g, _ = model.mask_decoder(
        image_embeddings=image_emb,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_g,
        dense_prompt_embeddings=model.mask_embedding((masks > 0.5).float()),
        multimask_output=False)
    probs_guide = torch.sigmoid(model.postprocess_masks(
        low_g, input_size=(image_size, image_size),
        original_size=(image_size, image_size)))
    loss_guide = seg_loss(probs_guide, masks)

    return loss_coarse, loss_err, loss_ref, loss_guide, probs_ref


def train_one_epoch(model, loader, device, optimizer, cfg, epoch,
                    n_epochs):
    model.train()
    # Prompt encoder parts stay frozen; keep them in eval (no dropout/BN
    # updates anywhere, but stay explicit).
    model.prompt_encoder.eval()
    running = [0.0, 0.0, 0.0, 0.0]

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        lc, le, lr_, lg, _ = forward_passes(model, images, masks, cfg,
                                            train=True)
        loss = lc + cfg['lr_weight'] * lr_ + cfg['lg_weight'] * lg + le
        loss.backward()
        optimizer.step()

        running[0] += lc.item()
        running[1] += le.item()
        running[2] += lr_.item()
        running[3] += lg.item()

    for param_group in optimizer.param_groups:
        param_group['lr'] = poly_lr(epoch, n_epochs, cfg['base_lr'],
                                    cfg['lr_exp'])
    n = max(len(loader), 1)
    return (running[0] / n, running[1] / n, running[2] / n, running[3] / n,
            optimizer.param_groups[0]['lr'])


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

        lc, le, lr_, lg, probs_ref = forward_passes(
            model, images, masks, cfg, train=False)
        running_loss += (0.5 * loss_ce(probs_ref, masks) + 0.5 * loss_dice(probs_ref, masks)).item()

        results = metric_dice_iou_prec_rec_hd95(y_pred=probs_ref, y_true=masks,
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


def train_cosam(model, train_loader, val_loader, device, cfg):
    model_dir = cfg['model_dir']
    prefix = cfg['prefix']
    n_epochs = cfg['n_epochs']
    patience = cfg.get('patience', 0)  # 0 = disabled

    os.makedirs(model_dir, exist_ok=True)
    best_path = os.path.join(model_dir, f'{prefix}_best.pth')
    last_path = os.path.join(model_dir, f'{prefix}_last.pth')

    optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                  model.parameters()),
                           lr=cfg['base_lr'])

    best_val_loss = float('inf')
    epochs_no_improve = 0

    start_time = time.time()
    epoch_times = []

    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        print(f'Epoch [{epoch}/{n_epochs}]')
        sys.stdout.flush()

        train_coarse, train_err, train_ref, train_guide, lr_now = \
            train_one_epoch(model, train_loader, device, optimizer, cfg,
                            epoch, n_epochs)
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

        print(f'  Train Loss: {train_coarse + train_ref + train_guide:.4f} | '
              f'Coarse: {train_coarse:.4f} | Err: {train_err:.4f} | Ref: {train_ref:.4f} | '
              f'Guide: {train_guide:.4f} | Val Loss: {val_epoch_loss:.4f} | Best Val Loss: {best_val_loss:.4f}')
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
    print(f'\nCoSAM training complete. Best Val Loss: {best_val_loss:.4f} -> {best_path}')
    sys.stdout.flush()
    return best_val_loss
