import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics import loss_ce, loss_dice
from utils.eval import evaluate
from utils.models import VGG16BN_Unet
from DG.Dofe.model import VGG16BN_DoFE


class DoFEWrapper(nn.Module):
    """Wraps VGG16BN_DoFE to return only the segmentation prediction.

    The evaluate() utility expects model(images) -> single tensor.
    DOFE model returns (prediction, domain_code, hal_scale, sel_scale).
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out, _, _, _ = self.model(x)
        return out


# ============================================================================
# Phase 1: Pre-training (vanilla VGG16BN_Unet, no DOFE modules)
# ============================================================================

def pretrain_one_epoch(model, loader, device, optimizer):
    model.train()
    total_loss = 0.0
    count = 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        preds = model(images)
        loss = 0.5 * loss_ce(preds, masks) + 0.5 * loss_dice(preds, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count += 1
    return total_loss / max(count, 1)


def format_duration(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def pretrain(model, train_loader, val_loader, device, optimizer, scheduler, n_epochs, model_dir, prefix):
    os.makedirs(model_dir, exist_ok=True)
    best_loss = float('inf')
    best_path = os.path.join(model_dir, f'{prefix}_pretrain.pth')
    start_time = time.time()
    epoch_times = []

    for epoch in range(n_epochs):
        epoch_start = time.time()
        print(f'Epoch [{epoch+1}/{n_epochs}]')
        sys.stdout.flush()

        train_loss = pretrain_one_epoch(model, train_loader, device, optimizer)

        val_loss, val_dice, val_iou, val_prec, val_rec, val_hd95 = evaluate(
            model=model, model_name='Unet', device=device,
            loader=val_loader, with_loss=True, with_hd95=True,
            print_results=False, write_results=False,
        )

        if val_loss < best_loss:
            prev_str = f'{best_loss:.4f}' if best_loss != float('inf') else 'N/A'
            best_loss = val_loss
            torch.save(model.state_dict(), best_path)

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        elapsed = time.time() - start_time
        avg_time = sum(epoch_times) / len(epoch_times)
        remaining = avg_time * (n_epochs - epoch - 1)

        print(f'  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Best Val Loss: {best_loss:.4f}')
        print(f'  Dice: {val_dice:.2f}  IoU: {val_iou:.2f}  Prec: {val_prec:.2f}  Rec: {val_rec:.2f}  HD95: {val_hd95:.2f}')
        print(f'  Epoch Time: {format_duration(epoch_time)} | Avg: {format_duration(avg_time)} | '
              f'Elapsed: {format_duration(elapsed)} | Remaining: {format_duration(remaining)}')
        if val_loss <= best_loss:
            print(f'  >>> New Best Validation Loss | Previous: {prev_str} | Current : {val_loss:.4f}')

        sys.stdout.flush()

    print(f'\n  Pretrain complete. Best loss: {best_loss:.4f} -> {best_path}')
    sys.stdout.flush()
    return best_path


# ============================================================================
# Phase 2: Centroid initialization from pretrained model features
# ============================================================================

def extract_features_for_centroids(pretrained_path, source_names, image_size, device, num_domains):
    model = VGG16BN_Unet(with_vgg16bn=True).to(device)
    model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True))
    model.eval()

    feat_hw = image_size // 16
    dofe_model = VGG16BN_DoFE(num_domains=num_domains, with_vgg16bn=True, feat_hw=feat_hw).to(device)
    dofe_model.load_encoder_weights(model.state_dict())

    features_dict = {d: [] for d in range(num_domains)}

    from DG.Dofe.data import _DoFEMultiSourceDataset
    dataset = _DoFEMultiSourceDataset(source_names, image_size, samples_per_domain=1)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    with torch.no_grad():
        for batch in loader:
            images, _, domain_labels, _ = batch
            images = images.squeeze(0).to(device)
            dl = domain_labels.squeeze(0)
            feat = dofe_model(images, extract_feature=True)
            for b in range(images.size(0)):
                d = dl[b].item()
                features_dict[d].append(feat[b:b+1].cpu())

    return dofe_model, features_dict


# ============================================================================
# Phase 3: Full DOFE training
# ============================================================================

def dofe_train_one_epoch(model, loader, device, optimizer, alpha=0.1, lam=0.9):
    model.train()
    total_seg_loss = 0.0
    total_dc_loss = 0.0
    count = 0

    for batch in loader:
        images, masks, domain_labels, soft_labels = batch
        images = images.squeeze(0).to(device, non_blocking=True)
        masks = masks.squeeze(0).to(device, non_blocking=True)
        domain_labels = domain_labels.squeeze(0).to(device)
        soft_labels = soft_labels.squeeze(0).to(device)

        optimizer.zero_grad(set_to_none=True)

        preds, domain_code, _, _ = model(
            images, domain_labels=domain_labels, update_memory=True, lam=lam,
        )

        loss_seg = 0.5 * loss_ce(preds, masks) + 0.5 * loss_dice(preds, masks)
        dc_probs = F.softmax(domain_code, dim=-1)
        loss_dc = alpha * F.mse_loss(dc_probs, soft_labels)

        loss = loss_seg + loss_dc
        loss.backward()
        optimizer.step()

        total_seg_loss += loss_seg.item()
        total_dc_loss += loss_dc.item()
        count += 1

    return total_seg_loss / max(count, 1), total_dc_loss / max(count, 1)


def train_dofe(model, train_loader, val_loader, device, optimizer, scheduler, n_epochs, model_dir, prefix, alpha=0.1, lam=0.9):
    os.makedirs(model_dir, exist_ok=True)
    best_loss = float('inf')
    best_path = os.path.join(model_dir, f'{prefix}_best.pth')
    val_wrapper = DoFEWrapper(model)
    start_time = time.time()
    epoch_times = []

    for epoch in range(n_epochs):
        epoch_start = time.time()
        print(f'Epoch [{epoch+1}/{n_epochs}]')
        sys.stdout.flush()

        seg_loss, dc_loss = dofe_train_one_epoch(model, train_loader, device, optimizer, alpha=alpha, lam=lam)

        val_loss, val_dice, val_iou, val_prec, val_rec, val_hd95 = evaluate(
            model=val_wrapper, model_name='DoFE', device=device,
            loader=val_loader, with_loss=True, with_hd95=True,
            print_results=False, write_results=False,
        )

        if val_loss < best_loss:
            prev_str = f'{best_loss:.4f}' if best_loss != float('inf') else 'N/A'
            best_loss = val_loss
            torch.save(model.state_dict(), best_path)

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        elapsed = time.time() - start_time
        avg_time = sum(epoch_times) / len(epoch_times)
        remaining = avg_time * (n_epochs - epoch - 1)

        print(f'  Train Loss: {seg_loss:.4f} (Seg: {seg_loss:.4f} DC: {dc_loss:.4f}) | Val Loss: {val_loss:.4f} | Best Val Loss: {best_loss:.4f}')
        print(f'  Dice: {val_dice:.2f}  IoU: {val_iou:.2f}  Prec: {val_prec:.2f}  Rec: {val_rec:.2f}  HD95: {val_hd95:.2f}')
        print(f'  Epoch Time: {format_duration(epoch_time)} | Avg: {format_duration(avg_time)} | '
              f'Elapsed: {format_duration(elapsed)} | Remaining: {format_duration(remaining)}')
        if val_loss <= best_loss:
            print(f'  >>> New Best Validation Loss | Previous: {prev_str} | Current : {val_loss:.4f}')

        sys.stdout.flush()

    print(f'  DOFE training complete. Best loss: {best_loss:.4f} -> {best_path}')
    sys.stdout.flush()
    return best_loss


# ============================================================================
# Source domain validation
# ============================================================================

def validate_dofe(model, val_loader, device):
    wrapper = DoFEWrapper(model)
    _, avg_dice, avg_iou, avg_prec, avg_rec, avg_hd95 = evaluate(
        model=wrapper, model_name='DoFE', device=device,
        loader=val_loader, with_loss=True, with_hd95=True,
        print_results=True, write_results=False,
    )
    return {'dice': avg_dice, 'iou': avg_iou, 'precision': avg_prec, 'recall': avg_rec, 'hd95': avg_hd95}
