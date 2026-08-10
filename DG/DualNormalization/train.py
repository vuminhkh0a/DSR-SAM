import os
import sys
import time
import torch
import numpy as np
from utils.metrics import loss_ce, loss_dice, metric_dice_iou_prec_rec_hd95
from utils.eval import evaluate


def repeat_dataloader(loader):
    while True:
        for batch in loader:
            yield batch


def train_dn_epoch(model, loaders, device, optimizer):
    model.train()
    total_loss = 0.0
    total_count = 0

    # Single combined loader (single-source mode) yields list of domain batches
    # Multiple loaders (multi-source mode): each is a separate domain
    if len(loaders) == 1:
        for batch in loaders[0]:
            batch_loss = 0.0
            for domain_id, (images, masks, _) in enumerate(batch):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                domain_label = torch.full((images.shape[0],), domain_id, dtype=torch.long, device=device)
                preds = model(images, domain_label=domain_label)
                loss = 0.5 * loss_ce(preds, masks) + 0.5 * loss_dice(preds, masks)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                total_count += 1
            total_loss += batch_loss
    else:
        iters = [repeat_dataloader(ld) for ld in loaders]
        for batch in loaders[0]:
            batch_loss = 0.0
            for domain_id in range(len(loaders)):
                images, masks, _ = next(iters[domain_id])
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                domain_label = torch.full((images.shape[0],), domain_id, dtype=torch.long, device=device)
                preds = model(images, domain_label=domain_label)
                loss = 0.5 * loss_ce(preds, masks) + 0.5 * loss_dice(preds, masks)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                total_count += 1
            total_loss += batch_loss

    return total_loss / max(total_count, 1)


def validate_epoch(model, val_loader, device, domain_id=0):
    avg_loss, avg_dice, avg_iou, avg_prec, avg_rec, avg_hd95 = evaluate(
        model=model, model_name='Unet2D_DN', device=device,
        loader=val_loader, with_loss=True, with_hd95=True,
        print_results=False, write_results=False,
        domain_label=domain_id,
    )
    return avg_loss, avg_dice, avg_iou, avg_prec, avg_rec, avg_hd95


def format_duration(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def train_dn(model, loaders, val_loader, device, optimizer, scheduler, n_epochs, model_dir, prefix='dn'):
    os.makedirs(model_dir, exist_ok=True)
    best_loss = float('inf')
    best_path = os.path.join(model_dir, f'{prefix}_best.pth')
    start_time = time.time()
    epoch_times = []

    for epoch in range(n_epochs):
        epoch_start = time.time()
        print(f'Epoch [{epoch+1}/{n_epochs}]')
        sys.stdout.flush()

        train_loss = train_dn_epoch(model, loaders, device, optimizer)

        val_loss, val_dice, val_iou, val_prec, val_rec, val_hd95 = validate_epoch(
            model, val_loader, device, domain_id=0,
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

    print(f'\n  Training complete. Best val loss: {best_loss:.4f} -> {best_path}')
    sys.stdout.flush()
    return best_loss


def validate_dn(model, val_loader, device, domain_id=0):
    model.eval()
    all_dices, all_ious, all_precs, all_recs, all_hd95s = [], [], [], [], []

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            domain_label = torch.full((images.shape[0],), domain_id, dtype=torch.long, device=device)
            preds = model(images, domain_label=domain_label)
            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()
            for b in range(images.shape[0]):
                res = metric_dice_iou_prec_rec_hd95(
                    torch.from_numpy(preds_np[b:b+1]),
                    torch.from_numpy(masks_np[b:b+1]),
                    with_hd95=True, threshold=0.5, eps=1e-7,
                )
                all_dices.append(res['dice'])
                all_ious.append(res['iou'])
                all_precs.append(res['precision'])
                all_recs.append(res['recall'])
                all_hd95s.append(res['hd95'])

    dice = 100.0 * np.mean(all_dices)
    iou = 100.0 * np.mean(all_ious)
    prec = 100.0 * np.mean(all_precs)
    rec = 100.0 * np.mean(all_recs)
    hd95 = np.mean(all_hd95s)

    print(f'Source Val - Dice: {dice:.2f}  IoU: {iou:.2f}  Prec: {prec:.2f}  Rec: {rec:.2f}  HD95: {hd95:.2f}')
    sys.stdout.flush()
    return {'dice': dice, 'iou': iou, 'precision': prec, 'recall': rec, 'hd95': hd95}
