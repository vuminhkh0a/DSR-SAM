"""
Frequency Dropout Training

Single-source training with Frequency Dropout regularization.
FD kernels are regenerated every train step (random filter + params + dropout).
"""
import os
import sys
import time
import torch
import torch.optim as optim
from utils.metrics import loss_ce, loss_dice
from utils.eval import evaluate


def format_duration(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def train_one_epoch(model, loader, device, optimizer):
    """Single training epoch with per-iteration FD kernel refresh."""
    model.train()
    total_loss = 0.0
    count = 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        model.get_new_kernels()

        optimizer.zero_grad(set_to_none=True)
        preds = model(images)
        loss = 0.5 * loss_ce(preds, masks) + 0.5 * loss_dice(preds, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1
    return total_loss / max(count, 1)


def train_fd(model, train_loader, val_loader, device, cfg):
    """Full FD training loop with best-epoch checkpointing."""
    model_dir = cfg['model_dir']
    prefix = cfg['prefix']
    n_epochs = cfg['n_epochs']
    lr = cfg['lr']

    os.makedirs(model_dir, exist_ok=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    best_path = os.path.join(model_dir, f'{prefix}_best.pth')
    best_loss = float('inf')

    start_time = time.time()
    epoch_times = []

    for epoch in range(n_epochs):
        epoch_start = time.time()
        print(f'Epoch [{epoch+1}/{n_epochs}]')
        sys.stdout.flush()

        train_loss = train_one_epoch(model, train_loader, device, optimizer)

        val_loss, val_dice, val_iou, val_prec, val_rec, val_hd95 = evaluate(
            model=model, model_name='FD', device=device,
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
        print(f'  Time: {format_duration(epoch_time)} | Avg: {format_duration(avg_time)} | '
              f'Elapsed: {format_duration(elapsed)} | Remaining: {format_duration(remaining)}')
        if val_loss <= best_loss:
            print(f'  >>> New Best Validation Loss | Previous: {prev_str} | Current : {val_loss:.4f}')
        sys.stdout.flush()

    print(f'\nFD training complete. Best loss: {best_loss:.4f} -> {best_path}')
    sys.stdout.flush()
    return best_path
