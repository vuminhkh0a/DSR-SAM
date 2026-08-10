import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from utils.metrics import loss_ce, loss_dice, loss_mse
from utils.eval import evaluate
from DG.MaxStyle.model import MaxStyleLayer


def format_duration(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def build_maxstyle_layers(batch_size, channel_map, cfg, device):
    layers = nn.ModuleDict()
    for layer_name in cfg['maxstyle_layers']:
        num_features = channel_map[layer_name]
        layers[layer_name] = MaxStyleLayer(
            batch_size=batch_size,
            num_features=num_features,
            p=cfg['maxstyle_p'],
            mix_style=True,
            no_noise=cfg['maxstyle_no_noise'],
            mix_learnable=True,
            noise_learnable=True,
            alpha=0.1,
            eps=1e-6,
        )
    return layers.to(device)


def train_one_epoch(model, loader, device, optimizer, cfg):
    model.train()
    total_loss = 0.0
    count = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        B = images.size(0)

        maxstyle_layers = build_maxstyle_layers(B, model.channel_map, cfg, device)

        adv_steps = cfg.get('adv_steps', 3)
        if adv_steps > 0 and len(list(maxstyle_layers.parameters())) > 0:
            ms_optimizer = optim.Adam(maxstyle_layers.parameters(), lr=cfg.get('adv_lr', 0.1))
            for _ in range(adv_steps):
                ms_optimizer.zero_grad()
                down1, down2, down3, down4, down5 = model.encode(images)
                recon_aug = model.img_decoder.forward_with_maxstyle(down5, maxstyle_layers)
                down1_a, down2_a, down3_a, down4_a, down5_a = model.encode(recon_aug)
                seg_aug = model.decode_seg(down1_a, down2_a, down3_a, down4_a, down5_a)
                adv_loss = 0.5 * loss_ce(seg_aug, masks) + 0.5 * loss_dice(seg_aug, masks)
                (-adv_loss).backward()
                ms_optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        down1, down2, down3, down4, down5 = model.encode(images)
        seg = model.decode_seg(down1, down2, down3, down4, down5)
        seg_loss = 0.5 * loss_ce(seg, masks) + 0.5 * loss_dice(seg, masks)

        recon = model.img_decoder.forward_with_maxstyle(down5, maxstyle_layers)
        recon_loss = loss_mse(recon, images)

        down1_a, down2_a, down3_a, down4_a, down5_a = model.encode(recon.detach())
        seg_aug = model.decode_seg(down1_a, down2_a, down3_a, down4_a, down5_a)
        aug_loss = 0.5 * loss_ce(seg_aug, masks) + 0.5 * loss_dice(seg_aug, masks)

        total = seg_loss + cfg['recon_w'] * recon_loss + cfg['aug_w'] * aug_loss
        total.backward()
        optimizer.step()

        total_loss += total.item()
        count += 1

    return total_loss / max(count, 1)


def train_maxstyle(model, train_loader, val_loader, device, cfg):
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

        train_loss = train_one_epoch(model, train_loader, device, optimizer, cfg)

        val_loss, val_dice, val_iou, val_prec, val_rec, val_hd95 = evaluate(
            model=model, model_name='MaxStyle', device=device,
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

    print(f'\nMaxStyle training complete. Best loss: {best_loss:.4f} -> {best_path}')
    sys.stdout.flush()
    return best_path
