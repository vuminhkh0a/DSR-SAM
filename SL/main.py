import sys
import time
import torch
import torch.optim as optim
from utils.seed import set_seed
from utils.data import get_dataloaders
from utils.metrics import loss_ce, loss_dice, save_results
from utils.eval import evaluate
from utils.models import VGG16BN_Unet

set_seed()

DATASETS = ['OTU', 'OVATUS', 'USOVA']

IMAGE_SIZE = 256
NUM_WORKERS = 4
PIN_MEMORY = True
DEVICE = 'cuda:0'
PHASE = 'train'
MODEL = 'Unet'
EPOCHS = 2
BATCH_SIZE = 8


def format_duration(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def train_one_epoch(model, loader, device, optimizer):
    model.train()
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        preds = model(images)
        loss = 0.5 * loss_ce(preds, masks) + 0.5 * loss_dice(preds, masks)
        loss.backward()
        optimizer.step()


def run_single_source(src, targets, device):
    best_model_path = f'weights/unet_s_{src}.pth'
    model = VGG16BN_Unet(with_vgg16bn=True).to(device)

    if PHASE == 'train':
        train_loader, valid_loader, _ = get_dataloaders(
            name=src, image_size=IMAGE_SIZE, transform=None,
            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

        print(f'\n--- Training on {src} ---')
        best_loss = float('inf')
        start_time = time.time()
        epoch_times = []
        for epoch in range(EPOCHS):
            epoch_start = time.time()
            sys.stdout.flush()
            train_one_epoch(model=model, loader=train_loader, device=device, optimizer=optimizer)
            val_loss, val_dice, val_iou, val_recall, val_precision, val_hd95 = evaluate(
                model=model, model_name=MODEL, device=device,
                loader=valid_loader, with_loss=True, with_hd95=True,
                print_results=False, write_results=False,
            )

            if val_loss < best_loss:
                prev_str = f'{best_loss:.4f}' if best_loss != float('inf') else 'N/A'
                print(f'\n  New Best Validation Loss!')
                print(f'  Previous: {prev_str}')
                print(f'  Current : {val_loss:.4f}')
                print(f'  Best model saved.\n')
                best_loss = val_loss
                torch.save(model.state_dict(), best_model_path)

            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            elapsed = time.time() - start_time
            avg_time = sum(epoch_times) / len(epoch_times)
            remaining = avg_time * (EPOCHS - epoch - 1)
            eta = time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time() + remaining))

            print(f'Epoch [{epoch+1}/{EPOCHS}] | Val Loss: {val_loss:.4f} | Best: {best_loss:.4f}')
            print(f'  Dice: {val_dice:.2f}  IoU: {val_iou:.2f}  Prec: {val_precision:.2f}  Rec: {val_recall:.2f}  HD95: {val_hd95:.2f}')
            print(f'  Time: {format_duration(epoch_time)} | Avg: {format_duration(avg_time)} | '
                  f'Elapsed: {format_duration(elapsed)} | Remaining: {format_duration(remaining)} | ETA: {eta}')

    model.load_state_dict(torch.load(best_model_path))

    for tgt in targets:
        _, _, test_loader = get_dataloaders(
            name=tgt, image_size=IMAGE_SIZE, transform=None,
            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        )

        print(f'\n--- Test on {tgt} ---')
        _, test_dice, test_iou, test_recall, test_precision, test_hd95 = evaluate(
            model=model, model_name=MODEL, device=device,
            loader=test_loader, with_loss=False, with_hd95=True,
            print_results=False, write_results=False,
        )
        print(f'Test Dice: {test_dice:.2f}')
        print(f'Test IoU: {test_iou:.2f}')
        print(f'Test Precision: {test_precision:.2f}')
        print(f'Test Recall: {test_recall:.2f}')
        print(f'Test HD95: {test_hd95:.2f}')

        save_results(f'unet_s_{src}_t_{tgt}', {
            'dice': round(test_dice, 2),
            'iou': round(test_iou, 2),
            'precision': round(test_precision, 2),
            'recall': round(test_recall, 2),
            'hd95': round(test_hd95, 2),
        })


if __name__ == '__main__':
    device = torch.device(DEVICE)
    for src in DATASETS:
        targets = [d for d in DATASETS if d != src]
        run_single_source(src, targets, device)
