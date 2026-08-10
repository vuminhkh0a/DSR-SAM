import torch
import numpy as np
from utils.metrics import *


def evaluate(model, model_name, device, loader, with_loss, with_hd95, print_results=True, write_results=False, domain_label=None):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_recall = 0.0
    running_precision = 0.0
    running_hd95 = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if domain_label is not None:
                dl = torch.full((images.shape[0],), domain_label, dtype=torch.long, device=device)
                preds = model(images, domain_label=dl)
            else:
                preds = model(images)

            if with_loss:
                loss = 0.5 * loss_ce(y_pred=preds, y_true=masks) + 0.5 * loss_dice(y_pred=preds, y_true=masks)     
                running_loss += loss.item()
            results = metric_dice_iou_prec_rec_hd95(y_pred=preds, y_true=masks, with_hd95=with_hd95, threshold=0.5, eps=1e-7)
            running_dice += results['dice']
            running_iou += results['iou']
            running_precision += results['precision']
            running_recall += results['recall']
            running_hd95 += results['hd95']

    avg_loss = running_loss / len(loader) if with_loss else 0.0
    avg_dice = running_dice / len(loader) * 100                 
    avg_iou = running_iou / len(loader) * 100 
    avg_recall = running_recall / len(loader) * 100           
    avg_precision = running_precision / len(loader) * 100 
    avg_hd95 = running_hd95 / len(loader)

    if print_results:
        print(f'Validation loss: {avg_loss:.4f} | Dice: {avg_dice:.2f} | IoU: {avg_iou:.2f} | Prec: {avg_precision:.2f} | Rec: {avg_recall:.2f} | HD95: {avg_hd95:.2f}')

    if write_results:
        save_results(model_name, {
            'loss': np.round(avg_loss, 2),
            'dice': np.round(avg_dice, 2),
            'iou': np.round(avg_iou, 2),
            'precision': np.round(avg_precision, 2),
            'recall': np.round(avg_recall, 2),
            'hd95': np.round(avg_hd95, 2),
        })

    return avg_loss, avg_dice, avg_iou, avg_recall, avg_precision, avg_hd95
