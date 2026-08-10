import json
import torch
import torch.nn.functional as F
import numpy as np

try:
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn
except ImportError:
    ssim_fn = None

try:
    from medpy.metric.binary import hd95 as medpy_hd95
except ImportError:
    medpy_hd95 = None

def metric_dice_iou_prec_rec_hd95(y_pred, y_true, with_hd95=True, threshold=0.5, eps=1e-7):
    B, C, H, W = y_pred.shape

    if threshold is not None:
        y_pred_bin = (y_pred > threshold).float()
        y_true_bin = (y_true > threshold).float()
    else:
        y_pred_bin = y_pred.float()
        y_true_bin = y_true.float()

    TP = (y_pred_bin * y_true_bin).sum(dim=(2, 3))
    FP = (y_pred_bin * (1 - y_true_bin)).sum(dim=(2, 3))
    FN = ((1 - y_pred_bin) * y_true_bin).sum(dim=(2, 3))

    dice = (2.0 * TP + eps) / (2.0 * TP + FP + FN + eps)
    iou = (TP + eps) / (TP + FP + FN + eps)
    precision = (TP + eps) / (TP + FP + eps)
    recall = (TP + eps) / (TP + FN + eps)

    dice_mean = dice.mean(dim=1).mean(dim=0).item()
    iou_mean = iou.mean(dim=1).mean(dim=0).item()
    prec_mean = precision.mean(dim=1).mean(dim=0).item()
    rec_mean = recall.mean(dim=1).mean(dim=0).item()

    hd95_mean = 0.0
    if with_hd95 and medpy_hd95 is not None:
        y_pred_np = y_pred_bin.detach().cpu().numpy().astype(bool)
        y_true_np = y_true_bin.detach().cpu().numpy().astype(bool)
        
        pred_sums = y_pred_np.sum(axis=(2, 3))
        true_sums = y_true_np.sum(axis=(2, 3))
        max_dist = np.sqrt(H**2 + W**2)
        
        hd95_vals = np.empty((B, C))
        for b in range(B):
            for c in range(C):
                if pred_sums[b, c] > 0 and true_sums[b, c] > 0:
                    hd95_vals[b, c] = medpy_hd95(y_pred_np[b, c], y_true_np[b, c])
                elif pred_sums[b, c] == 0 and true_sums[b, c] == 0:
                    hd95_vals[b, c] = 0.0
                else:
                    hd95_vals[b, c] = max_dist
        hd95_mean = hd95_vals.mean()

    return {"dice": dice_mean, "iou": iou_mean, "precision": prec_mean, "recall": rec_mean, "hd95": hd95_mean}

def loss_dice(y_pred, y_true, eps=1e-7):
    intersection = (y_pred * y_true).sum(dim=(2, 3))
    y_pred_sum = y_pred.sum(dim=(2, 3))
    y_true_sum = y_true.sum(dim=(2, 3))
    dice = (2.0 * intersection + eps) / (y_pred_sum + y_true_sum + eps)
    return (1.0 - dice).mean(dim=1).mean(dim=0)

def loss_iou(y_pred, y_true, eps=1e-7):
    intersection = (y_pred * y_true).sum(dim=(2, 3))
    union = y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return (1.0 - iou).mean(dim=1).mean(dim=0)

def loss_ce(y_pred, y_true, eps=1e-7):
    C = y_pred.shape[1]
    if C == 1:
        return F.binary_cross_entropy(y_pred, y_true, reduction='none').mean(dim=(2, 3)).mean(dim=1).mean(dim=0)
    else:
        return -(y_true * torch.log(y_pred + eps)).mean(dim=(2, 3)).mean(dim=1).mean(dim=0)

def loss_ssim(y_pred, y_true):
    B, C, H, W = y_pred.shape
    y_pred_flat = y_pred.reshape(B * C, 1, H, W)
    y_true_flat = y_true.reshape(B * C, 1, H, W)
    ssim_val = ssim_fn(y_pred_flat, y_true_flat, data_range=1.0, reduction='none')
    ssim_val = ssim_val.view(B, C)
    return (1.0 - ssim_val).mean(dim=1).mean(dim=0)

def loss_mse(y_pred, y_true):
    return F.mse_loss(y_pred, y_true, reduction='none').mean(dim=(2, 3)).mean(dim=1).mean(dim=0)


def save_results(name, results_dict, file_path='results.json'):
    existing_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            if isinstance(existing_data, dict):
                existing_data = [existing_data]
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    output_data = {'name': name, 'results': results_dict}
    existing_data.append(output_data)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=4)
