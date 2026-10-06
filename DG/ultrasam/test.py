"""
UltraSAM testing: box-prompt zero-shot evaluation of the official
checkpoint on a held-out target domain.

Forward runs at the native 1024 resolution (bilinear upsample, ImageNet
stats inside the model, boxes x4); probabilities are downsampled to 256
for the benchmark metrics. Metrics follow the other DG methods.
"""
import numpy as np
import torch
import torch.nn.functional as F

from DG.ultrasam.data import get_ultrasam_target_loader
from DG.ultrasam.model import NATIVE_SIZE
from utils.metrics import metric_dice_iou_prec_rec_hd95, save_results


@torch.no_grad()
def test_ultrasam_on_target(model, target_name, device, image_size,
                            batch_size, num_workers, pin_memory,
                            model_type='vit_b', write_results=True,
                            weight_tag='testonly'):
    model.eval()
    loader = get_ultrasam_target_loader(target_name, image_size, batch_size,
                                        num_workers, pin_memory, split='test')
    scale = NATIVE_SIZE / image_size

    running_dice = 0.0
    running_iou = 0.0
    running_precision = 0.0
    running_recall = 0.0
    running_hd95 = 0.0

    for images, masks, bbox in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        bbox = bbox.to(device, non_blocking=True) * scale

        images_1024 = F.interpolate(images, size=(NATIVE_SIZE, NATIVE_SIZE),
                                    mode='bilinear', align_corners=False)
        outputs = model(images_1024, bbox)
        probs = torch.sigmoid(outputs['masks'].float())
        probs = F.interpolate(probs, size=(image_size, image_size),
                              mode='bilinear', align_corners=False)

        results = metric_dice_iou_prec_rec_hd95(y_pred=probs, y_true=masks,
                                                with_hd95=True, threshold=0.5)
        running_dice += results['dice']
        running_iou += results['iou']
        running_precision += results['precision']
        running_recall += results['recall']
        running_hd95 += results['hd95']

    n = max(len(loader), 1)
    avg_dice = running_dice / n * 100
    avg_iou = running_iou / n * 100
    avg_precision = running_precision / n * 100
    avg_recall = running_recall / n * 100
    avg_hd95 = running_hd95 / n

    name = f'{model_type}_ultrasam_t_{target_name}_{weight_tag}'

    print(f'[weight: {weight_tag}] Target {target_name} | Dice: {avg_dice:.2f} | IoU: {avg_iou:.2f} | '
          f'Prec: {avg_precision:.2f} | Rec: {avg_recall:.2f} | HD95: {avg_hd95:.2f}')

    if write_results:
        save_results(name, {
            'dice': np.round(avg_dice, 2),
            'iou': np.round(avg_iou, 2),
            'precision': np.round(avg_precision, 2),
            'recall': np.round(avg_recall, 2),
            'hd95': np.round(avg_hd95, 2),
        })

    return avg_dice, avg_iou, avg_precision, avg_recall, avg_hd95
