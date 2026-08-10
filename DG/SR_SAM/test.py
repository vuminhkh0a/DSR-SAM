"""
SR-SAM testing: evaluate on a held-out target domain with bounding-box
prompts (standard box_coords.json). Metrics follow the other DG methods.
"""
import numpy as np
import torch

from DG.SR_SAM.data import get_sr_sam_target_loader
from utils.metrics import metric_dice_iou_prec_rec_hd95, save_results


@torch.no_grad()
def test_sr_sam_on_target(model, target_name, device, image_size, batch_size,
                          num_workers, pin_memory, source_name, write_results=True):
    model.eval()
    loader = get_sr_sam_target_loader(target_name, image_size, batch_size,
                                      num_workers, pin_memory, split='test')

    running_dice = 0.0
    running_iou = 0.0
    running_precision = 0.0
    running_recall = 0.0
    running_hd95 = 0.0

    for images, masks, bbox in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        bbox = bbox.to(device, non_blocking=True)

        outputs = model(images, multimask_output=False, image_size=images.shape[-1],
                        bbox_input=bbox, ema=False)
        probs = torch.sigmoid(outputs['masks'].float())

        results = metric_dice_iou_prec_rec_hd95(y_pred=probs, y_true=masks,
                                                with_hd95=True, threshold=0.5)
        running_dice += results['dice']
        running_iou += results['iou']
        running_precision += results['precision']
        running_recall += results['recall']
        running_hd95 += results['hd95']

    n = len(loader)
    avg_dice = running_dice / n * 100
    avg_iou = running_iou / n * 100
    avg_precision = running_precision / n * 100
    avg_recall = running_recall / n * 100
    avg_hd95 = running_hd95 / n

    name = f'vit_b_sr_sam_s_{source_name}_t_{target_name}'

    print(f'Target {target_name} | Dice: {avg_dice:.2f} | IoU: {avg_iou:.2f} | '
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
