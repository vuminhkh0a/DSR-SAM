"""
SAMMed testing on the held-out target domains (repo sam_4_preprocessed_bbox.py
test()): the saved refined bounding boxes (from the source-trained backbone)
prompt the fine-tuned SAM over merged 1024x1024 composites; predictions are
thresholded at theta_2 = 0.5 (paper Sec. 4.1) and full metrics are reported
(dice/iou/precision/recall/hd95), same as the other DG methods.
"""
import numpy as np
import torch

from utils.data import _make_loader
from utils.metrics import metric_dice_iou_prec_rec_hd95, save_results
from DG.SAMMed.data import SAMMedDataset, merge_batch, unmerge_batch, _collect_split


@torch.no_grad()
def predict_merged(model, images, boxes, device):
    """Merged-composite SAM inference: returns per-patch predictions
    (B, 1, 512, 512) in [0,1] (sigmoid)."""
    from DG.SAMMed.data import merge_batch
    mask_ph = torch.zeros_like(images[:, :1])
    merged_images, _, box_tensor, mbz, row_num, col_num = \
        merge_batch(images, mask_ph, boxes)
    merged_images = merged_images.to(device, non_blocking=True)

    image_embeddings = model.image_encoder(merged_images)
    dense_pe = model.prompt_encoder.get_dense_pe()
    predicted_masks = torch.zeros(mbz, 1, 1024, 1024, device=device)
    for idx in range(mbz):
        cur_embedding = image_embeddings[idx]
        cur_boxes = box_tensor[idx * 4:(idx + 1) * 4].to(device, non_blocking=True)
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=None, boxes=cur_boxes, masks=None)
        low_res, _ = model.mask_decoder(
            image_embeddings=cur_embedding.unsqueeze(0),
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        mask_predictions = model.postprocess_masks(
            low_res, input_size=[1024, 1024], original_size=[1024, 1024])
        for i in range(row_num):
            for j in range(col_num):
                predicted_masks[idx, :, i * 512:(i + 1) * 512,
                                j * 512:(j + 1) * 512] = \
                    mask_predictions[i * col_num + j, :,
                                     i * 512:(i + 1) * 512,
                                     j * 512:(j + 1) * 512]

    merged_pred = torch.sigmoid(predicted_masks)
    return unmerge_batch(merged_pred, mbz, row_num, col_num), merged_pred


def test_sammed(model, source_name, target_name, device, image_size=512,
                batch_size=4, num_workers=4, pin_memory=True, box_dir=None,
                thresh=0.5, write_results=True, print_results=True):
    model.eval()
    imgs, masks = _collect_split(target_name, image_size, 'test')
    dataset = SAMMedDataset(imgs, masks, box_dir, perturb=False, image_size=image_size)
    test_loader = _make_loader(dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=pin_memory)

    running_dice = 0.0
    running_iou = 0.0
    running_precision = 0.0
    running_recall = 0.0
    running_hd95 = 0.0

    for images, masks_b, boxes, _ in test_loader:
        images = images.to(device, non_blocking=True)
        masks_b = masks_b.to(device, non_blocking=True)
        boxes = boxes.to(device, non_blocking=True)

        preds, _ = predict_merged(model, images, boxes, device)
        preds_bin = (preds > thresh).float()

        results = metric_dice_iou_prec_rec_hd95(
            y_pred=preds_bin, y_true=masks_b, with_hd95=True, threshold=0.5)
        running_dice += results['dice']
        running_iou += results['iou']
        running_precision += results['precision']
        running_recall += results['recall']
        running_hd95 += results['hd95']

    n = len(test_loader)
    avg_dice = running_dice / n * 100
    avg_iou = running_iou / n * 100
    avg_precision = running_precision / n * 100
    avg_recall = running_recall / n * 100
    avg_hd95 = running_hd95 / n

    if print_results:
        print(f"Target {target_name} | Dice: {avg_dice:.2f} | IoU: {avg_iou:.2f} | "
              f"Prec: {avg_precision:.2f} | Rec: {avg_recall:.2f} | HD95: {avg_hd95:.2f}")

    if write_results:
        save_results(f'vit_b_sammed_s_{source_name}_t_{target_name}', {
            'dice': np.round(avg_dice, 2),
            'iou': np.round(avg_iou, 2),
            'precision': np.round(avg_precision, 2),
            'recall': np.round(avg_recall, 2),
            'hd95': np.round(avg_hd95, 2),
        })

    return avg_dice, avg_iou, avg_precision, avg_recall, avg_hd95
