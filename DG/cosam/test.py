"""
CoSAM testing: iterative self-correcting loop on the target domain
(paper Alg. 2). No GT is used: the coarse mask is corrected by flipping
the error points of the binarized error map, prompts are rebuilt from the
corrected mask, and the loop stops early when the error count grows.
Metrics follow the other DG methods.
"""
import numpy as np
import torch

from DG.cosam.data import get_cosam_target_loader
from DG.cosam.prompts import binarize, topk_points, batch_cc_boxes
from utils.metrics import metric_dice_iou_prec_rec_hd95, save_results


@torch.no_grad()
def refine_loop(model, images, cfg):
    """Alg. 2 refinement; returns the final probability masks."""
    K = cfg['num_points']
    n_iters = cfg['num_iters']
    image_size = images.shape[-1]

    image_emb = model.encode_image(images)
    coarse, _ = model.decode_coarse(image_emb, image_size)
    cur = torch.sigmoid(coarse)
    n_prev = float('inf')

    for _ in range(n_iters):
        bin_cur = (cur > 0.5).float()
        err_pred = model.predict_error(image_emb, bin_cur)
        bin_err = (torch.sigmoid(err_pred) > 0.5).float()
        n_w = float(bin_err.sum().item())
        if n_w >= n_prev:
            break
        # Correct the mask by inverting the error points.
        corrected = torch.where(bin_err > 0.5, 1 - bin_cur, bin_cur)
        pt_coords, pt_labels = topk_points(torch.sigmoid(err_pred),
                                           corrected, K)
        box = batch_cc_boxes(corrected, image_size)
        sparse, _ = model.prompt_encoder(points=(pt_coords, pt_labels),
                                         boxes=box, masks=None)
        low_r, _ = model.mask_decoder(
            image_embeddings=image_emb,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=model.mask_embedding(corrected),
            multimask_output=False)
        cur = torch.sigmoid(model.postprocess_masks(
            low_r, input_size=(image_size, image_size),
            original_size=(image_size, image_size)))
        n_prev = n_w
    return cur


@torch.no_grad()
def test_cosam_on_target(model, target_name, device, image_size, batch_size,
                         num_workers, pin_memory, source_name, model_type='vit_b',
                         write_results=True, weight_tag='best', cfg=None):
    if cfg is None:
        cfg = {'num_points': 64, 'num_iters': 4}
    model.eval()
    loader = get_cosam_target_loader(target_name, image_size, batch_size,
                                     num_workers, pin_memory, split='test')

    running_dice = 0.0
    running_iou = 0.0
    running_precision = 0.0
    running_recall = 0.0
    running_hd95 = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        probs = refine_loop(model, images, cfg)

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

    name = f'{model_type}_cosam_s_{source_name}_t_{target_name}_{weight_tag}'

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
