import numpy as np
import torch
from torchvision.ops import batched_nms

from utils.data import get_target_loader
from utils.metrics import metric_dice_iou_prec_rec_hd95, save_results


def build_point_grid(n_per_side=9):
    """2D grid of points evenly spaced in [0,1]x[0,1] (repo utils/amg.py)."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    return np.stack([points_x, points_y], axis=-1).reshape(-1, 2)


def batched_mask_to_box(masks):
    """Boxes in XYXY format around masks; [0,0,0,0] for empty masks (repo utils/amg.py)."""
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)
    shape = masks.shape
    h, w = shape[-2:]
    if len(shape) > 2:
        masks = masks.flatten(0, -3)
    else:
        masks = masks.unsqueeze(0)

    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + h * (~in_height)
    top_edges, _ = torch.min(in_height_coords, dim=-1)

    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + w * (~in_width)
    left_edges, _ = torch.min(in_width_coords, dim=-1)

    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)

    if len(shape) > 2:
        out = out.reshape(*shape[:-2], 4)
    else:
        out = out[0]
    return out


def remove_small_regions(mask, area_thresh, mode):
    """Removes small disconnected regions/holes; requires cv2 (repo utils/amg.py)."""
    import cv2

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True


@torch.no_grad()
def generate_mask_from_gridpoints(model, image, device, grid=9, iou_thresh=0.5,
                                  box_nms_thresh=0.7, min_mask_region_area=1000,
                                  points_per_batch=27):
    """DeSAM automatic segmentation: 9x9 grid point prompts -> fused binary mask.

    Mirrors the repo's SamAutomaticMaskGenerator pipeline (points_per_side=grid,
    pred_iou_thresh, box NMS, min_mask_region_area postprocessing) with
    crop_n_layers=0 (edge filter is a no-op for the full-image crop).
    """
    image_embeddings = model.encode_image(image)            # each (1, C, 64, 64)
    dense_embeddings = model.prompt_encoder.get_dense_pe()  # (1, 256, 64, 64)

    grid_points = build_point_grid(grid)                    # (N, 2) in [0,1]
    coords = torch.as_tensor(grid_points * 256 * 4, device=device)   # 256-res -> 1024-res
    labels = torch.ones(coords.shape[0], dtype=torch.int, device=device)

    masks_logits_list, iou_preds_list = [], []
    for i in range(0, coords.shape[0], points_per_batch):
        batch_coords = coords[i:i + points_per_batch, None, :]
        batch_labels = labels[i:i + points_per_batch, None]
        n = batch_coords.shape[0]

        sparse_embeddings, _ = model.prompt_encoder(
            points=(batch_coords, batch_labels), boxes=None, masks=None,
        )
        # expand all features to the number of prompts (repo predictor.py)
        batch_embeddings = [torch.repeat_interleave(e, n, dim=0) for e in image_embeddings]

        masks_logits, iou_preds = model.mask_decoder(
            image_embeddings=batch_embeddings,
            image_pe=dense_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        masks_logits_list.append(masks_logits[:, 0])
        iou_preds_list.append(iou_preds[:, 0])

    masks_logits = torch.cat(masks_logits_list)             # (N, 256, 256)
    iou_preds = torch.cat(iou_preds_list)                   # (N,)

    keep = iou_preds > iou_thresh
    masks_logits = masks_logits[keep]
    if masks_logits.shape[0] == 0:
        return torch.zeros((256, 256), dtype=torch.bool, device=device)

    masks_bin = torch.sigmoid(masks_logits) > 0.5               # (N, 256, 256)
    scores = iou_preds[keep]

    boxes = batched_mask_to_box(masks_bin)
    keep_by_nms = batched_nms(boxes.float(), scores, torch.zeros(len(boxes), device=device),
                              iou_threshold=box_nms_thresh)
    masks_bin = masks_bin[keep_by_nms].cpu().numpy()

    final_mask = np.zeros((256, 256), dtype=bool)
    for m in masks_bin:
        m, _ = remove_small_regions(m, min_mask_region_area, mode="holes")
        m, _ = remove_small_regions(m, min_mask_region_area, mode="islands")
        final_mask = final_mask | m

    return torch.as_tensor(final_mask, device=device)


def test_desam(model, source_name, target_name, device, image_size=256, batch_size=1,
               num_workers=4, pin_memory=True, grid=9, iou_thresh=0.5, write_results=True,
               print_results=True):
    model.eval()
    test_loader = get_target_loader(target_name, image_size, batch_size, num_workers, pin_memory)

    running_dice = 0.0
    running_iou = 0.0
    running_precision = 0.0
    running_recall = 0.0
    running_hd95 = 0.0

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            pred = torch.zeros_like(masks)
            for i in range(images.shape[0]):
                mask = generate_mask_from_gridpoints(model, images[i:i + 1], device,
                                                     grid=grid, iou_thresh=iou_thresh)
                pred[i, 0] = mask

            results = metric_dice_iou_prec_rec_hd95(y_pred=pred, y_true=masks, with_hd95=True, threshold=0.5)
            running_dice += results['dice']
            running_iou += results['iou']
            running_precision += results['precision']
            running_recall += results['recall']
            running_hd95 += results['hd95']

    avg_dice = running_dice / len(test_loader) * 100
    avg_iou = running_iou / len(test_loader) * 100
    avg_precision = running_precision / len(test_loader) * 100
    avg_recall = running_recall / len(test_loader) * 100
    avg_hd95 = running_hd95 / len(test_loader)

    if print_results:
        print(f"Target {target_name} | Dice: {avg_dice:.2f} | IoU: {avg_iou:.2f} | "
              f"Prec: {avg_precision:.2f} | Rec: {avg_recall:.2f} | HD95: {avg_hd95:.2f}")

    if write_results:
        save_results(f'vit_h_desam_s_{source_name}_t_{target_name}', {
            'dice': np.round(avg_dice, 2),
            'iou': np.round(avg_iou, 2),
            'precision': np.round(avg_precision, 2),
            'recall': np.round(avg_recall, 2),
            'hd95': np.round(avg_hd95, 2),
        })

    return avg_dice, avg_iou, avg_precision, avg_recall, avg_hd95
