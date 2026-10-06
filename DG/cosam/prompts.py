"""
CoSAM prompt utilities (paper Sec. 4.4-4.5).

Shared by train.py (Alg. 1: refined prompts from the error map, guided
prompts from the GT) and test.py (Alg. 2: iterative correction loop).
Boxes come from the masks themselves (largest foreground connected
region), so the benchmark box_coords.json is not needed.
"""
import cv2
import numpy as np
import torch


def binarize(logits, thresh=0.5):
    return (torch.sigmoid(logits) > thresh).float()


def perturb_mask(bin_mask, alpha):
    """Bernoulli flip perturbation (paper Eq. 3, train only)."""
    if alpha <= 0:
        return bin_mask
    psi = (torch.rand_like(bin_mask) < alpha).float()
    return bin_mask * (1 - psi) + (1 - bin_mask) * psi


def error_label(bin_coarse, gt):
    """XOR label: correct=0, error=1 (paper Sec. 4.3)."""
    return (bin_coarse != (gt > 0.5).float()).float()


def error_weight(err):
    """Balanced weight w = log((nw + nr) / nw) (paper Eq. 6)."""
    nw = float((err > 0.5).sum().item())
    nr = float((err <= 0.5).sum().item())
    if nw == 0:
        return 1.0
    return float(np.log((nw + nr) / nw))


def topk_points(err_probs, bin_mask, k):
    """Top-K error points with labels from the binary mask.

    err_probs/bin_mask: [B,1,H,W] -> coords [B,K,2] (x, y), labels [B,K].
    """
    B, _, H, W = err_probs.shape
    flat_err = err_probs.reshape(B, -1)
    flat_bin = bin_mask.reshape(B, -1)
    k = min(k, flat_err.shape[1])
    idx = torch.topk(flat_err, k, dim=1).indices
    ys = (idx // W).float()
    xs = (idx % W).float()
    coords = torch.stack([xs, ys], dim=-1)
    labels = flat_bin.gather(1, idx).long()
    return coords, labels


def _largest_cc_box(mask_np, image_size):
    """Min bounding box [x1,y1,x2,y2] of the largest foreground CC."""
    m = (mask_np > 0.5).astype(np.uint8)
    if m.sum() == 0:
        return np.array([0, 0, image_size, image_size], dtype=np.float32)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.array([0, 0, image_size, image_size], dtype=np.float32)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = stats[biggest, :4]
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def batch_cc_boxes(bin_masks, image_size):
    """[B,1,H,W] binary -> [B,4] largest-CC boxes."""
    boxes = []
    for b in range(bin_masks.shape[0]):
        boxes.append(_largest_cc_box(bin_masks[b, 0].cpu().numpy(),
                                     image_size))
    return torch.as_tensor(np.stack(boxes), dtype=torch.float32,
                           device=bin_masks.device)


def random_gt_points(gt, k, rng=None):
    """K positive + K negative GT points (guided prompts, Sec. 4.4)."""
    if rng is None:
        rng = np.random
    B, _, H, W = gt.shape
    all_coords, all_labels = [], []
    for b in range(B):
        m = (gt[b, 0].cpu().numpy() > 0.5)
        pos = np.argwhere(m)
        neg = np.argwhere(~m)
        if len(pos) == 0:
            pos = np.argwhere(~m)
        if len(neg) == 0:
            neg = np.argwhere(m)
        pi = pos[rng.randint(len(pos), size=k)][:, ::-1]  # (x, y)
        ni = neg[rng.randint(len(neg), size=k)][:, ::-1]
        all_coords.append(np.vstack([pi, ni]).astype(np.float32))
        all_labels.append(np.concatenate([np.ones(k), np.zeros(k)]))
    coords = torch.as_tensor(np.stack(all_coords), dtype=torch.float32,
                             device=gt.device)
    labels = torch.as_tensor(np.stack(all_labels), dtype=torch.int64,
                             device=gt.device)
    return coords, labels
