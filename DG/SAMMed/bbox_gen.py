"""
SAMMed stage-2: mask-filtering module -> refined bounding boxes
(repo save_resnet_bbox.py + utils/utils.py filter_mask).

The coarse predictions of the stage-1 segmentation backbone are thresholded
at theta_1 = 0.75, downscaled to 128x128 (paper Sec. 3.3: speed vs accuracy
trade-off), and only the LARGEST CONNECTED COMPONENT is kept (BFS-based
filtering of the repo); the resulting box [x_min, y_min, x_max, y_max] is
scaled back by 4 to the 512x512 space and saved as .npy (one file per
image id, mirroring the repo's src_domain_idx_{domain}_75 folders).
"""
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.data_io import read_image_mask
from DG.SAMMed.data import preprocess_image

FILTER_SIZE = 128   # downscaled mask size for the BFS filtering
SCALE = 4           # 512 // 128


class _ImageMaskDataset(Dataset):
    def __init__(self, images, masks, image_size):
        self.images = images
        self.masks = masks
        self.image_size = image_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image, mask = read_image_mask(self.images[i], self.masks[i], self.image_size)
        image = preprocess_image(image)
        return image, mask


def filter_mask_largest_component(bin_mask):
    """Keep only the largest connected component of a binary mask
    (BFS largest-region filtering of repo utils/utils.py, implemented with
    cv2 connected components - identical result)."""
    if not bin_mask.any():
        return np.zeros(4, dtype=np.int64)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (bin_mask > 0).astype(np.uint8), 4)
    if n_labels <= 1:
        return np.zeros(4, dtype=np.int64)
    sizes = stats[1:, -1]
    largest = int(np.argmax(sizes)) + 1
    ys, xs = np.where(labels == largest)
    x_min, y_min = int(xs.min()), int(ys.min())
    x_max, y_max = int(xs.max()), int(ys.max())
    return np.array([x_min, y_min, x_max, y_max], dtype=np.int64)


@torch.no_grad()
def generate_bboxes(model, images, masks, ids, save_dir, image_size=512,
                    thresh=0.75, batch_size=8, num_workers=4, device='cpu'):
    """Run the segmentation backbone over a split and save the refined
    bounding boxes (one .npy per image id). Idempotent per file: only the
    ids whose box file is missing are processed (train/val of the same
    domain share one folder, like the repo's src_domain_idx_{domain}_75)."""
    os.makedirs(save_dir, exist_ok=True)
    todo = [(im, mk, id) for im, mk, id in zip(images, masks, ids)
            if not os.path.exists(os.path.join(save_dir, id + '.npy'))]
    if not todo:
        print(f'BBoxes already exist ({len(ids)}) -> {save_dir}')
        return save_dir
    model.eval()
    imgs = [t[0] for t in todo]
    mks = [t[1] for t in todo]
    ids_todo = [t[2] for t in todo]
    dataset = _ImageMaskDataset(imgs, mks, image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    generated = 0
    idx = 0
    for images_b, masks_b in loader:
        images_b = images_b.to(device, non_blocking=True)
        preds = torch.sigmoid(model(images_b)[1])
        preds = torch.nn.functional.interpolate(
            preds, (FILTER_SIZE, FILTER_SIZE), mode='bilinear', align_corners=False)
        preds = (preds > thresh).cpu().numpy()[:, 0]
        for k in range(preds.shape[0]):
            box = filter_mask_largest_component(preds[k]) * SCALE
            np.save(os.path.join(save_dir, ids_todo[idx + k] + '.npy'), box)
            generated += 1
        idx += preds.shape[0]
    print(f'BBoxes saved: {generated} -> {save_dir}')
    return save_dir


def ensure_bboxes(model, dataset_name, split, save_dir, image_size=512,
                  thresh=0.75, batch_size=8, num_workers=4, device='cpu'):
    """Generate refined boxes for a split if they do not exist yet."""
    from DG.SAMMed.data import get_resnet_dataset
    imgs, masks, ids = get_resnet_dataset(dataset_name, image_size, split)
    return generate_bboxes(model, imgs, masks, ids, save_dir, image_size=image_size,
                           thresh=thresh, batch_size=batch_size,
                           num_workers=num_workers, device=device)
