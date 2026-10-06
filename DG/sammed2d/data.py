"""
SAM-Med2D data loading: test-only box-prompt evaluation.

Images/masks are read with utils.data_io (same as every other DG method);
USOVA blank-GT slices are already filtered by utils.data.get_datasets.
Boxes come from the standard y_DG/box_coords.json (256-space), as in the
official SAM-Med2D box-prompt inference (upstream test.py,
boxes_prompt=True). Native resolution is 256, so no box rescaling.
"""
import json
import os

import torch
from torch.utils.data import Dataset

from utils.data import _make_loader, get_datasets
from utils.data_io import read_image_mask

BOX_COORDS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'box_coords.json')


def _load_box_map():
    """Map image path -> first bounding box [x1,y1,x2,y2] (256-space)."""
    with open(BOX_COORDS_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    box_map = {}
    for entries in data.values():
        for e in entries:
            box = e['boxes'][0] if e['boxes'] else None
            box_map[e['image']] = box
    return box_map


BOX_MAP = _load_box_map()


class SAMMed2DBoxDataset(Dataset):
    """Returns (image [3,H,W], mask [1,H,W], bbox [4]) per slice; missing
    boxes fall back to the full image."""

    def __init__(self, images, masks, image_size=256):
        self.images = images
        self.masks = masks
        self.image_size = image_size
        self.boxes = []
        for img in images:
            box = BOX_MAP.get(img)
            if box is None:
                box = [0, 0, image_size, image_size]
            self.boxes.append(list(box))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image, mask = read_image_mask(self.images[i], self.masks[i],
                                      self.image_size)
        bbox = torch.as_tensor(self.boxes[i], dtype=torch.float32)
        return image, mask, bbox


def _collect_split(name, image_size, split):
    train_ds, valid_ds, test_ds = get_datasets(name=name,
                                               image_size=image_size,
                                               transform=None)
    ds = {'train': train_ds, 'val': valid_ds, 'test': test_ds}[split]
    return ds.images, ds.masks


def get_sammed2d_target_loader(target_name, image_size, batch_size,
                               num_workers, pin_memory, split='test'):
    """Target loader with standard box prompts (split='test')."""
    imgs, masks = _collect_split(target_name, image_size, split)
    dataset = SAMMed2DBoxDataset(imgs, masks, image_size=image_size)
    return _make_loader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)
