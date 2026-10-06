"""
SR-SAM data loading: single-source leave-one-out benchmark.

The SR-SAM paper trains on a SINGLE source domain and evaluates on the
other (OOD) domains. Images/masks are read with utils.data_io (same as
every other DG method), and bounding-box prompts come from the standard
y_DG/box_coords.json. No data augmentation is used (the paper and the
official repo do not specify any).
"""
import json
import os

import torch
from torch.utils.data import Dataset

from utils.data import _make_loader, get_datasets
from utils.data_io import read_image_mask

BOX_COORDS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                               'box_coords.json')


def _load_box_map():
    """Map image path -> first bounding box [x_min, y_min, x_max, y_max] (256-space)."""
    with open(BOX_COORDS_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    box_map = {}
    for entries in data.values():
        for e in entries:
            box = e['boxes'][0] if e['boxes'] else None
            box_map[e['image']] = box
    return box_map


BOX_MAP = _load_box_map()


def _get_boxes(name, images):
    boxes = []
    for img in images:
        box = BOX_MAP.get(img)
        if box is None:
            box = [0, 0, 256, 256]
        boxes.append(list(box))
    return boxes


class SRSAMDataset(Dataset):
    """Single-source dataset with box prompts.

    Returns (image [3,H,W], mask [1,H,W], bbox [1,4]) per slice.
    """

    def __init__(self, images, masks, boxes, image_size=256):
        self.images = images
        self.masks = masks
        self.boxes = boxes
        self.image_size = image_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image, mask = read_image_mask(self.images[i], self.masks[i], self.image_size)
        bbox = torch.as_tensor(self.boxes[i], dtype=torch.float32).unsqueeze(0)  # [1,4]
        return image, mask, bbox


def _collect_split(name, image_size, split):
    train_ds, valid_ds, test_ds = get_datasets(name=name, image_size=image_size, transform=None)
    ds = {'train': train_ds, 'val': valid_ds, 'test': test_ds}[split]
    boxes = _get_boxes(name, ds.images)
    return ds.images, ds.masks, boxes


def get_sr_sam_loaders(source_name, image_size, batch_size, num_workers,
                       pin_memory):
    """Train/val loaders for one source domain."""
    train_imgs, train_masks, train_boxes = _collect_split(source_name, image_size, 'train')
    print(f'Dataset: {source_name} | Source train: {len(train_imgs)}')
    val_imgs, val_masks, val_boxes = _collect_split(source_name, image_size, 'val')
    print(f'Dataset: {source_name} | Source val: {len(val_imgs)}')

    train_dataset = SRSAMDataset(train_imgs, train_masks, train_boxes, image_size=image_size)
    val_dataset = SRSAMDataset(val_imgs, val_masks, val_boxes, image_size=image_size)

    train_loader = _make_loader(train_dataset, batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=pin_memory,
                                drop_last=True)
    val_loader = _make_loader(val_dataset, batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader


def get_sr_sam_target_loader(target_name, image_size, batch_size, num_workers,
                             pin_memory, split='test'):
    """Target loader with box prompts (split='test' by default)."""
    imgs, masks, boxes = _collect_split(target_name, image_size, split)
    dataset = SRSAMDataset(imgs, masks, boxes, image_size=image_size)
    loader = _make_loader(dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory)
    return loader
