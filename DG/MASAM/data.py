"""
MA-SAM data loading: multi-source leave-one-out benchmark.

The MA-SAM paper trains jointly on multiple imaging modalities, so the
leave-one-out protocol uses TWO source datasets for training and the
remaining one for testing. Images/masks are read with utils.data_io
(same as every other DG method), and bounding-box prompts come from the
standard y_DG/box_coords.json.

The random augmentation is a faithful 2D port of the repo's
datasets/dataset_bbox.py RandomGenerator (rot90/flip, random rotation,
gamma light adjustment, plus two random ops chosen from shear, scale,
translate, posterize, contrast, brightness, sharpness). Bounding boxes
are transformed together with the rot/flip/rotate operations exactly like
the repo (transform_bounding_box / rotate_bounding_box).
"""
import json
import os
import random

import cv2
import numpy as np
import PIL.Image
import PIL.ImageEnhance
import PIL.ImageOps
import torch
from scipy import ndimage
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


# ============================================================
# Bounding-box transforms (repo datasets/dataset_bbox.py)
# ============================================================

def transform_bounding_box(bbox, shape, k, axis):
    """Transforms the bounding box for rot90(k) + flip(axis).
    bbox: tuple (x_min, y_min, x_max, y_max); shape: (height, width)."""
    x_min, y_min, x_max, y_max = bbox
    height, width = shape
    for _ in range(k):
        x_min, y_min, x_max, y_max = y_min, width - x_max, y_max, width - x_min
        height, width = width, height
    if axis == 0:
        y_min, y_max = height - y_max, height - y_min
    elif axis == 1:
        x_min, x_max = width - x_max, width - x_min
    return x_min, y_min, x_max, y_max


def rotate_point(x, y, angle, cx, cy):
    angle = np.radians(angle)
    x_new = cx + (x - cx) * np.cos(angle) - (y - cy) * np.sin(angle)
    y_new = cy + (x - cx) * np.sin(angle) + (y - cy) * np.cos(angle)
    return x_new, y_new


def rotate_bounding_box(bbox, angle, shape):
    """Rotates the bounding box by angle degrees around the image center."""
    x_min, y_min, x_max, y_max = bbox
    height, width = shape
    cx, cy = width // 2, height // 2
    corners = [(x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)]
    rotated_corners = [rotate_point(x, y, angle, cx, cy) for x, y in corners]
    x_coords, y_coords = zip(*rotated_corners)
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    return x_min, y_min, x_max, y_max


# ============================================================
# Augmentation ops (2D port of the repo's RandomGenerator)
# ============================================================

def convert_to_PIL(img):
    img = np.clip(img, 0, 1)
    return PIL.Image.fromarray((img * 255).astype(np.uint8))


def convert_to_np(img):
    return np.array(img).astype(np.float32) / 255


def convert_to_PIL_label(label):
    return PIL.Image.fromarray(label.astype(np.uint8))


def convert_to_np_label(label):
    return np.array(label).astype(np.float32)


def posterize(img, label, v):
    v = int(v)
    img = convert_to_PIL(img)
    img = PIL.ImageOps.posterize(img, bits=v)
    img = convert_to_np(img)
    return img, label


def contrast(img, label, v):
    img = convert_to_PIL(img)
    img = PIL.ImageEnhance.Contrast(img).enhance(v)
    img = convert_to_np(img)
    return img, label


def brightness(img, label, v):
    img = convert_to_PIL(img)
    img = PIL.ImageEnhance.Brightness(img).enhance(v)
    img = convert_to_np(img)
    return img, label


def sharpness(img, label, v):
    img = convert_to_PIL(img)
    img = PIL.ImageEnhance.Sharpness(img).enhance(v)
    img = convert_to_np(img)
    return img, label


def identity(img, label, v):
    return img, label


def adjust_light(image, label):
    image = image * 255.0
    gamma = random.random() * 3 + 0.5
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    image = cv2.LUT(np.array(image).astype(np.uint8), table).astype(np.uint8)
    image = image / 255.0
    return image, label


def _affine(img, mat, resample):
    img = convert_to_PIL(img)
    img = img.transform(img.size, PIL.Image.AFFINE, mat, resample=resample)
    return convert_to_np(img)


def _affine_label(label, mat):
    label = convert_to_PIL_label(label)
    label = label.transform(label.size, PIL.Image.AFFINE, mat, resample=PIL.Image.NEAREST)
    return convert_to_np_label(label)


def shear_x(img, label, v):
    shear_mat = [1, v, -v * img.shape[1] / 2, 0, 1, 0]
    img = _affine(img, shear_mat, PIL.Image.BILINEAR)
    label = _affine_label(label, shear_mat)
    return img, label


def shear_y(img, label, v):
    shear_mat = [1, 0, 0, v, 1, -v * img.shape[0] / 2]
    img = _affine(img, shear_mat, PIL.Image.BILINEAR)
    label = _affine_label(label, shear_mat)
    return img, label


def translate_x(img, label, v):
    translate_mat = [1, 0, v * img.shape[1], 0, 1, 0]
    img = _affine(img, translate_mat, PIL.Image.BILINEAR)
    label = _affine_label(label, translate_mat)
    return img, label


def translate_y(img, label, v):
    translate_mat = [1, 0, 0, 0, 1, v * img.shape[0]]
    img = _affine(img, translate_mat, PIL.Image.BILINEAR)
    label = _affine_label(label, translate_mat)
    return img, label


def scale(img, label, v):
    img = _affine(img, [v, 0, 0, 0, v, 0], PIL.Image.BILINEAR)
    label = _affine_label(label, [v, 0, 0, 0, v, 0])
    return img, label


def random_rot_flip(image, label, bbox):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(0, 1))
    label = np.rot90(label, k, axes=(0, 1))
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    new_bbox = transform_bounding_box(bbox, image.shape[:2], k, axis)
    return image, label, new_bbox


def random_rotate(image, label, bbox):
    angle = np.random.randint(-15, 15)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    new_bbox = rotate_bounding_box(bbox, angle, image.shape[:2])
    return image, label, new_bbox


class RandomGenerator(object):
    """2D port of the repo's RandomGenerator (datasets/dataset_bbox.py)."""

    def __init__(self):
        self.rng = np.random.default_rng(42)
        self.n = 2
        self.scale = (0.8, 1.2, 0)
        self.translate = (-0.2, 0.2, 0)
        self.shear = (-0.3, 0.3, 0)
        self.posterize = (4, 8.99, 2)
        self.contrast = (0.7, 1.3, 2)
        self.brightness = (0.7, 1.3, 2)
        self.sharpness = (0.1, 1.9, 2)
        self.create_ops()

    def create_ops(self):
        ops = [
            (shear_x, self.shear),
            (shear_y, self.shear),
            (scale, self.scale),
            (translate_x, self.translate),
            (translate_y, self.translate),
            (posterize, self.posterize),
            (contrast, self.contrast),
            (brightness, self.brightness),
            (sharpness, self.sharpness),
            (identity, (0, 1, 1)),
        ]
        self.ops = [op for op in ops if op[1][2] != 0]

    def __call__(self, image, label, bbox):
        if random.random() > 0.5:
            image, label, bbox = random_rot_flip(image, label, bbox)
        if random.random() > 0.5:
            image, label, bbox = random_rotate(image, label, bbox)
        if random.random() > 0.5:
            image, label = adjust_light(image, label)

        inds = self.rng.choice(len(self.ops), size=self.n, replace=False)
        for i in inds:
            op = self.ops[i]
            aug_func = op[0]
            aug_params = op[1]
            v = self.rng.uniform(aug_params[0], aug_params[1])
            image, label = aug_func(image, label, v)

        return image.astype(np.float32), label.astype(np.float32), np.array(bbox).astype(np.float32)


class MASAMDataset(Dataset):
    """Multi-source dataset with MA-SAM augmentation and box prompts.

    Returns (image [3,H,W], mask [1,H,W], bbox [1,4]) per slice.
    """

    def __init__(self, images, masks, boxes, image_size=256, augment=True):
        self.images = images
        self.masks = masks
        self.boxes = boxes
        self.image_size = image_size
        self.augment = augment
        self.transform = RandomGenerator() if augment else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image, mask = read_image_mask(self.images[i], self.masks[i], self.image_size)
        image = image.permute(1, 2, 0).numpy()  # [H, W, 3] float [0,1]
        mask = mask[0].numpy()                  # [H, W] float {0,1}
        bbox = list(self.boxes[i])

        if self.transform is not None:
            image, mask, bbox = self.transform(image, mask, bbox)

        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)  # [3,H,W]
        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)        # [1,H,W]
        bbox = torch.as_tensor(bbox, dtype=torch.float32).unsqueeze(0)       # [1,4]
        return image, mask, bbox


def _collect_source(name, image_size):
    """Returns (images, masks, boxes) for the train split of one source."""
    train_ds, _, _ = get_datasets(name=name, image_size=image_size, transform=None)
    boxes = _get_boxes(name, train_ds.images)
    return train_ds.images, train_ds.masks, boxes


def _collect_val(name, image_size):
    _, valid_ds, _ = get_datasets(name=name, image_size=image_size, transform=None)
    boxes = _get_boxes(name, valid_ds.images)
    return valid_ds.images, valid_ds.masks, boxes


def get_masam_loaders(source_names, image_size, batch_size, num_workers,
                      pin_memory, augment=True):
    """Combine two source domains for training; validation uses both sources' val splits."""
    train_imgs, train_masks, train_boxes = [], [], []
    val_imgs, val_masks, val_boxes = [], [], []
    for name in source_names:
        imgs, masks, boxes = _collect_source(name, image_size)
        print(f'Dataset: {name} | Source train: {len(imgs)}')
        train_imgs += imgs
        train_masks += masks
        train_boxes += boxes
        imgs, masks, boxes = _collect_val(name, image_size)
        print(f'Dataset: {name} | Source val: {len(imgs)}')
        val_imgs += imgs
        val_masks += masks
        val_boxes += boxes

    train_dataset = MASAMDataset(train_imgs, train_masks, train_boxes,
                                 image_size=image_size, augment=augment)
    val_dataset = MASAMDataset(val_imgs, val_masks, val_boxes,
                               image_size=image_size, augment=False)

    train_loader = _make_loader(train_dataset, batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=pin_memory,
                                drop_last=True)
    val_loader = _make_loader(val_dataset, batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader


def get_masam_target_loader(target_name, image_size, batch_size, num_workers,
                            pin_memory, split='test'):
    """Target loader with box prompts (split='test' by default)."""
    _, _, test_ds = get_datasets(name=target_name, image_size=image_size, transform=None)
    boxes = _get_boxes(target_name, test_ds.images)
    dataset = MASAMDataset(test_ds.images, test_ds.masks, boxes,
                           image_size=image_size, augment=False)
    loader = _make_loader(dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory)
    return loader
