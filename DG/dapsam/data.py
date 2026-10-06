"""
DAPSAM data loading: single-source leave-one-out benchmark.

Images/masks are read with utils.data_io (same as every other DG method);
USOVA blank-GT slices are already filtered by utils.data.get_datasets.

Training uses a port of the repo's prostate augmentations
(prostate/datasets/prostate/transform.py, batchgenerators
get_train_transform): MirrorTransform(axes=(0,1)), SpatialTransform
rotation +-15 deg (p=0.2) / scale 0.85-1.25 (p=0.2), Gaussian noise
(p=0.1), Gaussian blur (p=0.2), multiplicative / additive brightness
(p=0.15 each), contrast (p=0.15) and gamma (p=0.15). Elastic deformation
and low-resolution simulation are omitted (batchgenerators not available).
Validation/testing are unaugmented; the benchmark's shared [0,1]
normalization is kept.

DAPSAM needs no prompts: the prototype prompt generator creates the dense
prompt automatically, so plain (image, mask) pairs are returned.
"""
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from utils.data import _make_loader, get_datasets
from utils.data_io import read_image_mask


def _affine(image, mask, angle, scale):
    image = TF.affine(image, angle, (0.0, 0.0), scale, 0.0,
                      interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)
    mask = TF.affine(mask, angle, (0.0, 0.0), scale, 0.0,
                     interpolation=TF.InterpolationMode.NEAREST, fill=0.0)
    return image, mask


def _dapsam_augment(image, mask):
    """Port of the repo prostate training pipeline (see module docstring)."""
    # MirrorTransform(axes=(0, 1)): independent per-axis reflection.
    if np.random.rand() < 0.5:
        image, mask = TF.hflip(image), TF.hflip(mask)
    if np.random.rand() < 0.5:
        image, mask = TF.vflip(image), TF.vflip(mask)
    # SpatialTransform: rotation / scale.
    if np.random.rand() < 0.2:
        image, mask = _affine(image, mask, np.random.uniform(-15, 15), 1.0)
    if np.random.rand() < 0.2:
        image, mask = _affine(image, mask, 0.0, np.random.uniform(0.85, 1.25))
    # GaussianNoise.
    if np.random.rand() < 0.1:
        image = (image + torch.randn_like(image)
                 * np.random.uniform(0.0, 0.1)).clamp(0.0, 1.0)
    # GaussianBlur.
    if np.random.rand() < 0.2:
        sigma = float(np.random.uniform(0.5, 1.0))
        image = TF.gaussian_blur(image, kernel_size=[15, 15],
                                 sigma=[sigma, sigma])
    # BrightnessMultiplicative.
    if np.random.rand() < 0.15:
        image = (image * np.random.uniform(0.75, 1.25)).clamp(0.0, 1.0)
    # Brightness (additive, per channel).
    if np.random.rand() < 0.15:
        image = (image + torch.randn_like(image) * 0.1).clamp(0.0, 1.0)
    # ContrastAugmentation.
    if np.random.rand() < 0.15:
        image = TF.adjust_contrast(
            image, np.random.uniform(0.75, 1.25)).clamp(0.0, 1.0)
    # GammaTransform.
    if np.random.rand() < 0.15:
        image = image.clamp(1e-6, 1.0) ** float(np.random.uniform(0.7, 1.5))
    return image, mask


class DapsamDataset(Dataset):
    """Single-source dataset, returns (image [3,H,W], mask [1,H,W])."""

    def __init__(self, images, masks, image_size=256, augment=False):
        self.images = images
        self.masks = masks
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image, mask = read_image_mask(self.images[i], self.masks[i],
                                      self.image_size)
        if self.augment:
            image, mask = _dapsam_augment(image, mask)
        return image, mask


def _collect_split(name, image_size, split):
    train_ds, valid_ds, test_ds = get_datasets(name=name,
                                               image_size=image_size,
                                               transform=None)
    ds = {'train': train_ds, 'val': valid_ds, 'test': test_ds}[split]
    return ds.images, ds.masks


def get_dapsam_loaders(source_name, image_size, batch_size, num_workers,
                       pin_memory):
    """Train/val loaders for one source domain."""
    train_imgs, train_masks = _collect_split(source_name, image_size, 'train')
    print(f'Dataset: {source_name} | Source train: {len(train_imgs)}')
    val_imgs, val_masks = _collect_split(source_name, image_size, 'val')
    print(f'Dataset: {source_name} | Source val: {len(val_imgs)}')

    train_loader = _make_loader(
        DapsamDataset(train_imgs, train_masks, image_size=image_size,
                      augment=True),
        batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, drop_last=True)
    val_loader = _make_loader(
        DapsamDataset(val_imgs, val_masks, image_size=image_size,
                      augment=False),
        batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory)
    return train_loader, val_loader


def get_dapsam_target_loader(target_name, image_size, batch_size, num_workers,
                             pin_memory, split='test'):
    """Target loader (split='test' by default)."""
    imgs, masks = _collect_split(target_name, image_size, split)
    dataset = DapsamDataset(imgs, masks, image_size=image_size)
    return _make_loader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)
