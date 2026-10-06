"""
Nora data loading: single-source leave-one-out benchmark.

Images/masks are read with utils.data_io (same as every other DG method);
USOVA blank-GT slices are already filtered by utils.data.get_datasets.
Training uses the repo's joint augmentation (joint_transforms.RandomAffine
with translate=(0.125, 0.125) and RandomHorizontallyFlip); validation and
testing are unaugmented.

Nora needs no prompts: the noise-robust prompt generator creates sparse
and dense prompts automatically, so plain (image, mask) pairs are returned.
"""
import random

import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from utils.data import _make_loader, get_datasets
from utils.data_io import read_image_mask


def _random_affine(image, mask):
    """Repo joint_transforms.RandomAffine(degrees=0, translate=(0.125,0.125)), p=0.5."""
    if random.random() < 0.5:
        _, h, w = image.shape
        max_dx, max_dy = 0.125 * w, 0.125 * h
        tx = float(round(random.uniform(-max_dx, max_dx)))
        ty = float(round(random.uniform(-max_dy, max_dy)))
        image = TF.affine(image, angle=0.0, translate=[tx, ty], scale=1.0,
                          shear=0.0,
                          interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.affine(mask, angle=0.0, translate=[tx, ty], scale=1.0,
                         shear=0.0,
                         interpolation=TF.InterpolationMode.NEAREST)
    return image, mask


def _random_hflip(image, mask):
    """Repo joint_transforms.RandomHorizontallyFlip, p=0.5."""
    if random.random() < 0.5:
        return TF.hflip(image), TF.hflip(mask)
    return image, mask


class NoraDataset(Dataset):
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
            image, mask = _random_affine(image, mask)
            image, mask = _random_hflip(image, mask)
        return image, mask


def _collect_split(name, image_size, split):
    train_ds, valid_ds, test_ds = get_datasets(name=name,
                                               image_size=image_size,
                                               transform=None)
    ds = {'train': train_ds, 'val': valid_ds, 'test': test_ds}[split]
    return ds.images, ds.masks


def get_nora_loaders(source_name, image_size, batch_size, num_workers,
                     pin_memory):
    """Train/val loaders for one source domain."""
    train_imgs, train_masks = _collect_split(source_name, image_size, 'train')
    print(f'Dataset: {source_name} | Source train: {len(train_imgs)}')
    val_imgs, val_masks = _collect_split(source_name, image_size, 'val')
    print(f'Dataset: {source_name} | Source val: {len(val_imgs)}')

    train_loader = _make_loader(
        NoraDataset(train_imgs, train_masks, image_size=image_size,
                    augment=True),
        batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, drop_last=True)
    val_loader = _make_loader(
        NoraDataset(val_imgs, val_masks, image_size=image_size,
                    augment=False),
        batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory)
    return train_loader, val_loader


def get_nora_target_loader(target_name, image_size, batch_size, num_workers,
                           pin_memory, split='test'):
    """Target loader (split='test' by default)."""
    imgs, masks = _collect_split(target_name, image_size, split)
    dataset = NoraDataset(imgs, masks, image_size=image_size)
    return _make_loader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)
