"""
Trans-SAM data loading: single-source leave-one-out benchmark.

Images/masks are read with utils.data_io (same as every other DG method);
USOVA blank-GT slices are already filtered by utils.data.get_datasets.

Training uses the repo's Dataset_train.img_transform augmentation
(model_utils/dataset_split.py): GaussianBlur(25, sigma 0.001-2),
ColorJitter(brightness 0.4, contrast 0.5, saturation 0.25, hue 0.01),
random horizontal/vertical flip (p=0.5) and a random affine
(angle +-180, translate +-32, scale 0.5-1.5, shear +-22.5). Validation and
testing are unaugmented. The benchmark's shared [0,1] normalization is kept
(the repo normalizes differently for train vs val/test).

Trans-SAM is prompt-free, so plain (image, mask) pairs are returned.
"""
import random

import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset

from utils.data import _make_loader, get_datasets
from utils.data_io import read_image_mask

_IMG_AUG = transforms.Compose([
    transforms.GaussianBlur(kernel_size=(25, 25), sigma=(0.001, 2.0)),
    transforms.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.25,
                           hue=0.01),
])


def _transsam_augment(image, mask):
    """Repo Dataset_train.img_transform (normalization handled by benchmark)."""
    image = _IMG_AUG(image)                       # [3,H,W] in [0,1]
    if random.random() < 0.5:
        image, mask = TF.hflip(image), TF.hflip(mask)
    if random.random() < 0.5:
        image, mask = TF.vflip(image), TF.vflip(mask)
    angle = random.uniform(-180.0, 180.0)
    h_trans = random.uniform(-32.0, 32.0)
    v_trans = random.uniform(-32.0, 32.0)
    scale = random.uniform(0.5, 1.5)
    shear = random.uniform(-22.5, 22.5)
    image = TF.affine(image, angle, (h_trans, v_trans), scale, shear,
                      interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)
    mask = TF.affine(mask, angle, (h_trans, v_trans), scale, shear,
                     interpolation=TF.InterpolationMode.NEAREST, fill=0.0)
    return image, mask


class TranssamDataset(Dataset):
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
            image, mask = _transsam_augment(image, mask)
        return image, mask


def _collect_split(name, image_size, split):
    train_ds, valid_ds, test_ds = get_datasets(name=name,
                                               image_size=image_size,
                                               transform=None)
    ds = {'train': train_ds, 'val': valid_ds, 'test': test_ds}[split]
    return ds.images, ds.masks


def get_transsam_loaders(source_name, image_size, batch_size, num_workers,
                         pin_memory):
    """Train/val loaders for one source domain."""
    train_imgs, train_masks = _collect_split(source_name, image_size, 'train')
    print(f'Dataset: {source_name} | Source train: {len(train_imgs)}')
    val_imgs, val_masks = _collect_split(source_name, image_size, 'val')
    print(f'Dataset: {source_name} | Source val: {len(val_imgs)}')

    train_loader = _make_loader(
        TranssamDataset(train_imgs, train_masks, image_size=image_size,
                        augment=True),
        batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, drop_last=True)
    val_loader = _make_loader(
        TranssamDataset(val_imgs, val_masks, image_size=image_size,
                        augment=False),
        batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory)
    return train_loader, val_loader


def get_transsam_target_loader(target_name, image_size, batch_size,
                               num_workers, pin_memory, split='test'):
    """Target loader (split='test' by default)."""
    imgs, masks = _collect_split(target_name, image_size, split)
    dataset = TranssamDataset(imgs, masks, image_size=image_size)
    return _make_loader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)
