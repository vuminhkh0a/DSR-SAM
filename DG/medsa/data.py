"""
MedSA data loading: single-source leave-one-out benchmark.

Images/masks are read with utils.data_io (same as every other DG method);
USOVA blank-GT slices are already filtered by utils.data.get_datasets.
No data augmentation is used (benchmark convention; the paper presents no
augmentation component).

Prompts follow the repo (utils.py generate_click_prompt, 2D path): one
click point per slice - random foreground point for training, center
foreground point for val/test. Labels are 1 (foreground); if the mask has
no foreground (should not happen after the blank-GT filter) the image
center with label 0 is used.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.data import _make_loader, get_datasets
from utils.data_io import read_image_mask


def _random_point(mask_np):
    """Repo generate_click_prompt (2D): random foreground point (x, y)."""
    ys, xs = np.argwhere(mask_np > 0.5).T if (mask_np > 0.5).any() else (None, None)
    if ys is None:
        return np.array([[mask_np.shape[1] // 2, mask_np.shape[0] // 2]]), np.array([0])
    idx = np.random.randint(len(ys))
    return np.array([[xs[idx], ys[idx]]]), np.array([1])


def _center_point(mask_np):
    """Deterministic eval click: middle foreground point (x, y)."""
    fg = np.argwhere(mask_np > 0.5)
    if len(fg) == 0:
        return np.array([[mask_np.shape[1] // 2, mask_np.shape[0] // 2]]), np.array([0])
    y, x = fg[len(fg) // 2]
    return np.array([[x, y]]), np.array([1])


class MedSADataset(Dataset):
    """Single-source dataset with click-point prompts.

    Returns (image [3,H,W], mask [1,H,W], pt [1,2], plabel [1]) per slice.
    """

    def __init__(self, images, masks, image_size=256, train=True):
        self.images = images
        self.masks = masks
        self.image_size = image_size
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image, mask = read_image_mask(self.images[i], self.masks[i],
                                      self.image_size)
        mask_np = mask[0].numpy()
        if self.train:
            pt, plabel = _random_point(mask_np)
        else:
            pt, plabel = _center_point(mask_np)
        return (image, mask,
                torch.as_tensor(pt, dtype=torch.float32),
                torch.as_tensor(plabel, dtype=torch.int64))


def _collect_split(name, image_size, split):
    train_ds, valid_ds, test_ds = get_datasets(name=name,
                                               image_size=image_size,
                                               transform=None)
    ds = {'train': train_ds, 'val': valid_ds, 'test': test_ds}[split]
    return ds.images, ds.masks


def get_medsa_loaders(source_name, image_size, batch_size, num_workers,
                      pin_memory):
    """Train/val loaders for one source domain."""
    train_imgs, train_masks = _collect_split(source_name, image_size, 'train')
    print(f'Dataset: {source_name} | Source train: {len(train_imgs)}')
    val_imgs, val_masks = _collect_split(source_name, image_size, 'val')
    print(f'Dataset: {source_name} | Source val: {len(val_imgs)}')

    train_dataset = MedSADataset(train_imgs, train_masks,
                                 image_size=image_size, train=True)
    val_dataset = MedSADataset(val_imgs, val_masks, image_size=image_size,
                               train=False)
    train_loader = _make_loader(train_dataset, batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=pin_memory,
                                drop_last=True)
    val_loader = _make_loader(val_dataset, batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader


def get_medsa_target_loader(target_name, image_size, batch_size, num_workers,
                            pin_memory, split='test'):
    """Target loader with fixed center click prompts (split='test')."""
    imgs, masks = _collect_split(target_name, image_size, split)
    dataset = MedSADataset(imgs, masks, image_size=image_size, train=False)
    return _make_loader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)
