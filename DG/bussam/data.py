"""
BUSSAM data loading: single-source leave-one-out benchmark.

Images/masks are read with utils.data_io (same as every other DG method);
USOVA blank-GT slices are already filtered by utils.data.get_datasets.

Training uses the repo's JointTransform2D augmentation (utils/data_us.py,
train.py: p_rota=0.5, p_scale=0.5, p_contr=0.5, p_gama=0.5; p_flip=0,
p_gaussn=0, p_distor=0). Validation/testing are unaugmented. The benchmark's
shared [0,1] normalization is kept.

Prompts follow the repo (same click-prompt scheme as SAMUS): a single
foreground click point per slice - random foreground point for training,
center foreground point for val/test, computed after augmentation. Labels
are 1 (foreground); if the mask has no foreground (should not happen after
the blank-GT filter) the image center with label 0 is used.
"""
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset

from utils.data import _make_loader, get_datasets
from utils.data_io import read_image_mask


def _bussam_augment(image, mask):
    """Repo JointTransform2D (train params) on [3,H,W]/[1,H,W] tensors."""
    H, W = mask.shape[-2], mask.shape[-1]
    if np.random.rand() < 0.5:                       # gamma enhancement
        g = np.random.randint(10, 25) / 10.0
        image = image ** (1.0 / g)
    image_p = TF.to_pil_image(image)
    mask_p = TF.to_pil_image(mask)
    if np.random.rand() < 0.5:                       # random rotation
        angle = transforms.RandomRotation.get_params((-30, 30))
        image_p = TF.rotate(image_p, angle)
        mask_p = TF.rotate(mask_p, angle)
    if np.random.rand() < 0.5:                       # random scale + crop back
        scale = np.random.uniform(1, 1.3)
        new_h, new_w = int(H * scale), int(W * scale)
        image_p = TF.resize(image_p, (new_h, new_w),
                            interpolation=TF.InterpolationMode.BILINEAR)
        mask_p = TF.resize(mask_p, (new_h, new_w),
                           interpolation=TF.InterpolationMode.NEAREST)
        i, j, h, w = transforms.RandomCrop.get_params(image_p, (H, W))
        image_p = TF.crop(image_p, i, j, h, w)
        mask_p = TF.crop(mask_p, i, j, h, w)
    if np.random.rand() < 0.5:                       # random contrast
        image_p = transforms.ColorJitter(contrast=(0.8, 2.0))(image_p)
    image_p = TF.resize(image_p, (H, W),
                        interpolation=TF.InterpolationMode.BILINEAR)
    mask_p = TF.resize(mask_p, (H, W),
                       interpolation=TF.InterpolationMode.NEAREST)
    return TF.to_tensor(image_p), TF.to_tensor(mask_p)


def _random_point(mask_np):
    """Random foreground point (x, y), label 1."""
    ys, xs = np.argwhere(mask_np > 0.5).T if (mask_np > 0.5).any() else (None, None)
    if ys is None:
        return np.array([[mask_np.shape[1] // 2, mask_np.shape[0] // 2]]), np.array([0])
    idx = np.random.randint(len(ys))
    return np.array([[xs[idx], ys[idx]]]), np.array([1])


def _center_point(mask_np):
    """Middle foreground point (x, y), label 1."""
    fg = np.argwhere(mask_np > 0.5)
    if len(fg) == 0:
        return np.array([[mask_np.shape[1] // 2, mask_np.shape[0] // 2]]), np.array([0])
    y, x = fg[len(fg) // 2]
    return np.array([[x, y]]), np.array([1])


class BussamDataset(Dataset):
    """Single-source dataset with click-point prompts.

    Returns (image [3,H,W], mask [1,H,W], pt [1,2], plabel [1]) per slice.
    """

    def __init__(self, images, masks, image_size=256, train=True, augment=False):
        self.images = images
        self.masks = masks
        self.image_size = image_size
        self.train = train
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image, mask = read_image_mask(self.images[i], self.masks[i],
                                      self.image_size)
        if self.augment:
            image, mask = _bussam_augment(image, mask)
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


def get_bussam_loaders(source_name, image_size, batch_size, num_workers,
                       pin_memory):
    """Train/val loaders for one source domain."""
    train_imgs, train_masks = _collect_split(source_name, image_size, 'train')
    print(f'Dataset: {source_name} | Source train: {len(train_imgs)}')
    val_imgs, val_masks = _collect_split(source_name, image_size, 'val')
    print(f'Dataset: {source_name} | Source val: {len(val_imgs)}')

    train_dataset = BussamDataset(train_imgs, train_masks,
                                  image_size=image_size, train=True,
                                  augment=True)
    val_dataset = BussamDataset(val_imgs, val_masks, image_size=image_size,
                                train=False, augment=False)
    train_loader = _make_loader(train_dataset, batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=pin_memory,
                                drop_last=True)
    val_loader = _make_loader(val_dataset, batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader


def get_bussam_target_loader(target_name, image_size, batch_size, num_workers,
                             pin_memory, split='test'):
    """Target loader with fixed center click prompts (split='test')."""
    imgs, masks = _collect_split(target_name, image_size, split)
    dataset = BussamDataset(imgs, masks, image_size=image_size, train=False)
    return _make_loader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)
