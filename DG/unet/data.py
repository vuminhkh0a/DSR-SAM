"""
UNet data loading: single-source leave-one-out benchmark.

Images/masks are read with utils.data_io (same as every other DG method);
USOVA blank-GT slices are already filtered by utils.data.get_datasets.
No data augmentation is used (benchmark convention).
"""
from torch.utils.data import Dataset

from utils.data import _make_loader, get_datasets
from utils.data_io import read_image_mask


class UNetDataset(Dataset):
    """Single-source dataset, returns (image [3,H,W], mask [1,H,W])."""

    def __init__(self, images, masks, image_size=256):
        self.images = images
        self.masks = masks
        self.image_size = image_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return read_image_mask(self.images[i], self.masks[i], self.image_size)


def _collect_split(name, image_size, split):
    train_ds, valid_ds, test_ds = get_datasets(name=name,
                                               image_size=image_size,
                                               transform=None)
    ds = {'train': train_ds, 'val': valid_ds, 'test': test_ds}[split]
    return ds.images, ds.masks


def get_unet_loaders(source_name, image_size, batch_size, num_workers,
                     pin_memory):
    """Train/val loaders for one source domain."""
    train_imgs, train_masks = _collect_split(source_name, image_size, 'train')
    print(f'Dataset: {source_name} | Source train: {len(train_imgs)}')
    val_imgs, val_masks = _collect_split(source_name, image_size, 'val')
    print(f'Dataset: {source_name} | Source val: {len(val_imgs)}')

    train_loader = _make_loader(
        UNetDataset(train_imgs, train_masks, image_size=image_size),
        batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, drop_last=True)
    val_loader = _make_loader(
        UNetDataset(val_imgs, val_masks, image_size=image_size),
        batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory)
    return train_loader, val_loader


def get_unet_target_loader(target_name, image_size, batch_size, num_workers,
                           pin_memory, split='test'):
    """Target loader (split='test' by default)."""
    imgs, masks = _collect_split(target_name, image_size, split)
    dataset = UNetDataset(imgs, masks, image_size=image_size)
    return _make_loader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)
