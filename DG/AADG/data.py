import numpy as np
from torch.utils.data import Dataset
from utils.data import get_datasets, _make_loader
from utils.data_io import load_image, load_mask, img_to_tensor, mask_to_tensor, read_image_mask
from utils.seed import set_seed
from DG.AADG.policy import Policy, default_policies

set_seed()


class _SourceCombinedDataset(Dataset):
    def __init__(self, source_name, image_size, limit=None):
        train_ds, valid_ds, _ = get_datasets(name=source_name, image_size=image_size, transform=None)
        self.images = train_ds.images + valid_ds.images
        self.masks = train_ds.masks + valid_ds.masks
        self.image_size = image_size
        self.aadg_policy = Policy(default_policies())
        self.aug_prob = 0.5
        if limit is not None:
            self.images = self.images[:limit]
            self.masks = self.masks[:limit]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = load_image(self.images[i], self.image_size)
        mask = load_mask(self.masks[i], self.image_size)

        if np.random.random() < self.aug_prob:
            img_aug, mask_aug = self.aadg_policy(img, mask)
        else:
            img_aug, mask_aug = img.copy(), mask.copy()

        return img_to_tensor(img_aug), mask_to_tensor(mask_aug)


class _SourceValDataset(Dataset):
    def __init__(self, source_name, image_size):
        _, valid_ds, _ = get_datasets(name=source_name, image_size=image_size, transform=None)
        self.images = valid_ds.images
        self.masks = valid_ds.masks
        self.image_size = image_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return read_image_mask(self.images[i], self.masks[i], self.image_size)


class _TargetDataset(Dataset):
    def __init__(self, target_name, image_size=256, split='test'):
        self.image_size = image_size
        _, _, test_ds = get_datasets(name=target_name, image_size=image_size, transform=None)
        self.images = test_ds.images
        self.masks = test_ds.masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return read_image_mask(self.images[i], self.masks[i], self.image_size)


class _MultiSourceDataset(Dataset):
    def __init__(self, source_names, image_size, limit=None):
        self.images = []
        self.masks = []
        self.image_size = image_size
        self.aadg_policy = Policy(default_policies())
        self.aug_prob = 0.5

        for name in source_names:
            train_ds, valid_ds, _ = get_datasets(name=name, image_size=image_size, transform=None)
            imgs = train_ds.images + valid_ds.images
            mks = train_ds.masks + valid_ds.masks
            if limit is not None:
                imgs = imgs[:limit]
                mks = mks[:limit]
            self.images.extend(imgs)
            self.masks.extend(mks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = load_image(self.images[i], self.image_size)
        mask = load_mask(self.masks[i], self.image_size)

        if np.random.random() < self.aug_prob:
            img, mask = self.aadg_policy(img, mask)

        return img_to_tensor(img), mask_to_tensor(mask)


def get_source_loader(source_name, image_size, batch_size, num_workers, pin_memory, limit_per_domain=None):
    dataset = _SourceCombinedDataset(source_name, image_size, limit=limit_per_domain)
    return _make_loader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )


def get_multi_source_loader(source_names, image_size, batch_size, num_workers, pin_memory, limit_per_domain=None):
    dataset = _MultiSourceDataset(source_names, image_size, limit=limit_per_domain)
    return _make_loader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )


def get_source_val_loader(source_name, image_size, batch_size, num_workers, pin_memory):
    dataset = _SourceValDataset(source_name, image_size)
    return _make_loader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )


def get_target_loader(target_name, image_size, batch_size, num_workers, pin_memory, split='test'):
    dataset = _TargetDataset(target_name, image_size, split=split)
    return _make_loader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
