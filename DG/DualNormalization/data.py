import torch
from torch.utils.data import Dataset
from utils.data import get_datasets, _make_loader
from utils.data_io import load_image, load_mask, img_to_tensor, mask_to_tensor
from utils.style_aug import nonlinear_transformation_multi_channel


def _read_image_mask(image_path, mask_path, image_size, apply_style_aug=False, style_aug_prob=0.5):
    img = load_image(image_path, image_size)
    mask = load_mask(mask_path, image_size)
    if apply_style_aug:
        img = nonlinear_transformation_multi_channel(img, prob=style_aug_prob)
    return img_to_tensor(img), mask_to_tensor(mask)


# ---------------------------------------------------------------------------
# Single-source combined dataset: reads each image ONCE
# ---------------------------------------------------------------------------

class _SingleSourceCombinedDataset(Dataset):
    def __init__(self, source_name, image_size, limit=None):
        train_ds, valid_ds, _ = get_datasets(name=source_name, image_size=image_size, transform=None)
        self.images = train_ds.images + valid_ds.images
        self.masks = train_ds.masks + valid_ds.masks
        self.image_size = image_size
        if limit is not None:
            self.images = self.images[:limit]
            self.masks = self.masks[:limit]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = load_image(self.images[i], self.image_size)
        mask = load_mask(self.masks[i], self.image_size)

        mask_t = mask_to_tensor(mask)

        img_orig = img_to_tensor(img)
        label_0 = torch.tensor(0, dtype=torch.long)

        img_aug = nonlinear_transformation_multi_channel(img, prob=1.0)
        img_aug_t = img_to_tensor(img_aug)
        label_1 = torch.tensor(1, dtype=torch.long)

        return img_orig, mask_t.clone(), label_0, img_aug_t, mask_t.clone(), label_1


def _collate_single_source(batch):
    orig_imgs = torch.stack([item[0] for item in batch])
    orig_masks = torch.stack([item[1] for item in batch])
    orig_labels = torch.stack([item[2] for item in batch])
    aug_imgs = torch.stack([item[3] for item in batch])
    aug_masks = torch.stack([item[4] for item in batch])
    aug_labels = torch.stack([item[5] for item in batch])
    return [(orig_imgs, orig_masks, orig_labels), (aug_imgs, aug_masks, aug_labels)]


# ---------------------------------------------------------------------------
# Original per-domain datasets (used in multi-source mode)
# ---------------------------------------------------------------------------

class _SourceDomainDataset(Dataset):
    def __init__(self, source_name, image_size, domain_id=0,
                 apply_style_aug=False, style_aug_prob=0.5, limit=None):
        self.image_size = image_size
        self.domain_id = domain_id
        self.apply_style_aug = apply_style_aug
        self.style_aug_prob = style_aug_prob

        train_ds, valid_ds, _ = get_datasets(name=source_name, image_size=image_size, transform=None)
        self.images = train_ds.images + valid_ds.images
        self.masks = train_ds.masks + valid_ds.masks

        if limit is not None:
            self.images = self.images[:limit]
            self.masks = self.masks[:limit]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image_t, mask_t = _read_image_mask(
            self.images[i], self.masks[i], self.image_size,
            apply_style_aug=self.apply_style_aug, style_aug_prob=self.style_aug_prob,
        )
        domain_label = torch.tensor(self.domain_id, dtype=torch.long)
        return image_t, mask_t, domain_label


# ---------------------------------------------------------------------------
# Loader factories
# ---------------------------------------------------------------------------

def get_single_source_loaders(source_name, image_size, batch_size, num_workers, pin_memory,
                               limit_per_domain=None):
    dataset = _SingleSourceCombinedDataset(
        source_name, image_size, limit=limit_per_domain
    )
    loader = _make_loader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=True, collate_fn=_collate_single_source,
    )
    return [loader]


class _MultiSourceDataset(Dataset):
    def __init__(self, source_names, image_size, apply_style_aug=True, limit=None):
        self.image_size = image_size
        self.apply_style_aug = apply_style_aug
        self.samples = []

        for domain_id, name in enumerate(source_names):
            train_ds, valid_ds, _ = get_datasets(name=name, image_size=image_size, transform=None)
            imgs = train_ds.images + valid_ds.images
            masks_l = train_ds.masks + valid_ds.masks

            if limit is not None:
                imgs = imgs[:limit]
                masks_l = masks_l[:limit]

            for img_p, mask_p in zip(imgs, masks_l):
                self.samples.append({
                    'img_path': img_p,
                    'gt_path': mask_p,
                    'domain_id': domain_id,
                    'domain_name': name,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        image_t, mask_t = _read_image_mask(
            s['img_path'], s['gt_path'], self.image_size,
            apply_style_aug=self.apply_style_aug, style_aug_prob=0.5,
        )
        domain_label = torch.tensor(s['domain_id'], dtype=torch.long)
        return image_t, mask_t, domain_label


class _MultiSourceDatasetSubset(Dataset):
    def __init__(self, full_dataset, sample_infos, image_size, apply_style_aug=True):
        self.samples = sample_infos
        self.image_size = image_size
        self.apply_style_aug = apply_style_aug

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        image_t, mask_t = _read_image_mask(
            s['img_path'], s['gt_path'], self.image_size,
            apply_style_aug=self.apply_style_aug, style_aug_prob=0.5,
        )
        domain_label = torch.tensor(s['domain_id'], dtype=torch.long)
        return image_t, mask_t, domain_label


def get_multi_source_loaders(source_names, image_size, batch_size, num_workers, pin_memory,
                              apply_style_aug=True, limit_per_domain=None):
    dataset = _MultiSourceDataset(source_names, image_size, apply_style_aug=apply_style_aug, limit=limit_per_domain)

    domain_datasets = {name: [] for name in source_names}
    for s in dataset.samples:
        domain_datasets[s['domain_name']].append(s)

    loaders = []
    for name in source_names:
        sub_dataset = _MultiSourceDatasetSubset(dataset, domain_datasets[name], image_size,
                                                 apply_style_aug=apply_style_aug)
        loader = _make_loader(
            sub_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
        )
        loaders.append(loader)

    return loaders


def get_source_loaders(source_names, image_size, batch_size, num_workers, pin_memory,
                       mode='multi_source', apply_style_aug=True, limit_per_domain=None):
    if mode == 'single_source':
        if len(source_names) != 1:
            raise ValueError(f"Single-source mode expects exactly 1 source, got {len(source_names)}")
        return get_single_source_loaders(
            source_names[0], image_size, batch_size, num_workers, pin_memory,
            limit_per_domain=limit_per_domain,
        )
    elif mode == 'multi_source':
        return get_multi_source_loaders(
            source_names, image_size, batch_size, num_workers, pin_memory,
            apply_style_aug=apply_style_aug, limit_per_domain=limit_per_domain,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'single_source' or 'multi_source'.")


# ---------------------------------------------------------------------------
# Validation and target datasets
# ---------------------------------------------------------------------------

class _SourceValDataset(Dataset):
    def __init__(self, source_name, image_size):
        _, valid_ds, _ = get_datasets(name=source_name, image_size=image_size, transform=None)
        self.images = valid_ds.images
        self.masks = valid_ds.masks
        self.image_size = image_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return _read_image_mask(self.images[i], self.masks[i], self.image_size)


class _TargetDataset(Dataset):
    def __init__(self, target_name, image_size=256, split='test'):
        self.image_size = image_size
        _, _, test_ds = get_datasets(name=target_name, image_size=image_size, transform=None)
        self.images = test_ds.images
        self.masks = test_ds.masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return _read_image_mask(self.images[i], self.masks[i], self.image_size)


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
