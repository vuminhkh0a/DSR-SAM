import numpy as np
import random
import torch
from torch.utils.data import Dataset
from utils.data import get_datasets
from utils.data_io import load_image, load_mask, img_to_tensor, mask_to_tensor, read_image_mask
from utils.seed import set_seed

set_seed()


def soft_label(domain_idx, num_domains):
    """Generate soft domain label matching paper's SoftLable().

    Correct class gets 0.8-1.0 (random); remainder distributed randomly
    among other classes. Sum = 1.0.
    """
    label = np.zeros(num_domains, dtype=np.float32)
    correct_val = 0.8 + random.random() * 0.2
    label[domain_idx] = correct_val
    remaining = 1.0 - correct_val
    other_indices = [i for i in range(num_domains) if i != domain_idx]
    for i, idx in enumerate(other_indices):
        if i == len(other_indices) - 1:
            label[idx] = remaining
        else:
            val = random.random() * remaining
            label[idx] = val
            remaining -= val
    return label


class _DoFEMultiSourceDataset(Dataset):
    """Multi-source dataset returning domain-balanced batches.

    Each __getitem__ returns a pre-batched tensor containing
    samples_per_domain images per domain, stacked along batch dim.

    Matches the paper's batch construction: each training iteration
    samples equally from all source domains.
    """
    def __init__(self, source_names, image_size, samples_per_domain=4):
        self.samples_per_domain = samples_per_domain
        self.image_size = image_size
        self.num_domains = len(source_names)
        self.domain_images = []
        self.domain_masks = []
        self.source_names = source_names

        for name in source_names:
            train_ds, valid_ds, _ = get_datasets(name=name, image_size=image_size, transform=None)
            imgs = train_ds.images + valid_ds.images
            mks = train_ds.masks + valid_ds.masks
            self.domain_images.append(imgs)
            self.domain_masks.append(mks)

        self.domain_lengths = [len(imgs) for imgs in self.domain_images]
        self._size = max(self.domain_lengths) // samples_per_domain

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        batch_images, batch_masks, batch_domains = [], [], []

        for d in range(self.num_domains):
            for _ in range(self.samples_per_domain):
                i = np.random.randint(len(self.domain_images[d]))
                img = load_image(self.domain_images[d][i], self.image_size)
                mask = load_mask(self.domain_masks[d][i], self.image_size)
                batch_images.append(img_to_tensor(img))
                batch_masks.append(mask_to_tensor(mask))
                batch_domains.append(d)

        images = torch.stack(batch_images)
        masks = torch.stack(batch_masks)
        domain_labels = torch.tensor(batch_domains, dtype=torch.long)

        soft_labels = torch.stack([
            torch.from_numpy(soft_label(d.item(), self.num_domains))
            for d in domain_labels
        ])

        return images, masks, domain_labels, soft_labels


class _DoFEPretrainDataset(Dataset):
    """Combined multi-source dataset for pretraining phase.

    Simply concatenates all images from all source domains.
    """
    def __init__(self, source_names, image_size):
        self.images = []
        self.masks = []
        self.image_size = image_size

        for name in source_names:
            train_ds, valid_ds, _ = get_datasets(name=name, image_size=image_size, transform=None)
            self.images.extend(train_ds.images + valid_ds.images)
            self.masks.extend(train_ds.masks + valid_ds.masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return read_image_mask(self.images[i], self.masks[i], self.image_size)


def get_dofe_train_loader(source_names, image_size, batch_size, num_workers, pin_memory):
    num_domains = len(source_names)
    samples_per_domain = max(1, batch_size // num_domains)
    dataset = _DoFEMultiSourceDataset(source_names, image_size, samples_per_domain=samples_per_domain)
    from utils.data import _make_loader
    return _make_loader(
        dataset, batch_size=1, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )


def get_pretrain_loader(source_names, image_size, batch_size, num_workers, pin_memory):
    dataset = _DoFEPretrainDataset(source_names, image_size)
    from utils.data import _make_loader
    return _make_loader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )


def get_dofe_val_loader(source_name, image_size, batch_size, num_workers, pin_memory):
    from utils.data import get_source_val_loader
    return get_source_val_loader(source_name, image_size, batch_size, num_workers, pin_memory)


def get_dofe_target_loader(target_name, image_size, batch_size, num_workers, pin_memory, split='test'):
    from utils.data import get_target_loader
    return get_target_loader(target_name, image_size, batch_size, num_workers, pin_memory, split=split)
