import numpy as np
import torch
from torch.utils.data import Dataset
from utils.data import get_datasets
from utils.seed import set_seed
from utils.data_io import load_image, load_mask, img_to_tensor, mask_to_tensor

set_seed()


class _CDDSAMultiSourceDataset(Dataset):
    def __init__(self, source_names, image_size):
        self.image_size = image_size
        self.num_domains = len(source_names)
        self.source_names = source_names
        self.domain_images = []
        self.domain_masks = []

        for name in source_names:
            train_ds, valid_ds, _ = get_datasets(name=name, image_size=image_size, transform=None)
            imgs = train_ds.images + valid_ds.images
            mks = train_ds.masks + valid_ds.masks
            self.domain_images.append(imgs)
            self.domain_masks.append(mks)

        self._len = max(len(imgs) for imgs in self.domain_images)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        sample = []
        for d in range(self.num_domains):
            i = np.random.randint(len(self.domain_images[d]))
            img = load_image(self.domain_images[d][i], self.image_size)
            mask = load_mask(self.domain_masks[d][i], self.image_size)
            sample.append({'image': img_to_tensor(img), 'label': mask_to_tensor(mask), 'dc': d})
        return sample


def get_cddsa_train_loader(source_names, image_size, batch_size, num_workers, pin_memory):
    dataset = _CDDSAMultiSourceDataset(source_names, image_size)
    from utils.data import _make_loader
    return _make_loader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )


def get_cddsa_val_loader(source_name, image_size, batch_size, num_workers, pin_memory):
    from utils.data import get_source_val_loader
    return get_source_val_loader(source_name, image_size, batch_size, num_workers, pin_memory)


def get_cddsa_target_loader(target_name, image_size, batch_size, num_workers, pin_memory, split='test'):
    from utils.data import get_target_loader
    return get_target_loader(target_name, image_size, batch_size, num_workers, pin_memory, split=split)