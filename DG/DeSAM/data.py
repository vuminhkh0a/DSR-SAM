import numpy as np
import torch
from torch.utils.data import Dataset

from utils.data import _make_loader, get_datasets
from utils.data_io import read_image_mask


class DeSAMDataset(Dataset):
    """DeSAM point-prompt dataset (gridpoints training mode).

    Mirrors the repo's ProstateDataset: for each slice a single prompt point
    is sampled - background point with prob neg_points/(neg_points+1),
    foreground point otherwise - together with its IoU label (0/1).
    Coordinates are returned at 1024-resolution (256 * image_scale).
    """

    def __init__(self, images, masks, neg_points=1, image_size=256):
        self.images = images
        self.masks = masks
        self.neg_points = neg_points
        self.image_size = image_size
        self.image_scale = 4  # 1024 / 256

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, mask = read_image_mask(self.images[index], self.masks[index], self.image_size)
        mask_np = mask[0].numpy()

        is_background = np.random.randint(0, self.neg_points + 1)
        if is_background:
            y_indices, x_indices = np.where(mask_np == 0)
            if len(y_indices) == 0:
                y_indices, x_indices = np.where(mask_np > 0)
                iou_label = torch.tensor([1.0])
            else:
                iou_label = torch.tensor([0.0])
        else:
            y_indices, x_indices = np.where(mask_np > 0)
            if len(y_indices) == 0:
                y_indices, x_indices = np.where(mask_np == 0)
                iou_label = torch.tensor([0.0])
            else:
                iou_label = torch.tensor([1.0])

        random_idx = np.random.randint(0, len(y_indices))
        prompt_points = np.array((
            x_indices[random_idx] * self.image_scale,
            y_indices[random_idx] * self.image_scale,
        ))

        return image, mask, torch.as_tensor(prompt_points).float(), 1, iou_label


def get_desam_loaders(source_name, image_size, batch_size, num_workers, pin_memory, neg_points=1):
    train_ds, valid_ds, _ = get_datasets(name=source_name, image_size=image_size, transform=None)

    train_dataset = DeSAMDataset(train_ds.images, train_ds.masks, neg_points=neg_points, image_size=image_size)
    valid_dataset = DeSAMDataset(valid_ds.images, valid_ds.masks, neg_points=neg_points, image_size=image_size)

    train_loader = _make_loader(train_dataset, batch_size, shuffle=True, num_workers=num_workers,
                                pin_memory=pin_memory, drop_last=True)
    valid_loader = _make_loader(valid_dataset, 1, shuffle=False, num_workers=num_workers,
                                pin_memory=pin_memory)
    return train_loader, valid_loader
