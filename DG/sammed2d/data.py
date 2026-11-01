"""
SAM-Med2D data loading: test-only box-prompt evaluation.

Images/masks are read with utils.data_io (same as every other DG method);
USOVA blank-GT slices are already filtered by utils.data.get_datasets.
Boxes come from y_DG/box_coords.json (256-space) by default
(prompt_method='gt') or box_coords_yolov8_<size>.json
(prompt_method='yolov8'). Native resolution is 256, so no box rescaling.
"""
import torch
from torch.utils.data import Dataset

from utils.data import _make_loader, get_datasets
from utils.data_io import read_image_mask
from utils.prompts import boxes_for_images

class SAMMed2DBoxDataset(Dataset):
    """Returns (image [3,H,W], mask [1,H,W], bbox [4]) per slice; missing
    boxes fall back to the full image."""

    def __init__(self, images, masks, image_size=256, boxes=None):
        self.images = images
        self.masks = masks
        self.image_size = image_size
        if boxes is None:
            boxes = [[0, 0, image_size, image_size]] * len(images)
        self.boxes = [list(b) for b in boxes]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image, mask = read_image_mask(self.images[i], self.masks[i],
                                      self.image_size)
        bbox = torch.as_tensor(self.boxes[i], dtype=torch.float32)
        return image, mask, bbox


def _collect_split(name, image_size, split):
    train_ds, valid_ds, test_ds = get_datasets(name=name,
                                               image_size=image_size,
                                               transform=None)
    ds = {'train': train_ds, 'val': valid_ds, 'test': test_ds}[split]
    return ds.images, ds.masks


def get_sammed2d_target_loader(target_name, image_size, batch_size,
                             num_workers, pin_memory, split='test',
                             prompt_method='gt', yolo_size='m', source_name=None):
    """Target loader with standard box prompts (split='test')."""
    imgs, masks = _collect_split(target_name, image_size, split)
    boxes = boxes_for_images(
        imgs, prompt_method=prompt_method, yolo_size=yolo_size,
        source=source_name if source_name is not None else target_name,
        target=target_name, image_size=image_size,
    )
    dataset = SAMMed2DBoxDataset(imgs, masks, image_size=image_size, boxes=boxes)
    return _make_loader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)
