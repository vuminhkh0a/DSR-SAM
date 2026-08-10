import numpy as np
import os
import cv2
import json
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from utils.seed import *

set_seed()
generator = get_generator()


def _make_loader(dataset, batch_size, shuffle, num_workers, pin_memory, drop_last=False, collate_fn=None):
    kwargs = dict(
        batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=drop_last,
        generator=generator, worker_init_fn=worker_init_fn,
    )
    if num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = 4
    if collate_fn is not None:
        kwargs['collate_fn'] = collate_fn
    return DataLoader(dataset, **kwargs)


class Custom_Dataset(Dataset):
    def __init__(self, images, masks, transform=None, name=None, image_size=None):

        self.images = images
        self.masks = masks
        self.name = name
        self.transform = transform
        self.image_size = image_size

        self.no_transform = A.Compose([A.ToTensorV2()])
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):

        image_path = self.images[i]
        image = cv2.resize(cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), (self.image_size, self.image_size)) / 255.0
        mask_path = self.masks[i]
        mask = cv2.resize(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(np.where(mask == 0, 0.0, 1.0), -1)

        if self.transform:
            aug = self.transform(image=image.astype(np.float32), mask=mask.astype(np.float32))
            image, mask = aug['image'], torch.tensor(aug['mask']).permute(2, 0, 1)
        else:
            aug = self.no_transform(image=image.astype(np.float32), mask=mask.astype(np.float32))
            image = aug['image']
            mask = aug['mask'].permute(2, 0, 1)

        return image.float(), mask.float()
    


def get_datasets(name, image_size, transform):

    train_x, train_y, valid_x, valid_y, test_x, test_y = [], [], [], [], [], []

    if name == 'OTU':
        OTU_PATH = '/mnt/nvme0/home/utbt/KhoaVM/OTU-2D-Dataset/OTU_2D/'
        with open('/mnt/nvme0/home/utbt/KhoaVM/OTU-2D-Dataset/OTU_2D_850-150-469.json', 'r') as f:
            data = json.load(f)
        for item in data:
            for i in range(len(data[item])):
                if item == 'train':
                    train_x.append(OTU_PATH + str(data[item][i]['image']))
                    train_y.append(OTU_PATH + str(data[item][i]['mask']))
                elif item == 'val':
                    valid_x.append(OTU_PATH + str(data[item][i]['image']))
                    valid_y.append(OTU_PATH + str(data[item][i]['mask']))
                elif item == 'test':
                    test_x.append(OTU_PATH + str(data[item][i]['image']))
                    test_y.append(OTU_PATH + str(data[item][i]['mask']))


    elif name == 'USOVA':
        ANNOTATOR = "follicle_r1"
        VARIANT   = "binary"
        DATASET_ROOT = "/mnt/nvme0/home/utbt/KhoaVM/USOVA3D_Dataset"
        s = '/mnt/nvme0/home/utbt/KhoaVM/USOVA3D_Dataset/split.json'
        with open(s, encoding="utf-8") as f:
            split = json.load(f)
        def get_split(split_name, annotator=ANNOTATOR, variant=VARIANT):
            imgs, masks = [], []
            for _, data in split["split"][split_name].items():
                for img_rel in data["images"]:
                    imgs.append(os.path.join(DATASET_ROOT, img_rel))
                for mask_rel in data["labels"][annotator][variant]:
                    masks.append(os.path.join(DATASET_ROOT, mask_rel))
            return imgs, masks
        train_x, train_y = get_split("train")
        valid_x, valid_y = get_split("val")
        test_x,  test_y  = get_split("test")


    elif name == 'OVATUS':
        JSON_PATH = '/mnt/nvme0/home/utbt/KhoaVM/OvaTUS/annotations_split.json'
        BASE = '/mnt/nvme0/home/utbt/KhoaVM/'
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            split = item.get('split', '').lower()
            img = os.path.join(BASE, item.get('file_path_img', ''))
            ann = os.path.join(BASE, item.get('file_path_ann', ''))
            if split == 'train':
                train_x.append(img)
                train_y.append(ann)
            elif split == 'val':
                valid_x.append(img)
                valid_y.append(ann)
            elif split == 'test':
                test_x.append(img)
                test_y.append(ann)
    
    
    
    print(f"Dataset: {name} | Training data: {len(train_x)} | Validation data: {len(valid_x)} | Testing data: {len(test_x)}")

    train_dataset = Custom_Dataset(images=train_x, masks=train_y, name=name, image_size=image_size, transform=transform)
    valid_dataset = Custom_Dataset(images=valid_x, masks=valid_y, name=name, image_size=image_size, transform=False)
    test_dataset = Custom_Dataset(images=test_x, masks=test_y, name=name, image_size=image_size, transform=False)
  
    return train_dataset, valid_dataset, test_dataset


class _SourceValDataset(Dataset):
    def __init__(self, source_name, image_size):
        _, valid_ds, _ = get_datasets(name=source_name, image_size=image_size, transform=None)
        self.images = valid_ds.images
        self.masks = valid_ds.masks
        self.image_size = image_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        from utils.data_io import read_image_mask
        return read_image_mask(self.images[i], self.masks[i], self.image_size)


class _TargetDataset(Dataset):
    def __init__(self, target_name, image_size=256, split='test'):
        _, _, test_ds = get_datasets(name=target_name, image_size=image_size, transform=None)
        self.images = test_ds.images
        self.masks = test_ds.masks
        self.image_size = image_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        from utils.data_io import read_image_mask
        return read_image_mask(self.images[i], self.masks[i], self.image_size)


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


def get_dataloaders(name, image_size, transform, batch_size, num_workers, pin_memory):
    train_dataset, valid_dataset, test_dataset = get_datasets(name=name, image_size=image_size, transform=transform)
    train_loader = _make_loader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    valid_loader = _make_loader(valid_dataset, batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = _make_loader(test_dataset, batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader
