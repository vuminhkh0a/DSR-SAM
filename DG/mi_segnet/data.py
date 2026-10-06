import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.data import get_datasets, _make_loader
from utils.data_io import load_image, load_mask
from utils.seed import set_seed

set_seed()


class _MISegNetPairedDataset(Dataset):
    def __init__(self, source_name, image_size, limit=None):
        super().__init__()
        train_ds, valid_ds, _ = get_datasets(name=source_name, image_size=image_size, transform=None)
        self.images = train_ds.images + valid_ds.images
        self.masks = train_ds.masks + valid_ds.masks
        self.image_size = image_size
        self.weights_aug = np.array([5, 2, 1, 4, 3], dtype=np.float32)
        self.prob_list = np.array([0.111, 0.222, 0.333, 0.444, 0.555, 0.666, 0.777, 0.888, 0.999])
        if limit is not None:
            self.images = self.images[:limit]
            self.masks = self.masks[:limit]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = load_image(self.images[i], self.image_size)
        mask = load_mask(self.masks[i], self.image_size)

        img_uint8 = (img * 255).astype(np.uint8)
        mask_uint8 = (mask * 255).astype(np.uint8)

        aug_style_diff = 0
        aug_spatial_diff = 1
        while aug_style_diff < 3 and aug_spatial_diff != 0:
            prob_style_1 = np.random.rand(5)
            alpha_style_1 = np.random.rand(5)
            prob_spatial_1 = np.random.rand(3)
            alpha_spatial_1 = np.random.rand(3)
            prob_style_2 = np.random.rand(5)
            alpha_style_2 = np.random.rand(5)
            prob_spatial_2 = np.random.rand(3)
            alpha_spatial_2 = np.random.rand(3)

            aug_style_diff = np.sum(np.abs(
                (prob_style_1 < 0.3) * alpha_style_1 * self.weights_aug -
                (prob_style_2 < 0.3) * alpha_style_2 * self.weights_aug
            ))
            aug_spatial_diff = np.sum(
                (self.prob_list > prob_spatial_1[1]) * (self.prob_list < prob_spatial_1[1] + 0.111) *
                (self.prob_list > prob_spatial_2[1]) * (self.prob_list < prob_spatial_2[1] + 0.111)
            )

        aug1, label1 = self._apply_augmentation(prob_style_1, alpha_style_1, prob_spatial_1, alpha_spatial_1, img_uint8, mask_uint8)
        aug2, label2 = self._apply_augmentation(prob_style_2, alpha_style_2, prob_spatial_2, alpha_spatial_2, img_uint8, mask_uint8)
        aug12, _ = self._apply_augmentation(prob_style_1, alpha_style_1, prob_spatial_2, alpha_spatial_2, img_uint8, mask_uint8)
        aug21, _ = self._apply_augmentation(prob_style_2, alpha_style_2, prob_spatial_1, alpha_spatial_1, img_uint8, mask_uint8)

        def _to_tensor(x):
            return torch.from_numpy(x.astype(np.float32) / 255.0).permute(2, 0, 1) if x.ndim == 3 else torch.from_numpy(x.astype(np.float32) / 255.0).unsqueeze(0)

        return (
            _to_tensor(aug1), _to_tensor(aug2),
            _to_tensor(aug12), _to_tensor(aug21),
            _to_tensor(label1), _to_tensor(label2),
        )

    def _apply_augmentation(self, prob_style, alpha_style, prob_spatial, alpha_spatial, img, label):
        h, w = img.shape[:2]
        if prob_spatial[0] < 0.5:
            img, label = self._crop(img, label, alpha_spatial[0:2], prob_spatial[1])
        if prob_spatial[2] < 0.05:
            img, label = self._flip(img, label)
        if prob_style[0] < 0.1:
            img = self._sharpness(img, alpha_style[0])
        if prob_style[1] < 0.1:
            img = self._blurriness(img, alpha_style[1])
        if prob_style[2] < 0.1:
            img = self._noise(img, alpha_style[2])
        if prob_style[3] < 0.1:
            img = self._brightness(img, alpha_style[3])
        if prob_style[4] < 0.1:
            img = self._contrast(img, alpha_style[4])
        return img, label

    def _crop(self, img, label, alpha, prob):
        alpha = alpha * 0.2 + 0.7
        h, w = img.shape[:2]
        croped_h, croped_w = int(h * alpha[0]), int(w * alpha[1])

        if prob < 0.111:
            c = [0, 0]
        elif prob < 0.222:
            c = [0, int(w * (1 - alpha[1]))]
        elif prob < 0.333:
            c = [0, int(w * (1 - alpha[1]) // 2)]
        elif prob < 0.444:
            c = [int(h * (1 - alpha[0])), 0]
        elif prob < 0.555:
            c = [int(h * (1 - alpha[0])), int(w * (1 - alpha[1]))]
        elif prob < 0.666:
            c = [int(h * (1 - alpha[0])), int(w * (1 - alpha[1]) // 2)]
        elif prob < 0.777:
            c = [int(h * (1 - alpha[0]) // 2), 0]
        elif prob < 0.888:
            c = [int(h * (1 - alpha[0]) // 2), int(w * (1 - alpha[1]))]
        else:
            c = [int(h * (1 - alpha[0]) // 2), int(w * (1 - alpha[1]) // 2)]

        img_crop = img[c[0]:croped_h + c[0], c[1]:croped_w + c[1]]
        label_crop = label[c[0]:croped_h + c[0], c[1]:croped_w + c[1]]
        interp = cv2.INTER_LANCZOS4 if img_crop.shape[0] > 0 and img_crop.shape[1] > 0 else cv2.INTER_NEAREST
        img_out = cv2.resize(img_crop, (self.image_size, self.image_size), interpolation=interp)
        label_out = cv2.resize(label_crop, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        if label_out.ndim == 2 and img_out.ndim == 3:
            label_out = np.expand_dims(label_out, -1)
        return img_out, label_out

    def _flip(self, img, label):
        return cv2.flip(img, 1), cv2.flip(label, 1)

    def _sharpness(self, img, alpha):
        alpha = alpha * 20 + 10
        blur = cv2.GaussianBlur(img, (0, 0), 1.0)
        blurr = cv2.GaussianBlur(blur, (0, 0), 1.0)
        return np.clip(cv2.addWeighted(blur, alpha + 1, blurr, -alpha, 0), 0, 255).astype(np.uint8)

    def _blurriness(self, img, alpha):
        alpha = alpha * 1.25 + 0.25
        return np.clip(cv2.GaussianBlur(img, (0, 0), alpha), 0, 255).astype(np.uint8)

    def _noise(self, img, alpha):
        alpha = alpha * 0.04 + 0.01
        gaussian = np.random.normal(0, alpha, img.shape).astype(np.float32) * 255
        return np.clip(img.astype(np.float32) + gaussian, 0, 255).astype(np.uint8)

    def _brightness(self, img, alpha):
        alpha = int((alpha * 0.2 - 0.1) * 255)
        return np.clip(img.astype(np.int32) + alpha, 0, 255).astype(np.uint8)

    def _contrast(self, img, alpha):
        alpha = alpha * 2.5 + 0.5
        inv_gamma = 1.0 / alpha
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(img, table)


def get_source_loader(source_name, image_size, batch_size, num_workers, pin_memory, limit_per_domain=None):
    dataset = _MISegNetPairedDataset(source_name, image_size, limit=limit_per_domain)
    return _make_loader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
