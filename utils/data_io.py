import cv2
import numpy as np
import torch


def load_image(image_path, image_size):
    img = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size)).astype(np.float32) / 255.0
    return img


def load_mask(mask_path, image_size):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    mask = np.where(mask == 0, 0.0, 1.0).astype(np.float32)
    return mask


def img_to_tensor(img):
    return torch.from_numpy(img).permute(2, 0, 1)


def mask_to_tensor(mask):
    return torch.from_numpy(mask).unsqueeze(0)


def read_image_mask(image_path, mask_path, image_size):
    img = load_image(image_path, image_size)
    mask_ = load_mask(mask_path, image_size)
    return img_to_tensor(img), mask_to_tensor(mask_)
