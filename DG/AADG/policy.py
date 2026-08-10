import random
import numpy as np
import cv2


random_mirror = True


def _shear_x(img, mask, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    h, w = img.shape[:2]
    M = np.float32([[1, v, 0], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return img, mask


def _shear_y(img, mask, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    h, w = img.shape[:2]
    M = np.float32([[1, 0, 0], [v, 1, 0]])
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return img, mask


def _translate_x(img, mask, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    h, w = img.shape[:2]
    v_px = int(v * w)
    M = np.float32([[1, 0, v_px], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return img, mask


def _translate_y(img, mask, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    h, w = img.shape[:2]
    v_px = int(v * h)
    M = np.float32([[1, 0, 0], [0, 1, v_px]])
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return img, mask


def _rotate(img, mask, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), v, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return img, mask


def _auto_contrast(img, mask, _):
    img_uint8 = (img * 255).astype(np.uint8)
    mn = img_uint8.min(axis=(0, 1), keepdims=True)
    mx = img_uint8.max(axis=(0, 1), keepdims=True)
    scale = 255.0 / (mx - mn + 1e-7)
    img = ((img_uint8 - mn) * scale).astype(np.float32) / 255.0
    return np.clip(img, 0, 1), mask


def _invert(img, mask, _):
    return 1.0 - img, mask


def _equalize(img, mask, _):
    img_uint8 = (img * 255).astype(np.uint8)
    if img_uint8.ndim == 3:
        for c in range(3):
            img_uint8[..., c] = cv2.equalizeHist(img_uint8[..., c])
    else:
        img_uint8 = cv2.equalizeHist(img_uint8)
    return img_uint8.astype(np.float32) / 255.0, mask


def _solarize(img, mask, v):
    threshold = int(v * 255)
    img_uint8 = (img * 255).astype(np.uint8)
    img_uint8 = np.where(img_uint8 < threshold, img_uint8, 255 - img_uint8)
    return img_uint8.astype(np.float32) / 255.0, mask


def _posterize(img, mask, v):
    bits = max(1, int(v))
    shift = 8 - bits
    img_uint8 = (img * 255).astype(np.uint8)
    img_uint8 = (img_uint8 >> shift) << shift
    return img_uint8.astype(np.float32) / 255.0, mask


def _contrast(img, mask, v):
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    mn = gray.mean()
    img = img * v + (1 - v) * (mn / 255.0)
    return np.clip(img, 0, 1), mask


def _color(img, mask, v):
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_3c = np.stack([gray] * 3, axis=-1).astype(np.float32) / 255.0
    img = img * v + gray_3c * (1 - v)
    return np.clip(img, 0, 1), mask


def _brightness(img, mask, v):
    return np.clip(img * v, 0, 1), mask


def _sharpness(img, mask, v):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    img_uint8 = (img * 255).astype(np.uint8)
    sharp = cv2.filter2D(img_uint8, -1, kernel).astype(np.float32) / 255.0
    img = img * (1 - v) + sharp * v
    return np.clip(img, 0, 1), mask


def _cutout(img, mask, v):
    if v <= 0:
        return img, mask
    h, w = img.shape[:2]
    cut_h = int(h * v)
    cut_w = int(w * v)
    if cut_h < 1 or cut_w < 1:
        return img, mask
    cx = np.random.randint(0, w)
    cy = np.random.randint(0, h)
    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(w, x1 + cut_w)
    y2 = min(h, y1 + cut_h)
    img[y1:y2, x1:x2] = 0.5
    mask[y1:y2, x1:x2] = 0.0
    return img, mask


_AUGMENT_LIST = [
    (_auto_contrast, 0, 1),
    (_invert, 0, 1),
    (_equalize, 0, 1),
    (_solarize, 0, 1),
    (_posterize, 4, 8),
    (_contrast, 0.1, 1.9),
    (_color, 0.1, 1.9),
    (_brightness, 0.1, 1.9),
    (_sharpness, 0.1, 1.9),
    (_cutout, 0, 0.2),
]


def augment_list_raw():
    return list(_AUGMENT_LIST)


def augment_list():
    return _AUGMENT_LIST


def apply_augment(img, mask, name, level):
    for fn, low, high in _AUGMENT_LIST:
        if fn.__name__ == name:
            v = level * (high - low) + low
            return fn(img.copy(), mask.copy(), v)
    return img, mask


class Policy:
    def __init__(self, policy):
        self.policy = policy

    def __call__(self, img, mask):
        sub_policy = random.choice(self.policy)
        for op_name, mag in sub_policy:
            img, mask = apply_augment(img, mask, op_name, mag)
        return img, mask


def default_policies():
    names = [fn.__name__ for fn, _, _ in _AUGMENT_LIST]
    policies = [
        [(names[0], 0.5), (names[7], 0.5)],
        [(names[2], 0.5), (names[8], 0.5)],
        [(names[5], 0.5), (names[4], 0.5)],
        [(names[3], 0.3), (names[9], 0.3)],
        [(names[8], 0.5), (names[5], 0.5)],
        [(names[1], 0.5), (names[2], 0.5)],
        [(names[0], 0.5), (names[6], 0.5)],
        [(names[6], 0.5), (names[7], 0.5)],
        [(names[4], 0.5), (names[3], 0.5)],
        [(names[7], 0.5), (names[0], 0.5)],
    ]
    return [policies]
