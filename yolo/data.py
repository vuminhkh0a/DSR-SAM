"""Build YOLOv8 datasets from the existing DG splits and box_coords.json.

Nothing in ``box_coords.json`` is modified.  For every domain we copy the
images referenced by ``utils.data.get_datasets`` into a YOLO-format tree
(``images/<split>`` + ``labels/<split>``) and derive YOLO labels from the
ground-truth boxes already stored in ``box_coords.json``.  USOVA blank-GT
slices are excluded by ``get_datasets`` and therefore never enter training
or the generated box files, which keeps sample identity exactly in sync
with every DG method.
"""
from __future__ import annotations

import json
import os
import shutil
from functools import lru_cache
from pathlib import Path

from yolo import config as C
from utils.data import get_datasets


def _norm(path):
    return os.path.normpath(os.path.abspath(str(path)))


def load_gt_box_coords():
    with open(C.BOX_COORDS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _gt_entries_by_domain():
    raw = load_gt_box_coords()
    return {k: list(v) for k, v in raw.items()}


def gt_entries(box_key):
    """All entries for one box_coords.json domain, in file order."""
    return list(_gt_entries_by_domain().get(box_key, []))


def gt_box_map(box_key):
    """image path -> list of [x_min, y_min, x_max, y_max] GT boxes."""
    out = {}
    for e in gt_entries(box_key):
        out[_norm(e['image'])] = [list(b) for b in e.get('boxes', [])]
    return out


@lru_cache(maxsize=len(C.DOMAINS))
def split_lists(domain):
    """Return {split: [(image, mask), ...]} exactly as the DG methods see it."""
    train_ds, valid_ds, test_ds = get_datasets(
        name=domain, image_size=C.IMAGE_SIZE, transform=None)
    return {
        'train': list(zip(train_ds.images, train_ds.masks)),
        'val': list(zip(valid_ds.images, valid_ds.masks)),
        'test': list(zip(test_ds.images, test_ds.masks)),
    }


def split_sample_set(domain):
    """Union of train/val/test image paths (normalized) for one domain."""
    out = set()
    for pairs in split_lists(domain).values():
        for img, _ in pairs:
            out.add(_norm(img))
    return out


def ordered_entries(domain):
    """box_coords.json entries for ``domain`` that the DG splits actually use.

    Order is preserved from ``box_coords.json`` (so it is also the DG/data.py
    order); USOVA blank-GT entries are dropped because get_datasets drops them.
    """
    box_key = C.DOMAIN_TO_BOX_KEY[domain]
    keep = split_sample_set(domain)
    entries = [e for e in gt_entries(box_key) if _norm(e['image']) in keep]
    if len(entries) != len(keep):
        missing = len(keep) - len(entries)
        raise RuntimeError(
            f'{domain}: {missing} data.py samples are missing from '
            f'box_coords.json key {box_key!r}')
    return entries


def dataset_dir(domain):
    return C.DATASETS_DIR / C.DOMAIN_TO_BOX_KEY[domain]


def data_yaml(domain):
    return dataset_dir(domain) / 'data.yaml'


def data_test_yaml(domain):
    return dataset_dir(domain) / 'data_test.yaml'


def manifest_path(domain):
    return dataset_dir(domain) / 'manifest.json'


def load_manifest(domain):
    with open(manifest_path(domain), 'r', encoding='utf-8') as f:
        return json.load(f)


def _write_yaml(path, root, train, val, test=None):
    lines = [
        f'path: {Path(root).resolve()}',
        f'train: {train}',
        f'val: {val}',
    ]
    if test is not None:
        lines.append(f'test: {test}')
    lines += ['names:', '  0: lesion', '']
    Path(path).write_text('\n'.join(lines), encoding='utf-8')


def _yolo_label_lines(boxes):
    lines = []
    for x1, y1, x2, y2 in boxes:
        x1 = max(0.0, min(float(x1), C.IMAGE_SIZE))
        y1 = max(0.0, min(float(y1), C.IMAGE_SIZE))
        x2 = max(0.0, min(float(x2), C.IMAGE_SIZE))
        y2 = max(0.0, min(float(y2), C.IMAGE_SIZE))
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue
        cx = (x1 + x2) / 2.0 / C.IMAGE_SIZE
        cy = (y1 + y2) / 2.0 / C.IMAGE_SIZE
        nw = w / C.IMAGE_SIZE
        nh = h / C.IMAGE_SIZE
        lines.append(f'0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}')
    return lines


def build_domain_dataset(domain, force=False):
    """Create the copied image/label tree and data yamls for one domain."""
    box_key = C.DOMAIN_TO_BOX_KEY[domain]
    root = dataset_dir(domain)
    man_path = manifest_path(domain)
    if man_path.exists() and not force:
        return root

    if root.exists() and force:
        shutil.rmtree(root)
    for split in ('train', 'val', 'test'):
        (root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (root / 'labels' / split).mkdir(parents=True, exist_ok=True)

    gt_map = gt_box_map(box_key)
    manifest = {
        'domain': domain,
        'box_key': box_key,
        'image_size': C.IMAGE_SIZE,
        'splits': {},
    }
    for split, pairs in split_lists(domain).items():
        records = []
        for idx, (img, mask) in enumerate(pairs):
            suffix = Path(img).suffix.lower()
            name = f'{idx:06d}{suffix}'
            dst_img = root / 'images' / split / name
            dst_lbl = root / 'labels' / split / f'{idx:06d}.txt'
            shutil.copyfile(img, dst_img)
            boxes = gt_map.get(_norm(img), [])
            dst_lbl.write_text('\n'.join(_yolo_label_lines(boxes)) + ('\n' if boxes else ''),
                               encoding='utf-8')
            records.append({
                'image': img,
                'mask': mask,
                'boxes': boxes,
                'dataset_image': str(dst_img.relative_to(root)),
                'dataset_label': str(dst_lbl.relative_to(root)),
            })
        manifest['splits'][split] = records
        print(f'YOLO dataset: {domain} | {split}: {len(records)} images')

    man_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    _write_yaml(root / 'data.yaml', root, 'images/train', 'images/val',
                test='images/test')
    # The reference validator reads data['val']; point it at the test split for
    # target-domain evaluation.
    _write_yaml(root / 'data_test.yaml', root, 'images/test', 'images/test')
    return root


def build_all_datasets(force=False, domains=None):
    domains = list(domains or C.DOMAINS)
    for domain in domains:
        build_domain_dataset(domain, force=force)


def split_entries(domain, split):
    """Manifest entries (image, mask, boxes) for one split."""
    return list(load_manifest(domain)['splits'][split])


if __name__ == '__main__':
    build_all_datasets()
