import os
import json
import math
import argparse
import cv2
import numpy as np
from utils.data_io import load_mask


def extract_boxes_from_mask(mask_path, image_size=256, tightness=0.1):
    """Read a 1-channel ground-truth mask with utils/data_io.load_mask and extract one
    bounding box [x_low, y_low, x_high, y_high] (SAM order) per connected region.

    Args:
        mask_path (str): full path to the binary mask (values {0, 255} or {0, 1}).
        image_size (int): masks are resized to (image_size, image_size) by load_mask,
                          so the returned coordinates live in that resized space.
        tightness (float): box area is enlarged by tightness * 100 % around the
                           region center (e.g. 0.1 -> 10% larger area). 0 keeps the
                           tight bounding box.

    Returns:
        list of [x_low, y_low, x_high, y_high] boxes, one per connected region.
    """
    mask = load_mask(mask_path, image_size)
    mask = (mask > 0).astype(np.uint8)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    scale = math.sqrt(1.0 + max(tightness, 0.0))
    boxes = []
    for i in range(1, num_labels):
        x_low = int(stats[i, cv2.CC_STAT_LEFT])
        y_low = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        x_high = x_low + w
        y_high = y_low + h

        if tightness > 0:
            cx = x_low + w / 2.0
            cy = y_low + h / 2.0
            nw = w * scale
            nh = h * scale
            x_low = int(round(cx - nw / 2.0))
            y_low = int(round(cy - nh / 2.0))
            x_high = int(round(cx + nw / 2.0))
            y_high = int(round(cy + nh / 2.0))

        x_low = max(0, min(x_low, image_size))
        y_low = max(0, min(y_low, image_size))
        x_high = max(0, min(x_high, image_size))
        y_high = max(0, min(y_high, image_size))
        boxes.append([x_low, y_low, x_high, y_high])

    return boxes


def main(tightness=0.1, image_size=256, output_path='box_coords.json'):
    result = {}

    otu2d_path = '/mnt/nvme0/home/utbt/KhoaVM/OTU-2D-Dataset/OTU_2D/'
    otu2d_json = '/mnt/nvme0/home/utbt/KhoaVM/OTU-2D-Dataset/OTU_2D_850-150-469.json'
    with open(otu2d_json, 'r') as f:
        otu2d_data = json.load(f)
    otu2d_entries = []
    for items in otu2d_data.values():
        for item in items:
            image_path = otu2d_path + str(item['image'])
            mask_path = otu2d_path + str(item['mask'])
            otu2d_entries.append({
                'image': image_path,
                'mask': mask_path,
                'boxes': extract_boxes_from_mask(mask_path, image_size, tightness),
            })
    result['OTU2D'] = otu2d_entries

    usova_root = '/mnt/nvme0/home/utbt/KhoaVM/USOVA3D_Dataset'
    annotator = 'ovary_r2'
    variant = 'binary'
    with open(os.path.join(usova_root, 'split.json'), 'r', encoding='utf-8') as f:
        usova_split = json.load(f)
    usova_entries = []
    for split_name in ['train', 'val', 'test']:
        for data in usova_split['split'][split_name].values():
            images = [os.path.join(usova_root, rel) for rel in data['images']]
            masks = [os.path.join(usova_root, rel) for rel in data['labels'][annotator][variant]]
            for image_path, mask_path in zip(images, masks):
                usova_entries.append({
                    'image': image_path,
                    'mask': mask_path,
                    'boxes': extract_boxes_from_mask(mask_path, image_size, tightness),
                })
    result['USOVA3D'] = usova_entries

    ovatus_json = '/mnt/nvme0/home/utbt/KhoaVM/OvaTUS/annotations_split.json'
    base = '/mnt/nvme0/home/utbt/KhoaVM/'
    with open(ovatus_json, 'r', encoding='utf-8') as f:
        ovatus_data = json.load(f)
    ovatus_entries = []
    for item in ovatus_data:
        image_path = os.path.join(base, item.get('file_path_img', ''))
        mask_path = os.path.join(base, item.get('file_path_ann', ''))
        ovatus_entries.append({
            'image': image_path,
            'mask': mask_path,
            'boxes': extract_boxes_from_mask(mask_path, image_size, tightness),
        })
    result['OVATUS'] = ovatus_entries

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    for name, entries in result.items():
        n_boxes = sum(len(e['boxes']) for e in entries)
        n_empty = sum(1 for e in entries if not e['boxes'])
        print(f"{name}: {len(entries)} masks, {n_boxes} boxes, {n_empty} empty masks")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract SAM bounding boxes from ground-truth masks')
    parser.add_argument('--tightness', type=float, default=0.1, help='box area enlargement ratio (0.1 = 10% larger area)')
    parser.add_argument('--image_size', type=int, default=256, help='resized mask resolution (coordinates are in this space)')
    parser.add_argument('--output', type=str, default='box_coords.json', help='output json path')
    args = parser.parse_args()
    main(tightness=args.tightness, image_size=args.image_size, output_path=args.output)

# Run command (from the y_DG folder):
#   python generate_bounding_box.py
#   python generate_bounding_box.py --tightness 0 --image_size 512
#   python generate_bounding_box.py --tightness 0.2 --output box_coords_loose.json
