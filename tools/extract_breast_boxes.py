"""Extract SAM-style bounding boxes for the three breast-ultrasound datasets
and merge them into the existing ``box_coords.json`` without touching the
coordinates that are already there.

Output format is identical to the ovarian datasets produced by
``utils/generate_bounding_box.py``::

    {
      "OTU2D":   [{"image": ..., "mask": ..., "boxes": [[x0, y0, x1, y1], ...]}, ...],
      "USOVA3D": [...],
      "OVATUS":  [...],
      "BUS_UC":   [...],   # added by this script
      "BUSBRA":   [...],   # added by this script
      "BUSI_WHU": [...]    # added by this script
    }

* Coordinates live in the 256x256 resized mask space.
* ``tightness=0.1`` enlarges each box area by 10% around its centre
  (linear scale ``sqrt(1.1)``), exactly like the ovarian datasets.
* One box per 8-connected mask component; blank masks get ``"boxes": []``.

Write policy
------------
The file is loaded, the three breast keys are (re)generated, every other
top-level key is left exactly as it was, and the result is written atomically
(temp file + ``os.replace``).  A ``box_coords.json.bak`` backup of the previous
file is kept.  The script asserts that the non-breast keys are unchanged before
replacing the file.

Usage (from ``y_DG``)::

    python tools/extract_breast_boxes.py
    python tools/extract_breast_boxes.py --tightness 0.1 --image-size 256
    python tools/extract_breast_boxes.py --output box_coords.json --no-backup
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

Y_DG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if Y_DG not in sys.path:
    sys.path.insert(0, Y_DG)

from utils.generate_bounding_box import extract_boxes_from_mask  # noqa: E402

KHOAVM = '/mnt/nvme0/home/utbt/KhoaVM'
MANIFEST_DIR = os.path.join(KHOAVM, 'splits')
BREAST_KEYS = ('BUS_UC', 'BUSBRA', 'BUSI_WHU')
DEFAULT_OUTPUT = os.path.join(Y_DG, 'box_coords.json')


def _manifest_entries(name, image_size, tightness):
    """Build the box entries for one breast dataset from its split manifest.

    Entries are emitted in a deterministic order: train, val, test, and then
    the order stored in the manifest.  Every image gets an entry; masks with no
    foreground produce ``boxes: []``.
    """
    manifest_path = os.path.join(MANIFEST_DIR, f'{name}.json')
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)['split']

    entries = []
    for split in ('train', 'val', 'test'):
        for rec in manifest[split]:
            image_path = os.path.join(KHOAVM, rec['image'])
            mask_path = os.path.join(KHOAVM, rec['mask'])
            if not os.path.exists(image_path):
                raise FileNotFoundError(f'missing image for {name}: {image_path}')
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f'missing mask for {name}: {mask_path}')
            entries.append({
                'image': image_path,
                'mask': mask_path,
                'boxes': extract_boxes_from_mask(mask_path, image_size, tightness),
            })
    return entries


def _non_breast(result):
    return {k: v for k, v in result.items() if k not in BREAST_KEYS}


def merge_into_box_coords(output_path=DEFAULT_OUTPUT, image_size=256,
                          tightness=0.1, datasets=BREAST_KEYS, backup=True):
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
    else:
        result = {}

    before = _non_breast(result)  # snapshot of the coordinates to preserve

    summary = {}
    for name in datasets:
        entries = _manifest_entries(name, image_size, tightness)
        result[name] = entries
        summary[name] = {
            'masks': len(entries),
            'boxes': sum(len(e['boxes']) for e in entries),
            'empty': sum(1 for e in entries if not e['boxes']),
        }

    # Hard guarantee: the pre-existing datasets were not modified.
    after = _non_breast(result)
    if before != after:
        raise RuntimeError('internal error: existing box coordinates changed')
    if before:
        result_old = {k: result[k] for k in before}
        assert result_old == before

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    old_mode = os.stat(output_path).st_mode & 0o777 if os.path.exists(output_path) else 0o664

    if backup and os.path.exists(output_path):
        shutil.copy2(output_path, output_path + '.bak')

    fd, tmp_path = tempfile.mkstemp(prefix='.box_coords.', suffix='.tmp', dir=out_dir)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_path, old_mode)  # keep the original permissions
        os.replace(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Post-write check: reload and compare the preserved keys.
    with open(output_path, 'r', encoding='utf-8') as f:
        reloaded = json.load(f)
    if _non_breast(reloaded) != before:
        raise RuntimeError('post-write check failed: existing coordinates changed')
    for name in datasets:
        if name not in reloaded or len(reloaded[name]) != summary[name]['masks']:
            raise RuntimeError(f'post-write check failed for {name}')

    print(f'wrote {output_path}')
    for name in datasets:
        s = summary[name]
        print(f"  {name}: {s['masks']} masks, {s['boxes']} boxes, {s['empty']} empty masks")
    if before:
        print(f"  preserved existing keys unchanged: {', '.join(before)}")
    if backup and os.path.exists(output_path + '.bak'):
        print(f"  backup: {output_path}.bak")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Extract 256x256 SAM boxes for BUS_UC/BUSBRA/BUSI_WHU and '
                    'merge them into box_coords.json without touching old coords.')
    parser.add_argument('--output', default=DEFAULT_OUTPUT,
                        help='box_coords.json path (default: y_DG/box_coords.json)')
    parser.add_argument('--tightness', type=float, default=0.1,
                        help='box area enlargement (0.1 = 10%% larger area, same as ovarian)')
    parser.add_argument('--image-size', type=int, default=256,
                        help='resized mask resolution; coordinates are in this space')
    parser.add_argument('--datasets', nargs='+', default=list(BREAST_KEYS),
                        choices=list(BREAST_KEYS),
                        help='subset of breast datasets to (re)generate')
    parser.add_argument('--no-backup', action='store_true',
                        help='do not create box_coords.json.bak')
    args = parser.parse_args()

    merge_into_box_coords(
        output_path=args.output,
        image_size=args.image_size,
        tightness=args.tightness,
        datasets=args.datasets,
        backup=not args.no_backup,
    )


if __name__ == '__main__':
    main()
