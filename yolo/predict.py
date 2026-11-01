"""Run the vendored YOLOv8 predictor and write box_coords_yolov8_<size>.json.

For every source model (seed 42) the file contains::

    {
      "OTU":    {"OTU2D": [...], "USOVA3D": [...], "OVATUS": [...]},
      "OVATUS": {...},
      "USOVA":  {...}
    }

Each list entry keeps exactly ``{"image", "mask", "boxes"}`` and the sample
order of ``box_coords.json`` (blank USOVA GT entries are absent because the DG
splits exclude them).  All splits are predicted for completeness; the boxes
needed by a DG run (source train+val, target test) are therefore included.

Predicted boxes are top-left/bottom-right ``[x_min, y_min, x_max, y_max]`` in
the 256x256 space used by ``box_coords.json``.  Multiple detections are kept
in descending confidence order; DG loaders use the first one.  Images with no
detection get ``boxes: []`` (DG loaders fall back to the full image).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from yolo import YOLO
from yolo import config as C
from yolo.data import ordered_entries
from yolo.train import yolo_device

_BOX_KEY_TO_DOMAIN = {v: k for k, v in C.DOMAIN_TO_BOX_KEY.items()}


def _norm(path):
    return os.path.normpath(os.path.abspath(str(path)))


def predict_boxes(model, image_paths, device=None):
    """Return {normalized image path: [[x1,y1,x2,y2], ...]} at 256-space."""
    image_paths = list(image_paths)
    device = str(device) if device is not None else yolo_device()
    preds = {}
    # This reference commit's predictor treats a Python list as a list of
    # PIL/numpy arrays. A *.txt file is treated as a list of image paths, which
    # is what we need. The file is temporary and only contains absolute paths.
    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.txt', prefix='yolo_paths_', dir='/tmp/opencode',
        delete=False, encoding='utf-8')
    try:
        for path in image_paths:
            tmp.write(str(path) + '\n')
        tmp.close()
        results = model.predict(
            source=tmp.name,
            stream=True,
            imgsz=C.IMAGE_SIZE,
            conf=C.PREDICT_CONF,
            iou=C.PREDICT_IOU,
            max_det=C.PREDICT_MAX_DET,
            device=device,
            verbose=False,
            save=False,
            augment=False,
            agnostic_nms=True,
        )
    except Exception:
        tmp.close()
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        raise
    try:
        for batch_results in results:
            for result in batch_results:
                path = _norm(result.path)
                boxes = result.boxes
                out = []
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.detach().cpu().numpy()
                    conf = boxes.conf.detach().cpu().numpy()
                    h, w = result.orig_shape
                    sx = C.IMAGE_SIZE / float(w)
                    sy = C.IMAGE_SIZE / float(h)
                    for idx in np.argsort(-conf):
                        x1, y1, x2, y2 = xyxy[idx].tolist()
                        x1 = int(round(min(max(x1 * sx, 0.0), C.IMAGE_SIZE)))
                        y1 = int(round(min(max(y1 * sy, 0.0), C.IMAGE_SIZE)))
                        x2 = int(round(min(max(x2 * sx, 0.0), C.IMAGE_SIZE)))
                        y2 = int(round(min(max(y2 * sy, 0.0), C.IMAGE_SIZE)))
                        if x2 > x1 and y2 > y1:
                            out.append([x1, y1, x2, y2])
                preds[path] = out
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
    # Any path the loader did not return (should not happen) becomes empty.
    for path in image_paths:
        preds.setdefault(_norm(path), [])
    return preds


def _predict_domain(model, domain, device):
    entries = ordered_entries(domain)
    preds = predict_boxes(model, [e['image'] for e in entries], device=device)
    out = []
    n_empty = 0
    for e in entries:
        boxes = preds.get(_norm(e['image']), [])
        if not boxes:
            n_empty += 1
        out.append({
            'image': e['image'],
            'mask': e['mask'],
            'boxes': boxes,
        })
    print(f'[yolo] {domain}: {len(out)} samples, {n_empty} without detection')
    return out


def predict_source_section(model_path, source, device=None):
    """Predict every domain with one source model, in box_coords order."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    device = str(device) if device is not None else yolo_device()
    model = YOLO(str(model_path))
    section = {}
    for box_key in C.BOX_KEY_ORDER:
        domain = _BOX_KEY_TO_DOMAIN[box_key]
        print(f'[yolo] predicting {source} model on {domain} ...')
        section[box_key] = _predict_domain(model, domain, device)
    return section


def write_size_box_file(size, seed=42, smoke=False, force=False,
                        sources=None, device=None):
    """Update the source-keyed box file for one model size.

    Existing source sections are preserved unless retrained/re-predicted, so a
    partially completed run can be resumed.
    """
    size = str(size).lower()
    output = Path(C.box_file(size, smoke=smoke))
    if output.exists() and not force:
        with open(output, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = {}

    sources = list(sources or C.DOMAINS)
    for source in sources:
        model_path = C.best_path(size, seed, source, smoke=smoke)
        if not model_path.exists():
            print(f'[yolo] no checkpoint for s_{source} ({model_path}); '
                  f'skipping box section')
            continue
        data[source] = predict_source_section(model_path, source, device=device)

    # Canonical key order: source domains first, then box_coords domain keys.
    ordered = {}
    for source in C.DOMAINS:
        if source in data:
            ordered[source] = {k: data[source][k] for k in C.BOX_KEY_ORDER
                               if k in data[source]}
    # Keep any unexpected sections at the end (defensive).
    for key, value in data.items():
        if key not in ordered:
            ordered[key] = value
    output.write_text(json.dumps(ordered, indent=2), encoding='utf-8')
    print(f'[yolo] wrote {output}')
    return output


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', default=C.SIZES[0], choices=C.ALL_SIZES)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--sources', default=','.join(C.DOMAINS))
    args = ap.parse_args()
    write_size_box_file(args.size, seed=args.seed, smoke=args.smoke,
                        force=args.force,
                        sources=args.sources.split(','))
