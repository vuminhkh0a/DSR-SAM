"""Detection metrics for a trained YOLOv8 model on a target test split.

Metrics are detection-only (the user's choice):
    mAP50, mAP50-95, precision, recall (reference DetectionValidator), and
    mean best-match box IoU (top-1 prediction vs GT boxes).

Results are appended/updated in ``logs_results/results_yolo.json`` (or
``logs_results/results_yolo_smoke.json`` for smoke runs) as a list of records with
``yolo_version``, ``size``, ``seed``, ``source``, ``target`` and ``results``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from yolo import YOLO
from yolo import config as C
from yolo.data import data_test_yaml, split_entries
from yolo.predict import predict_boxes
from yolo.train import yolo_device


def _norm(path):
    return os.path.normpath(os.path.abspath(str(path)))


def _box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate_detection(model_path, target, batch_size=None, workers=None,
                       device=None):
    """Run the reference DetectionValidator and return 0-100 metrics."""
    from ultralytics.yolo.configs import get_config
    from ultralytics.yolo.utils import DEFAULT_CONFIG
    from ultralytics.yolo.v8.detect.val import DetectionValidator

    device = str(device) if device is not None else yolo_device()
    batch_size = int(batch_size if batch_size is not None else C.BATCH_SIZE)
    workers = int(workers if workers is not None else C.WORKERS)
    run = Path(model_path).parent.parent.name
    overrides = {
        'task': 'detect',
        'mode': 'val',
        'model': str(model_path),
        'data': str(data_test_yaml(target)),
        'imgsz': C.IMAGE_SIZE,
        'batch': batch_size,
        'device': device,
        'workers': workers,
        'conf': 0.001,
        'iou': 0.6,
        'max_det': 300,
        'half': False,
        'plots': False,
        'save_json': False,
        'verbose': False,
        'single_cls': True,
        'project': str(C.WEIGHTS_DIR / 'val'),
        'name': f'{run}_t_{target}',
        'exist_ok': True,
    }
    args = get_config(DEFAULT_CONFIG, overrides)
    args.task = 'detect'
    args.mode = 'val'
    validator = DetectionValidator(args=args)
    stats = validator(model=str(model_path))
    return {
        'map50': float(stats['metrics/mAP50(B)']) * 100.0,
        'map50_95': float(stats['metrics/mAP50-95(B)']) * 100.0,
        'precision': float(stats['metrics/precision(B)']) * 100.0,
        'recall': float(stats['metrics/recall(B)']) * 100.0,
    }


def mean_box_iou(model_path, target, device=None):
    """Mean over GT boxes of the best IoU with any predicted box, in percent."""
    entries = split_entries(target, 'test')
    model = YOLO(str(model_path))
    preds = predict_boxes(model, [e['image'] for e in entries], device=device)
    per_image = []
    for e in entries:
        gt = e.get('boxes') or []
        if not gt:
            continue
        pr = preds.get(_norm(e['image']), [])
        if not pr:
            per_image.append(0.0)
            continue
        best_per_gt = []
        for g in gt:
            best_per_gt.append(max(_box_iou(g, p) for p in pr))
        per_image.append(float(np.mean(best_per_gt)))
    if not per_image:
        return 0.0
    return float(np.mean(per_image)) * 100.0


def _order(entry):
    size_i = C.ALL_SIZES.index(entry['size']) if entry['size'] in C.ALL_SIZES else 99
    seed_i = C.SEEDS.index(entry['seed']) if entry['seed'] in C.SEEDS else 99
    source_i = C.DOMAINS.index(entry['source']) if entry['source'] in C.DOMAINS else 99
    target_i = C.DOMAINS.index(entry['target']) if entry['target'] in C.DOMAINS else 99
    return (size_i, seed_i, source_i, target_i)


def save_result(entry, smoke=False):
    path = C.results_path(smoke=smoke)
    data = []
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(data, list):
                data = []
        except json.JSONDecodeError:
            data = []
    data = [e for e in data if e.get('name') != entry['name']]
    data.append(entry)
    data.sort(key=_order)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    return path


def evaluate_run(size, seed, source, target, model_path, smoke=False,
                 device=None, batch_size=None, workers=None):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    print(f'\n[yolo] EVAL size={size} seed={seed} source={source} '
          f'target={target}')
    metrics = evaluate_detection(model_path, target, batch_size=batch_size,
                                 workers=workers, device=device)
    metrics['box_iou'] = mean_box_iou(model_path, target, device=device)
    metrics = {k: round(float(v), 2) for k, v in metrics.items()}
    name = f'yolov8{size}_seed{seed}_s_{source}_t_{target}_best'
    entry = {
        'name': name,
        'yolo_version': C.YOLO_VERSION,
        'size': size,
        'seed': int(seed),
        'source': source,
        'target': target,
        'weight': 'best',
        'imgsz': C.IMAGE_SIZE,
        'results': metrics,
    }
    path = save_result(entry, smoke=smoke)
    print(f'[yolo] {name}: {metrics} -> {path}')
    return entry


def evaluate_source(size, seed, source, targets=None, smoke=False,
                    device=None, batch_size=None, workers=None):
    targets = list(targets or C.DOMAINS)
    out = []
    for target in targets:
        out.append(evaluate_run(
            size, seed, source, target, C.best_path(size, seed, source, smoke=smoke),
            smoke=smoke, device=device, batch_size=batch_size, workers=workers,
        ))
    return out
