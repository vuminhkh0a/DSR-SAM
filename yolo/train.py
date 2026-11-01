"""Train one vendored YOLOv8 model on one DG source domain.

The optimizer, loss, dataloader and training loop are the reference repo's
(the copied ``yolo/ultralytics`` code); this module only selects the source
split, seed, size and output directory.
"""
from __future__ import annotations

import os

import torch

from yolo import YOLO
from yolo import config as C
from yolo.data import build_domain_dataset, data_yaml
from utils.env import get_device, get_physical_gpu_index
from utils.seed import set_seed


def yolo_device():
    """Return a device string accepted by the reference select_device()."""
    dev = str(get_device())
    if dev.startswith('cuda'):
        idx = get_physical_gpu_index()
        return str(idx) if idx is not None else '0'
    return dev


def train_source(size, seed, source, epochs=None, force=False,
                 batch_size=None, workers=None, device=None, smoke=False):
    """Train and return the path to the best checkpoint."""
    size = str(size).lower()
    seed = int(seed)
    source = str(source).upper()
    if size not in C.ALL_SIZES:
        raise ValueError(f'size must be one of {C.ALL_SIZES}, got {size!r}')
    if source not in C.DOMAINS:
        raise ValueError(f'source must be one of {C.DOMAINS}, got {source!r}')

    best = C.best_path(size, seed, source, smoke=smoke)
    if best.exists() and not force:
        print(f'[yolo] skip training (best exists): {best}')
        return best

    weights = C.ensure_pretrained(size)

    build_domain_dataset(source)
    epochs = int(epochs if epochs is not None else C.EPOCHS)
    device = str(device) if device is not None else yolo_device()
    batch_size = int(batch_size if batch_size is not None else C.BATCH_SIZE)
    workers = int(workers if workers is not None else C.WORKERS)

    set_seed(seed)
    print(f'\n{"=" * 70}')
    print(f'[yolo] TRAIN size={size} seed={seed} source={source}')
    print(f'[yolo] weights={weights}')
    print(f'[yolo] data={data_yaml(source)}')
    print(f'[yolo] epochs={epochs} imgsz={C.IMAGE_SIZE} batch={batch_size} '
          f'device={device}')
    print(f'{"=" * 70}\n')

    model = YOLO(str(weights))
    model.train(
        data=str(data_yaml(source)),
        epochs=epochs,
        imgsz=C.IMAGE_SIZE,
        batch=batch_size,
        device=device,
        workers=workers,
        project=str(C.runs_root(smoke)),
        name=C.run_name(size, seed, source),
        exist_ok=True,
        seed=seed,
        **C.TRAIN_HYP,
    )

    best = C.best_path(size, seed, source, smoke=smoke)
    if not best.exists():
        # The reference trainer writes <project>/<name>/weights/best.pt.
        fallback = C.run_dir(size, seed, source, smoke) / 'weights' / 'best.pt'
        if fallback.exists():
            best = fallback
    print(f'[yolo] best checkpoint: {best}')
    torch.cuda.empty_cache()
    return best


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', default=C.SIZES[0], choices=C.ALL_SIZES)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--source', default='OTU', choices=C.DOMAINS)
    ap.add_argument('--epochs', type=int, default=C.EPOCHS)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    train_source(args.size, args.seed, args.source, epochs=args.epochs,
                 force=args.force)
