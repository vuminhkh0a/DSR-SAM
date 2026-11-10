"""Project paths and YOLOv8 run configuration.

All model/loss/training-loop code lives in the vendored
``yolo/ultralytics`` tree.  This module only holds the benchmark choices:
3 sources, configurable sizes (default n,s), 3 seeds, image size 256, and the
reference repo's default optimizer/hyperparameters
(``ultralytics/yolo/configs/default.yaml``).
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
YOLO_ROOT = PROJECT_ROOT / 'yolo'
WEIGHTS_DIR = PROJECT_ROOT / 'weights' / 'yolo'
RUNS_DIR = WEIGHTS_DIR / 'runs'
RUNS_SMOKE_DIR = WEIGHTS_DIR / 'runs_smoke'
DATASETS_DIR = YOLO_ROOT / 'datasets'
RESULTS_PATH = PROJECT_ROOT / 'logs_results' / 'results_yolo.json'
BOX_FILE_TEMPLATE = str(PROJECT_ROOT / 'box_coords_yolov8_{size}.json')
BOX_COORDS_PATH = PROJECT_ROOT / 'box_coords.json'

# DG domain name -> top-level key in box_coords.json.
DOMAINS = ['OTU', 'OVATUS', 'USOVA']
DOMAIN_TO_BOX_KEY = {'OTU': 'OTU2D', 'OVATUS': 'OVATUS', 'USOVA': 'USOVA3D'}
# box_coords.json key order, preserved in the YOLO box files.
BOX_KEY_ORDER = ['OTU2D', 'USOVA3D', 'OVATUS']

# Sizes accepted by the pipeline and the default full-run set.  The code is
# size-agnostic; SIZES is just what `python -m yolo.main` runs when --sizes is
# not given.  Use --sizes n,s,m,l,x to run a different subset.
ALL_SIZES = ['n', 's', 'm', 'l', 'x']
SIZES = ['n', 's']
SEEDS = [42, 0, 100]
YOLO_VERSION = 'v8'

PRETRAINED_URL_TEMPLATE = (
    'https://github.com/ultralytics/assets/releases/download/v0.0.0/'
    'yolov8{size}.pt')

# The benchmark/SAM input is 256x256 and box_coords.json coordinates are in
# that space.  YOLO trains and predicts at 256 so no coordinate remapping is
# needed; predictions are still scaled back to 256 explicitly.
IMAGE_SIZE = 256
BATCH_SIZE = 16
WORKERS = 4

# Inference for box prompts.  Keep all detections (sorted by confidence) so
# multi-region masks are represented; DG loaders consume the first box.
PREDICT_CONF = 0.25
PREDICT_IOU = 0.7
PREDICT_MAX_DET = 10

# Reference repo default.yaml / docs/config.md values.
# NOTE: docs/config.md lists an older ``box: 0.05``; the actual training code
# default.yaml used by the reference commit is ``box: 7.5``, which is what the
# repo's train loop applies.  We use the code defaults.
TRAIN_HYP = {
    'optimizer': 'SGD',
    'lr0': 0.01,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    'nbs': 64,
    'patience': 50,          # reference default.yaml; early stopping patched in
    'close_mosaic': 10,
    'cos_lr': False,
    'single_cls': True,
    'rect': False,
    'cache': False,
    'plots': False,
    'val': True,
    'save': True,
    'deterministic': True,
    'verbose': False,
}

EPOCHS = 100


def pretrained_path(size: str) -> Path:
    return WEIGHTS_DIR / f'yolov8{size}.pt'


def ensure_pretrained(size: str, verbose=True) -> Path:
    """Return the official YOLOv8 checkpoint, downloading it if missing.

    The download is a plain stdlib ``urlretrieve`` (no tqdm), writes to a
    ``.part`` file, and only replaces the final path after a plausible file
    size is received.
    """
    size = str(size).lower()
    path = pretrained_path(size)
    if path.exists() and path.stat().st_size > 1_000_000:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    url = PRETRAINED_URL_TEMPLATE.format(size=size)
    tmp = path.with_name(path.name + '.part')
    if verbose:
        print(f'[yolo] pretrained checkpoint missing; downloading {url}')
    try:
        urllib.request.urlretrieve(url, tmp)
        if not tmp.exists() or tmp.stat().st_size <= 1_000_000:
            raise RuntimeError('downloaded checkpoint is unexpectedly small')
        tmp.replace(path)
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f'Could not download the YOLOv8{size} checkpoint from {url}: '
            f'{exc}') from exc
    if verbose:
        print(f'[yolo] saved {path} ({path.stat().st_size / 1e6:.1f} MB)')
    return path


def run_name(size: str, seed: int, source: str) -> str:
    return f'yolov8{size}_seed{seed}_s_{source}'


def runs_root(smoke: bool = False) -> Path:
    return RUNS_SMOKE_DIR if smoke else RUNS_DIR


def run_dir(size: str, seed: int, source: str, smoke: bool = False) -> Path:
    return runs_root(smoke) / run_name(size, seed, source)


def best_path(size: str, seed: int, source: str, smoke: bool = False) -> Path:
    return run_dir(size, seed, source, smoke) / 'weights' / 'best.pt'


def last_path(size: str, seed: int, source: str, smoke: bool = False) -> Path:
    return run_dir(size, seed, source, smoke) / 'weights' / 'last.pt'


def box_file(size: str, smoke: bool = False) -> str:
    suffix = f'{size}_smoke' if smoke else size
    return BOX_FILE_TEMPLATE.format(size=suffix)


def results_path(smoke: bool = False) -> Path:
    if smoke:
        return PROJECT_ROOT / 'logs_results' / 'results_yolo_smoke.json'
    return RESULTS_PATH
