"""YOLOv8 benchmark entry point.

Full run (default):
    configured SIZES (default n,s) x 3 seeds (42,0,100) x 3 sources
    (OTU,OVATUS,USOVA) -> 18 trained models for n,s; each evaluated on all 3
    test domains.  seed=42 runs also write box_coords_yolov8_<size>.json.

Smoke run (all sources, 1 epoch, first configured size, seed 42):
    python -m yolo.main --smoke
    -> logs_results/results_yolo_smoke.json and box_coords_yolov8_n_smoke.json

Examples:
    python -m yolo.main --smoke
    python -m yolo.main --sizes n,s --seeds 42 --sources OTU
    python -m yolo.main --sizes m,l,x          # run the older sizes explicitly
    python -m yolo.main --skip-train          # evaluate existing checkpoints
    python -m yolo.main --skip-eval --skip-boxes   # train only
"""
from __future__ import annotations

import argparse
import sys
import time

import torch

from yolo import config as C
from yolo.data import build_all_datasets
from yolo.evaluate import evaluate_source
from yolo.predict import write_size_box_file
from yolo.train import train_source
from utils.seed import set_seed


def _parse_list(text, allowed, name):
    items = [x.strip() for x in str(text).split(',') if x.strip()]
    bad = [x for x in items if x not in allowed]
    if bad:
        raise SystemExit(f'Invalid {name}: {bad}; allowed: {allowed}')
    return items


def main(argv=None):
    ap = argparse.ArgumentParser(description='Train/test vendored YOLOv8 on the DG benchmark')
    ap.add_argument('--sizes', default=None,
                    help=','.join(C.SIZES) + ' by default')
    ap.add_argument('--seeds', default=None,
                    help=','.join(str(s) for s in C.SEEDS) + ' by default')
    ap.add_argument('--sources', default=','.join(C.DOMAINS))
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch', type=int, default=None)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--device', default=None,
                    help='physical GPU id / cpu (default: CUDA_DEVICE from .env)')
    ap.add_argument('--smoke', action='store_true',
                    help='first configured size (default n), seed 42, all sources, '
                         '1 epoch, *_smoke outputs')
    ap.add_argument('--force-data', action='store_true',
                    help='rebuild the copied YOLO datasets')
    ap.add_argument('--force-train', action='store_true',
                    help='retrain even if best.pt exists')
    ap.add_argument('--force-boxes', action='store_true',
                    help='recompute source sections even if already present')
    ap.add_argument('--skip-data', action='store_true')
    ap.add_argument('--skip-train', action='store_true')
    ap.add_argument('--skip-eval', action='store_true')
    ap.add_argument('--skip-boxes', action='store_true')
    args = ap.parse_args(argv)

    smoke = args.smoke
    sizes_text = args.sizes if args.sizes is not None else ','.join(C.SIZES)
    seeds_text = (args.seeds if args.seeds is not None
                  else ','.join(str(s) for s in C.SEEDS))
    sizes = _parse_list(sizes_text, C.ALL_SIZES, 'sizes')
    seeds = [int(s) for s in _parse_list(seeds_text, [str(x) for x in C.SEEDS], 'seeds')]
    sources = _parse_list(args.sources, C.DOMAINS, 'sources')
    if smoke:
        # A smoke run only needs the first size/seed unless overridden.
        if args.sizes is None:
            sizes = [C.SIZES[0]]
        if args.seeds is None:
            seeds = [42]
        if args.epochs is None:
            args.epochs = 1
    epochs = args.epochs if args.epochs is not None else C.EPOCHS

    print(f'[yolo] version={C.YOLO_VERSION} sizes={sizes} seeds={seeds} '
          f'sources={sources} epochs={epochs} smoke={smoke}')
    if not args.skip_data:
        build_all_datasets(force=args.force_data)

    start = time.time()
    total = len(sizes) * len(seeds) * len(sources)
    done = 0
    for size in sizes:
        for seed in seeds:
            for source in sources:
                done += 1
                print(f'\n{"=" * 70}')
                print(f'[yolo] RUN {done}/{total} | size={size} seed={seed} '
                      f'source={source}')
                print(f'{"=" * 70}')
                set_seed(seed)

                best = C.best_path(size, seed, source, smoke=smoke)
                if not args.skip_train:
                    best = train_source(
                        size, seed, source, epochs=epochs,
                        force=args.force_train,
                        batch_size=args.batch, workers=args.workers,
                        device=args.device, smoke=smoke,
                    )
                elif not best.exists():
                    print(f'[yolo] skip (no checkpoint): {best}')
                    continue

                if not args.skip_eval:
                    evaluate_source(
                        size, seed, source, targets=C.DOMAINS,
                        smoke=smoke, device=args.device,
                        batch_size=args.batch, workers=args.workers,
                    )

                # Canonical box files use seed 42 only.
                if (not args.skip_boxes and seed == 42 and best.exists()):
                    write_size_box_file(
                        size, seed=seed, smoke=smoke,
                        force=args.force_boxes, sources=[source],
                        device=args.device,
                    )

                torch.cuda.empty_cache()

    elapsed = time.time() - start
    print(f'\n[yolo] finished {done}/{total} runs in {elapsed / 3600:.3f} h')
    return 0


if __name__ == '__main__':
    sys.exit(main())
