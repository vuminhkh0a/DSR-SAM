# YOLOv8 prompt-box benchmark

This folder vendors the reference YOLOv8 repository
[`autogyro/yolo-V8`](https://github.com/autogyro/yolo-V8) at commit
`cc3c774` (Ultralytics `8.0.8`) and adds only thin benchmark glue.

* `yolo/ultralytics/` — copied reference code (models, loss, trainer,
  dataloaders, predictor, validator).
* `yolo/data.py` — builds YOLO-format datasets from the existing DG splits and
  `box_coords.json` (read-only).
* `yolo/train.py` — one training run (source, size, seed).
* `yolo/evaluate.py` — detection metrics on all three test domains.
* `yolo/predict.py` — writes `box_coords_yolov8_<size>.json`.
* `yolo/main.py` — runs configured sizes (default n,s) x 3 seeds x 3 sources.

The reference hyperparameters are used (`optimizer=SGD`, `lr0=0.01`,
`lrf=0.01`, `momentum=0.937`, `weight_decay=0.0005`, `warmup_epochs=3`,
`box=7.5`, `cls=0.5`, `dfl=1.5`, `nbs=64`, `patience=50`, `close_mosaic=10`).
Image size is **256**, matching the coordinate space of `box_coords.json` and
the SAM inputs.

## Environment

The reference commit was written for Python <= 3.11. On this machine the
following extra packages were installed into the existing `khoa-env`:

```bash
pip install GitPython tensorboard
pip install "setuptools==80.9.0"   # pkg_resources was removed in setuptools 82
```

`torch`, `torchvision`, `opencv`, `albumentations`, `hydra-core`, `omegaconf`,
`pandas`, `matplotlib` and `seaborn` were already present.

## Commands

```bash
# Smoke test: first configured size (n), seed 42, all 3 sources, 1 epoch.
# Writes results_yolo_smoke.json and box_coords_yolov8_n_smoke.json and does
# NOT overwrite the full-run files.
python -m yolo.main --smoke

# Full matrix: default sizes n,s x 3 seeds x 3 sources = 18 trainings.
python -m yolo.main

# One configuration
python -m yolo.main --sizes n --seeds 42 --sources OTU

# Run the older sizes explicitly (any subset of n,s,m,l,x)
python -m yolo.main --sizes m,l,x

# Re-evaluate existing checkpoints without training
python -m yolo.main --skip-train

# Train only (no evaluation / no box files)
python -m yolo.main --skip-eval --skip-boxes

# Rebuild the copied datasets from box_coords.json
python -m yolo.main --force-data
```

`run.sh` also has a `nohup` line for the full YOLO run (and a commented one
for the smoke run). Nothing is launched by this documentation alone.

All tqdm progress bars from the reference code are disabled at import time by
`yolo/__init__.py`, so logs only contain epoch summaries and metrics.

## Outputs

| Path | Contents |
| --- | --- |
| `weights/yolo/yolov8<n,s,m,l,x>.pt` | Pretrained COCO checkpoints; n/s are auto-downloaded on first use. |
| `weights/yolo/runs/yolov8<size>_seed<seed>_s_<source>/weights/best.pt` | Full-run checkpoints. |
| `weights/yolo/runs_smoke/...` | Smoke-run checkpoints (kept separate so a 1-epoch smoke model can never block a full run). |
| `results_yolo.json` | Full detection results (or `results_yolo_smoke.json`). |
| `box_coords_yolov8_{n,s,m,l,x}.json` | Full seed-42 prompt boxes, one file per size (or `..._n_smoke.json`). |
| `yolo/datasets/<OTU2D|OVATUS|USOVA3D>/` | Copied images + YOLO labels + `data*.yaml`. |

No existing weight file and **no `box_coords.json` content is changed**.

### `results_yolo.json`

A list of records, one per (size, seed, source, target):

```json
{
  "name": "yolov8m_seed42_s_OTU_t_OVATUS_best",
  "yolo_version": "v8",
  "size": "m",
  "seed": 42,
  "source": "OTU",
  "target": "OVATUS",
  "weight": "best",
  "imgsz": 256,
  "results": {
    "map50": 0.0,
    "map50_95": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "box_iou": 0.0
  }
}
```

`map50`, `map50_95`, `precision` and `recall` come from the reference
`DetectionValidator` (0–100 scale). `box_iou` is the mean over GT boxes of the
best IoU with any predicted box (0–100 scale).

### `box_coords_yolov8_<size>.json`

Seed 42 only. The file is source-keyed so the same target test image can have
different prompts for `s_OTU` and `s_USOVA`:

```json
{
  "OTU": {
    "OTU2D":   [{"image": "...", "mask": "...", "boxes": [[x_min, y_min, x_max, y_max]]}, ...],
    "USOVA3D": [...],
    "OVATUS":  [...]
  },
  "OVATUS": { ... },
  "USOVA":  { ... }
}
```

* Each entry keeps exactly `{image, mask, boxes}`, with `x_min y_min x_max
  y_max` in 256-space, in `box_coords.json` order.
* USOVA blank-GT slices are absent because `utils.data.get_datasets` excludes
  them (same samples as every DG method).
* All splits are predicted for completeness. The boxes required by a DG run
  (source train+val, target test) are therefore included.
* Boxes are sorted by confidence; DG loaders use the first box. Images with no
  detection get `boxes: []`, and the DG loader falls back to the full image.

## DG integration

The box-prompt DG methods now accept:

```python
CONFIG = {
    'prompt_method': 'gt',       # 'gt' -> box_coords.json
                                 # 'yolov8' -> box_coords_yolov8_<prompt_yolo_size>.json
    'prompt_yolo_size': 'm',     # n / s / m / l / x (default m)
}
```

Implemented in:
`DG/dsr_sam`, `DG/dsr_sam2`, `DG/sr_sam`, `DG/masam`, `DG/medsam`,
`DG/sammed2d`, `DG/ultrasam`.

Shared loading lives in `utils/prompts.py`; the duplicated `_load_box_map`
copies were removed from those seven methods. Training-source methods select
`yolo_data[source]`; test-only methods (`prompt_source=None`) select the
target's own section, i.e. the model trained on that target domain.

To run a DG method with YOLO prompts:

```bash
python -m DG.dsr_sam2.main   # edit CONFIG prompt_method='yolov8', prompt_yolo_size='m'
```

## Patches to the vendored reference code

These are minimal compatibility/behavior fixes; see `PATCHES.md` for details.

1. **Early stopping** using the repo's own `patience=50` (the reference
   commit defines `patience` but leaves the loop termination as a TODO).
2. Pillow >= 10: `FreeTypeFont.getsize` -> `getbbox` fallback in plotting.
3. Albumentations >= 2.0: `ImageCompression(quality_lower=...)` ->
   `quality_range=(75, 100)`.
4. Predictor attaches `Results.path` so streamed predictions can be mapped
   back to source images (later Ultralytics versions include this natively).

## Notes / decisions

* `box_coords.json` is opened read-only. The YOLO labels are derived from it;
  `generate_bounding_box.py` is not run.
* `docs/config.md` lists an old `box: 0.05`, while the actual reference
  `default.yaml` and training code use `box: 7.5`; the code default is used.
* Full training is `epochs=100`, batch 16, workers 4, imgsz 256. Change via
  CLI flags or `yolo/config.py`.
* `yolo/datasets/` is a disposable copy and is git-ignored.
