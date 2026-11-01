# Patches applied to the vendored reference code

The reference tree is `autogyro/yolo-V8` commit
`cc3c774bde86ffce694d202b7383da6cc1721c1b` (Ultralytics `8.0.8`). All files
under `yolo/ultralytics/` are copied verbatim except for the four patches
below. They are intentionally small and are required for this environment or
for the benchmark's `patience` requirement.

## 1. Early stopping (`yolo/ultralytics/yolo/engine/trainer.py`)

The reference `default.yaml` defines `patience: 50` with a TODO in the epoch
loop; no implementation exists in this commit. Added:

* `self.stop = False` and `self.epochs_no_improve = 0` in `BaseTrainer.__init__`.
* `prev_best_fitness = self.best_fitness` before validation.
* After `on_fit_epoch_end`, count epochs without a val-fitness improvement and
  `break` once `epochs_no_improve >= patience`.

This matches the semantics of later Ultralytics releases. The configured value
is the reference default `50`, not DSR-SAM's `20`.

## 2. Pillow >= 10 plotting compatibility
(`yolo/ultralytics/yolo/utils/plotting.py`)

`FreeTypeFont.getsize` was removed in Pillow 10. The code now uses `getsize`
when available and otherwise computes width/height from `getbbox`.

## 3. Albumentations >= 2.0 compatibility
(`yolo/ultralytics/yolo/data/augment.py`)

`A.ImageCompression(quality_lower=75)` was renamed. Changed to
`A.ImageCompression(quality_range=(75, 100))`. Without this, the optional
Albumentations pipeline fails to initialize (training still runs, but the
augmentations are silently disabled).

## 4. Predictor path metadata
(`yolo/ultralytics/yolo/engine/predictor.py`)

This early `Results` class has no `path` attribute, so streamed predictions
cannot be mapped back to their source image. Added
`results[i].path = p` inside `stream_inference`. Later Ultralytics versions
include this field natively.

---

No changes were made to `box_coords.json`, any existing weights, or any
existing DG result files.
