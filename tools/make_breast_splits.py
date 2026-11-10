"""Create leakage-safe, class-stratified train/val/test manifests for the
breast-ultrasound datasets used by y_DG.

Outputs (JSON only -- no image or mask is ever copied, moved or renamed):

    /mnt/nvme0/home/utbt/KhoaVM/splits/BUS_UC.json
    /mnt/nvme0/home/utbt/KhoaVM/splits/BUSBRA.json
    /mnt/nvme0/home/utbt/KhoaVM/splits/BUSI_WHU.json
    /mnt/nvme0/home/utbt/KhoaVM/splits/BUSI_WHU_labels.json

Splitting policy
----------------
BUS_UC   : image-level stratified 70/15/15 (the dataset ships no patient IDs).
BUSBRA   : patient-level (``Case``) stratified 70/15/15. All images of a case
           stay in one split, so there is no patient leakage. Cases are
           stratified by (class, images-per-case) so that both the patient
           and the image counts stay close to 70/15/15 and the class ratio is
           preserved in every split.
BUSI_WHU : existing train/valid/test folders are kept as-is and only described
           in a manifest.  Its original split is known to share base IDs
           between splits, so ``stats.leakage`` reports that explicitly.

Every split keeps the benign/malignant proportion of the full dataset
(within rounding of the integer counts).

Usage:
    python tools/make_breast_splits.py
    python tools/make_breast_splits.py --seed 42 --fractions 0.7 0.15 0.15
"""

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict

KHOAVM = "/mnt/nvme0/home/utbt/KhoaVM"
OUT_DIR = os.path.join(KHOAVM, "splits")

BUS_UC_ROOT = os.path.join(KHOAVM, "BUS_UC", "BUS_UC")
BUSBRA_ROOT = os.path.join(KHOAVM, "BUSBRA", "BUSBRA")
BUSI_WHU_ROOT = os.path.join(KHOAVM, "BUSI_WHU")

SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _largest_remainder(total, fractions):
    """Integer counts for ``fractions`` that sum exactly to ``total``."""
    raw = [total * f for f in fractions]
    counts = [int(x) for x in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in order[:remainder]:
        counts[i] += 1
    return counts


def _allocate_stratum(groups, fractions, rng):
    """Split one homogeneous stratum of groups 70/15/15 by *group count*.

    ``groups`` is a list of ``(group_id, n_items)``.  Groups in a stratum
    share the same class and the same number of items, so splitting by group
    count also keeps the item counts proportional.
    """
    ids = [gid for gid, _ in groups]
    rng.shuffle(ids)
    counts = _largest_remainder(len(ids), fractions)
    out = {}
    start = 0
    for split, n in zip(SPLITS, counts):
        for gid in ids[start:start + n]:
            out[gid] = split
        start += n
    return out


def _record(rel_image, rel_mask, label, group):
    return {
        "image": rel_image,
        "mask": rel_mask,
        "label": label,
        "group": str(group),
    }


def _stats(records_by_split):
    stats = {}
    for split in SPLITS:
        recs = records_by_split[split]
        stats[split] = {
            "images": len(recs),
            "benign": sum(r["label"] == "benign" for r in recs),
            "malignant": sum(r["label"] == "malignant" for r in recs),
            "groups": len({r["group"] for r in recs}),
        }
    return stats


def _write(name, root, records_by_split, extra=None, split_ratio=(0.7, 0.15, 0.15)):
    payload = {
        "dataset": name,
        "root": root,
        "split_ratio": list(split_ratio) if split_ratio is not None else None,
        "split": {s: records_by_split[s] for s in SPLITS},
        "stats": _stats(records_by_split),
    }
    if extra:
        payload.update(extra)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print(f"wrote {path}")
    for split in SPLITS:
        st = payload["stats"][split]
        print(
            f"  {split:5s}: {st['images']:4d} images "
            f"({st['benign']} benign / {st['malignant']} malignant), "
            f"{st['groups']} groups"
        )
    return payload


# --------------------------------------------------------------------------- #
# BUS_UC -- image-level stratified
# --------------------------------------------------------------------------- #
def make_bus_uc(fractions, rng):
    items = []
    for label in ("benign", "malignant"):
        folder = "Benign" if label == "benign" else "Malignant"
        img_dir = os.path.join(BUS_UC_ROOT, folder, "images")
        mask_dir = os.path.join(BUS_UC_ROOT, folder, "masks")
        for fn in sorted(os.listdir(img_dir)):
            if not fn.lower().endswith(".png"):
                continue
            if not os.path.exists(os.path.join(mask_dir, fn)):
                raise FileNotFoundError(f"missing mask for BUS_UC/{folder}/{fn}")
            items.append((label, fn))

    by_label = defaultdict(list)
    for label, fn in items:
        by_label[label].append(fn)

    records_by_split = {s: [] for s in SPLITS}
    for label in ("benign", "malignant"):
        names = by_label[label]
        rng.shuffle(names)
        counts = _largest_remainder(len(names), fractions)
        start = 0
        for split, n in zip(SPLITS, counts):
            for fn in names[start:start + n]:
                folder = "Benign" if label == "benign" else "Malignant"
                records_by_split[split].append(
                    _record(
                        os.path.join("BUS_UC", "BUS_UC", folder, "images", fn),
                        os.path.join("BUS_UC", "BUS_UC", folder, "masks", fn),
                        label,
                        fn,  # no patient ID in BUS_UC -> image is its own group
                    )
                )
            start += n

    for split in SPLITS:
        records_by_split[split].sort(key=lambda r: (r["label"], r["image"]))

    return _write(
        "BUS_UC",
        BUS_UC_ROOT,
        records_by_split,
        extra={
            "grouping": "image",
            "note": "BUS_UC provides no patient/case IDs; split is image-level "
                    "stratified. Images are already 256x256.",
        },
    )


# --------------------------------------------------------------------------- #
# BUSBRA -- patient-level (Case) stratified
# --------------------------------------------------------------------------- #
def make_busbra(fractions, rng):
    csv_path = os.path.join(BUSBRA_ROOT, "bus_data.csv")
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    case_label = {}
    case_images = defaultdict(list)
    for row in rows:
        cid = row["Case"]
        label = row["Pathology"].strip().lower()
        if label not in ("benign", "malignant"):
            raise ValueError(f"unexpected Pathology {label!r} for {row['ID']}")
        if cid in case_label and case_label[cid] != label:
            raise ValueError(f"case {cid} has mixed labels")
        case_label[cid] = label
        case_images[cid].append(row["ID"])

    assignment = {}
    for label in ("benign", "malignant"):
        sizes = sorted({len(ids) for cid, ids in case_images.items()
                        if case_label[cid] == label})
        for size in sizes:
            groups = [(cid, len(ids)) for cid, ids in case_images.items()
                      if case_label[cid] == label and len(ids) == size]
            assignment.update(_allocate_stratum(groups, fractions, rng))
            print(f"BUSBRA {label}, {size} image(s)/case: "
                  f"{len(groups)} cases -> {_largest_remainder(len(groups), fractions)}")

    records_by_split = {s: [] for s in SPLITS}
    for row in rows:
        cid = row["Case"]
        rid = row["ID"]
        records_by_split[assignment[cid]].append(
            _record(
                os.path.join("BUSBRA", "BUSBRA", "Images", f"{rid}.png"),
                os.path.join("BUSBRA", "BUSBRA", "Masks",
                             f"mask_{rid.replace('bus_', '')}.png"),
                case_label[cid],
                cid,
            )
        )

    for split in SPLITS:
        records_by_split[split].sort(key=lambda r: (r["label"], r["image"]))

    return _write(
        "BUSBRA",
        BUSBRA_ROOT,
        records_by_split,
        extra={
            "grouping": "patient (Case)",
            "note": "All images of a Case stay in the same split. Labels come "
                    "from bus_data.csv (Pathology).",
        },
    )


# --------------------------------------------------------------------------- #
# BUSI_WHU -- keep original splits, describe them
# --------------------------------------------------------------------------- #
def make_busi_whu():
    labels_path = os.path.join(OUT_DIR, "BUSI_WHU_labels.json")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"{labels_path} not found. It maps image filename -> benign/malignant."
        )
    with open(labels_path, encoding="utf-8") as f:
        labels = json.load(f)

    layout = {
        "train": ("train/img/ori", "train/gt/ori", True),
        "val": ("valid/img", "valid/gt", False),
        "test": ("test/img", "test/gt", False),
    }

    records_by_split = {}
    for split, (img_rel, gt_rel, is_train) in layout.items():
        img_dir = os.path.join(BUSI_WHU_ROOT, img_rel)
        gt_dir = os.path.join(BUSI_WHU_ROOT, gt_rel)
        recs = []
        for fn in sorted(os.listdir(img_dir)):
            if not fn.lower().endswith(".bmp"):
                continue
            gt_fn = fn.replace(".bmp", "_anno.bmp")
            if not os.path.exists(os.path.join(gt_dir, gt_fn)):
                raise FileNotFoundError(f"missing GT for BUSI_WHU {split}/{fn}")
            if fn not in labels:
                raise KeyError(f"no label for BUSI_WHU {fn}")
            recs.append(
                _record(
                    os.path.join("BUSI_WHU", img_rel, fn),
                    os.path.join("BUSI_WHU", gt_rel, gt_fn),
                    labels[fn],
                    fn[:-4].split("-")[0],  # base ID groups augmented variants
                )
            )
        records_by_split[split] = recs

    # leakage report: base IDs present in more than one split
    groups = {s: {r["group"] for r in records_by_split[s]} for s in SPLITS}
    pairs = {}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            pairs[f"{a}-{b}"] = len(groups[a] & groups[b])
    multi = sum(1 for g in set().union(*groups.values())
                if sum(g in groups[s] for s in SPLITS) > 1)

    return _write(
        "BUSI_WHU",
        BUSI_WHU_ROOT,
        records_by_split,
        split_ratio=None,
        extra={
            "grouping": "base image ID (augmented -N variants share a base)",
            "note": "Original train/valid/test folders kept unchanged (roughly "
                    "60/20/20). Labels were recovered by exact MD5 matching "
                    "against the official HuggingFace mirror "
                    "huangjin520/busi-whu-seg.",
            "leakage": {"overlapping_base_ids": multi, "per_pair": pairs},
        },
    )


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fractions", type=float, nargs=3,
                        default=[0.7, 0.15, 0.15])
    args = parser.parse_args()
    fracs = list(args.fractions)
    assert abs(sum(fracs) - 1.0) < 1e-9, "fractions must sum to 1"

    rng = random.Random(args.seed)
    make_bus_uc(fracs, rng)
    make_busbra(fracs, rng)
    make_busi_whu()


if __name__ == "__main__":
    main()
