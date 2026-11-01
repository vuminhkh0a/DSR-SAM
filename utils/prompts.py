"""Shared bounding-box prompt loading for every DG method.

The benchmark has two prompt sources:

``prompt_method='gt'``
    Ground-truth boxes from ``box_coords.json`` (unchanged, read-only).

``prompt_method='yolov8'``
    Boxes predicted by the vendored YOLOv8 models, stored in
    ``box_coords_yolov8_<size>.json`` (one file per model size, seed 42).
    Those files are source-keyed: ``data[source][box_key] -> entries``.  The
    section matching the DG training source is selected, so a target image can
    have different boxes for ``s_OTU`` and ``s_USOVA`` without overwriting.

All loaders return the first box of an entry, matching the historical
``box_coords.json`` behaviour (the DG datasets consume one prompt box per
slice).  Missing/empty boxes fall back to the full image, exactly like before.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOX_COORDS_PATH = PROJECT_ROOT / 'box_coords.json'
YOLO_BOX_COORDS_TEMPLATE = str(PROJECT_ROOT / 'box_coords_yolov8_{size}.json')

# DG domain name -> top-level key used by box_coords.json.
DOMAIN_TO_BOX_KEY = {
    'OTU': 'OTU2D',
    'OVATUS': 'OVATUS',
    'USOVA': 'USOVA3D',
}
BOX_KEY_TO_DOMAIN = {v: k for k, v in DOMAIN_TO_BOX_KEY.items()}
DOMAIN_ORDER = ['OTU', 'OVATUS', 'USOVA']

_PROMPT_METHODS = {'gt', 'yolov8', 'yolo'}
_YOLO_SIZES = {'n', 's', 'm', 'l', 'x'}


def normalize_prompt_method(prompt_method):
    method = str(prompt_method or 'gt').strip().lower()
    if method == 'yolo':
        method = 'yolov8'
    if method not in _PROMPT_METHODS:
        raise ValueError(
            f"prompt_method must be 'gt' or 'yolov8' (got {prompt_method!r})")
    return method


def normalize_yolo_size(yolo_size):
    size = str(yolo_size or 'm').strip().lower()
    if size not in _YOLO_SIZES:
        raise ValueError(
            f"prompt_yolo_size must be one of {sorted(_YOLO_SIZES)} "
            f"(got {yolo_size!r})")
    return size


def resolve_box_coords_path(prompt_method='gt', yolo_size='m', path=None):
    """Return the JSON file used for prompts.

    ``path`` overrides everything (useful for ablations / custom files).
    """
    if path:
        return str(path)
    method = normalize_prompt_method(prompt_method)
    if method == 'gt':
        return str(BOX_COORDS_PATH)
    return YOLO_BOX_COORDS_TEMPLATE.format(size=normalize_yolo_size(yolo_size))


def _norm_path(path):
    return os.path.normpath(os.path.abspath(str(path)))


def _load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _select_yolo_section(data, source=None, target=None):
    """Pick ``data[source]`` from a source-keyed YOLO box file.

    Falls back to ``data[target]`` (self-domain boxes, used by test-only
    methods that have no training source), then to the first section.
    """
    candidates = []
    for value in (source, target):
        if value is None:
            continue
        candidates.append(str(value))
        dg_name = BOX_KEY_TO_DOMAIN.get(str(value))
        if dg_name is not None:
            candidates.append(dg_name)
        box_key = DOMAIN_TO_BOX_KEY.get(str(value))
        if box_key is not None:
            candidates.append(box_key)
    for key in candidates:
        if key in data:
            return data[key], key
    if not data:
        raise ValueError('YOLO box file is empty')
    first = next(iter(data))
    return data[first], first


def load_entries(prompt_method='gt', yolo_size='m', source=None, target=None,
                 path=None):
    """Return one flat list of ``{image, mask, boxes}`` entries.

    For YOLO files the source section is selected first; ``source=None``
    (test-only DG methods) selects the target's own section.
    """
    method = normalize_prompt_method(prompt_method)
    resolved = resolve_box_coords_path(method, yolo_size, path)
    if not os.path.isfile(resolved):
        if method == 'yolov8':
            raise FileNotFoundError(
                f'YOLO prompt box file not found: {resolved}. Run '
                f'`python -m yolo.main --sizes {normalize_yolo_size(yolo_size)} '
                f'--seeds 42` first.')
        raise FileNotFoundError(f'Ground-truth prompt file not found: {resolved}')
    data = _load_json(resolved)
    if method == 'gt':
        entries = []
        for value in data.values():
            entries.extend(value)
        return entries
    section, _ = _select_yolo_section(data, source=source, target=target)
    entries = []
    for value in section.values():
        entries.extend(value)
    return entries


@lru_cache(maxsize=32)
def _cached_box_map(prompt_method, yolo_size, source, target, path):
    entries = load_entries(prompt_method, yolo_size, source=source,
                           target=target, path=path)
    box_map = {}
    for e in entries:
        box = e['boxes'][0] if e.get('boxes') else None
        box_map[_norm_path(e['image'])] = box
    return box_map


def load_box_map(prompt_method='gt', yolo_size='m', source=None, target=None,
                 path=None):
    """Map normalized image path -> first box ``[x_min,y_min,x_max,y_max]``.

    ``None`` means the image had no box (blank GT or no YOLO detection).
    """
    method = normalize_prompt_method(prompt_method)
    size = normalize_yolo_size(yolo_size)
    return _cached_box_map(method, size,
                           None if source is None else str(source),
                           None if target is None else str(target),
                           None if path is None else str(path))


def boxes_for_images(images, prompt_method='gt', yolo_size='m', source=None,
                     target=None, image_size=256, path=None):
    """Return one box per image, with full-image fallback for missing boxes."""
    box_map = load_box_map(prompt_method=prompt_method, yolo_size=yolo_size,
                           source=source, target=target, path=path)
    fallback = [0, 0, int(image_size), int(image_size)]
    boxes = []
    for img in images:
        box = box_map.get(_norm_path(img))
        boxes.append(list(box) if box is not None else list(fallback))
    return boxes


def clear_cache():
    _cached_box_map.cache_clear()
