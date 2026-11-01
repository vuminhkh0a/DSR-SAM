"""Validate a box_coords_yolov8_<size>.json file against box_coords.json.

Checks:
  * every present source section has the expected domain keys;
  * each domain list has exactly the DG samples, in box_coords.json order;
  * image/mask identity matches;
  * every box is ``[x_min, y_min, x_max, y_max]`` with integer 0..256 coords
    and positive width/height.

Usage:
    python -m yolo.check_boxes box_coords_yolov8_m.json
"""
from __future__ import annotations

import argparse
import json
import sys

from yolo import config as C
from yolo.data import ordered_entries


def check_file(path, strict_sources=True):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    problems = []
    seen_sources = 0
    for source in C.DOMAINS:
        if source not in data:
            if strict_sources:
                problems.append(f'missing source section: {source}')
            continue
        seen_sources += 1
        section = data[source]
        for box_key in C.BOX_KEY_ORDER:
            if box_key not in section:
                problems.append(f'{source}: missing domain key {box_key}')
                continue
            domain = {v: k for k, v in C.DOMAIN_TO_BOX_KEY.items()}[box_key]
            expected = ordered_entries(domain)
            actual = section[box_key]
            if len(actual) != len(expected):
                problems.append(
                    f'{source}/{box_key}: {len(actual)} entries, expected '
                    f'{len(expected)}')
                continue
            for i, (a, e) in enumerate(zip(actual, expected)):
                if a.get('image') != e['image'] or a.get('mask') != e['mask']:
                    problems.append(
                        f'{source}/{box_key}[{i}]: image/mask mismatch '
                        f'({a.get("image")} vs {e["image"]})')
                    break
                for b in a.get('boxes', []):
                    if (not isinstance(b, list) or len(b) != 4 or
                            not all(isinstance(v, int) for v in b)):
                        problems.append(f'{source}/{box_key}[{i}]: bad box {b}')
                        break
                    x1, y1, x2, y2 = b
                    if not (0 <= x1 < x2 <= C.IMAGE_SIZE and
                            0 <= y1 < y2 <= C.IMAGE_SIZE):
                        problems.append(f'{source}/{box_key}[{i}]: out-of-range box {b}')
                        break
            n_empty = sum(1 for a in actual if not a.get('boxes'))
            print(f'{source}/{box_key}: {len(actual)} entries, {n_empty} empty')
    if seen_sources == 0:
        problems.append('no source sections found')
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--allow-missing-sources', action='store_true')
    args = ap.parse_args()
    problems = check_file(args.path,
                          strict_sources=not args.allow_missing_sources)
    if problems:
        print('\nFAILED:')
        for p in problems:
            print('  -', p)
        return 1
    print('\nOK: sample identity, order and box format are valid.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
