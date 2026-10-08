"""Project-wide CUDA device configuration loaded from ``y_DG/.env``.

The ``.env`` file is the single source of truth for which GPU the entrypoints
use.  It is parsed here directly so that no extra dependency
(``python-dotenv``) is required.

Typical usage::

    from utils.env import get_device

    CONFIG = {'device': get_device(), ...}
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / '.env'

# Fallback used when ``.env`` is missing or does not define ``CUDA_DEVICE``.
DEFAULT_DEVICE = 'cuda:2'


def load_env(path=None, override=False):
    """Load ``KEY=VALUE`` pairs from a ``.env`` file into ``os.environ``.

    Existing environment variables win over the file unless ``override`` is
    True, so a value exported by the shell / scheduler still takes precedence.
    """
    env_path = Path(path) if path is not None else ENV_FILE
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[len('export '):].lstrip()
        key, sep, value = line.partition('=')
        if not sep:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip('"').strip("'")
        if override:
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)


def _visible_physical_ids():
    """Return the physical GPU ids listed in ``CUDA_VISIBLE_DEVICES``.

    Only plain integer specs (e.g. ``"2,0"``) are handled. Returns ``None``
    when the variable is unset or uses UUID / MIG specifications.
    """
    raw = os.environ.get('CUDA_VISIBLE_DEVICES')
    if raw is None:
        return None
    items = [part.strip() for part in raw.split(',') if part.strip()]
    if not items or any(not part.isdigit() for part in items):
        return None
    return items


def get_device(default=DEFAULT_DEVICE):
    """Return the CUDA device configured in ``.env`` (or CPU as a fallback).

    ``CUDA_DEVICE`` is interpreted as a **physical** GPU index. When
    ``CUDA_VISIBLE_DEVICES`` masks/renumbers the GPUs, the request is remapped
    to the matching logical index, so the same ``.env`` keeps working no matter
    how the shell exposes the hardware.
    """
    load_env()
    device = os.environ.get('CUDA_DEVICE', default).strip() or default
    if not device.startswith('cuda'):
        return device

    try:
        import torch
    except ImportError:
        return default
    if not torch.cuda.is_available():
        print(f'[y_DG] CUDA is not available; falling back to cpu '
              f'(requested {device}).')
        return 'cpu'

    count = torch.cuda.device_count()
    parts = device.split(':')
    if len(parts) == 1:
        return device  # plain 'cuda' -> torch picks the current device

    try:
        requested = int(parts[1])
    except ValueError:
        return device

    # When CUDA_VISIBLE_DEVICES is a plain integer list, CUDA_DEVICE is a
    # PHYSICAL GPU id (the number printed by nvidia-smi): remap it to the
    # logical index torch exposes. Otherwise logical == physical.
    visible = _visible_physical_ids()
    if visible is not None:
        if str(requested) not in visible:
            raise RuntimeError(
                f'CUDA_DEVICE={device} requests physical GPU {requested}, which '
                f'is not exposed (CUDA_VISIBLE_DEVICES='
                f'{os.environ["CUDA_VISIBLE_DEVICES"]}; visible physical GPUs: '
                f'[{", ".join(visible)}]). Set CUDA_DEVICE in {ENV_FILE.name} to '
                f'one of those, or unset CUDA_VISIBLE_DEVICES.'
            )
        logical = visible.index(str(requested))
        if logical != requested:
            print(f'[y_DG] CUDA_DEVICE={device} -> physical GPU {requested} is '
                  f'exposed as logical cuda:{logical} '
                  f'(CUDA_VISIBLE_DEVICES={os.environ["CUDA_VISIBLE_DEVICES"]}).')
        return f'cuda:{logical}'

    if 0 <= requested < count:
        return device

    raise RuntimeError(
        f'CUDA_DEVICE={device} is not a valid CUDA device: torch sees '
        f'{count} device(s). Set CUDA_DEVICE in {ENV_FILE.name} to '
        f'index 0..{count - 1}, or unset CUDA_VISIBLE_DEVICES.'
    )


# Make ``.env`` available as soon as this module is imported.
load_env()
