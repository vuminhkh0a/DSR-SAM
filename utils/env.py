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


def get_device(default=DEFAULT_DEVICE):
    """Return the CUDA device configured in ``.env`` (or CPU as a fallback)."""
    load_env()
    device = os.environ.get('CUDA_DEVICE', default).strip() or default
    if device.startswith('cuda'):
        try:
            import torch
        except ImportError:
            return default
        if not torch.cuda.is_available():
            return 'cpu'
    return device


# Make ``.env`` available as soon as this module is imported.
load_env()
