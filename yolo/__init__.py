"""Vendored Ultralytics YOLOv8 (reference: github.com/autogyro/yolo-V8).

The reference repository is the early ``ultralytics`` 8.0.8 tree, copied
verbatim into ``yolo/ultralytics``.  Importing this package puts that
directory in front of ``sys.path`` so ``import ultralytics`` resolves to the
vendored 8.0.8 code instead of any newer pip-installed ``ultralytics``.

Only glue code lives in ``yolo/*.py``; all model, loss, training-loop and
data-loading implementations come from the copied ``ultralytics`` package.
"""
from __future__ import annotations

import os
import sys

_VENDOR_ROOT = os.path.dirname(os.path.abspath(__file__))
if _VENDOR_ROOT not in sys.path:
    sys.path.insert(0, _VENDOR_ROOT)

# Make sure a previously imported pip ultralytics cannot shadow the vendored one.
_loaded = sys.modules.get('ultralytics')
if _loaded is not None and not os.path.abspath(getattr(_loaded, '__file__', '')).startswith(_VENDOR_ROOT):
    for _name in [m for m in sys.modules if m == 'ultralytics' or m.startswith('ultralytics.')]:
        del sys.modules[_name]


def _disable_tqdm():
    """Silence every tqdm progress bar used by the reference training code.

    tqdm 4.65 (installed here) does not implement the ``TQDM_DISABLE`` env
    var, so patch the class ``__init__`` before ultralytics imports it.
    ``from tqdm import tqdm`` in the vendored modules returns this same class.
    """
    os.environ['TQDM_DISABLE'] = '1'
    try:
        import tqdm as _tqdm
    except Exception:
        return
    cls = getattr(_tqdm, 'tqdm', None)
    if cls is None or getattr(cls, '_y_dg_tqdm_patched', False):
        return
    _original_init = cls.__init__

    def _init(self, *args, **kwargs):
        kwargs['disable'] = True
        _original_init(self, *args, **kwargs)

    cls.__init__ = _init
    cls._y_dg_tqdm_patched = True


_disable_tqdm()

from ultralytics import YOLO  # noqa: E402
import ultralytics as _ultralytics  # noqa: E402

__version__ = _ultralytics.__version__
__all__ = ['YOLO', '__version__', 'VENDOR_ROOT']

VENDOR_ROOT = _VENDOR_ROOT
