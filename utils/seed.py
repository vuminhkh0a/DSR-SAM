import faulthandler
import os
import random
import signal
import sys
import warnings
from contextlib import contextmanager

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Phase 1 (reproducibility): deterministic cuBLAS requires a workspace config
# set BEFORE the CUDA context is created. Must run before `import torch` below.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

# Current run seed (updated by set_seed), used as the default for the loader
# generator and DataLoader worker seeding so RNG scope is per-run, not global.
_SEED = 42

import numpy as np
import torch
import cv2
import torch.backends.cudnn as cudnn

# ---------------------------------------------------------------------------
# Notifications: by DEFAULT every DG method now prints warnings (torch, CUDA,
# determinism, ...) instead of swallowing them, so a run never stops without an
# explanation. Set Y_DG_SUPPRESS_WARNINGS=1 to opt back into the old quiet mode.
# ---------------------------------------------------------------------------
if os.environ.get('Y_DG_SUPPRESS_WARNINGS') == '1':
    warnings.filterwarnings('ignore', category=UserWarning, module='torch')
    warnings.filterwarnings('ignore', message='.*[Dd]eterministic.*')
    warnings.filterwarnings('ignore', message='.*[Cc][Uu][Dd][Aa].*')

# Turn silent, hard deaths (segfault / abort / bus error / illegal instruction)
# into a visible Python traceback written to stderr (and therefore to log.txt).
faulthandler.enable(all_threads=True)


def _log_signal(signum, _frame):
    """Print which catchable signal is killing us, then die with that signal."""
    try:
        name = signal.Signals(signum).name
    except ValueError:
        name = str(signum)
    print(f'\n[FATAL] Received {name} ({signum}); the process is being '
          f'terminated by the OS or an external kill.', flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


# Catch SIGTERM / SIGINT so an external `kill` leaves a trace in the log.
# SIGHUP is intentionally left untouched so `nohup` background jobs keep
# surviving terminal disconnects.
for _sig in (signal.SIGTERM, signal.SIGINT):
    try:
        signal.signal(_sig, _log_signal)
    except (ValueError, OSError):
        pass


def set_seed(seed=42):
    global _SEED
    _SEED = seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cv2.setRNGSeed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    # Phase 1 (reproducibility): TF32 changes numerics across GPUs / torch
    # versions, so disable it and force the highest float32 matmul precision.
    try:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
    except Exception:
        pass
    # Phase 1 (reproducibility): strict mode (warn_only=False) makes any
    # op without a deterministic implementation raise instead of silently
    # falling back to a nondeterministic one. Set Y_DG_STRICT_DETERMINISM=0
    # to restore the old warn-only behaviour.
    strict = os.environ.get('Y_DG_STRICT_DETERMINISM', '1') != '0'
    torch.use_deterministic_algorithms(True, warn_only=not strict)


def worker_init_fn(worker_id, seed=None):
    if seed is None:
        seed = _SEED
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    cv2.setNumThreads(0)
    torch.set_num_threads(1)


def get_generator(seed=None):
    if seed is None:
        seed = _SEED
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


@contextmanager
def deterministic_warn_only():
    """Temporarily downgrade deterministic mode to warn-only.

    Phase 1 (reproducibility): a few ops have no deterministic CUDA kernel in
    PyTorch yet are exact for their inputs here (e.g. ``cumsum`` over the
    constant grid in ``PositionEmbeddingRandom``). Wrap just those calls so
    strict mode stays on everywhere else. The previous setting is restored.
    """
    prev_enabled = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev_enabled, warn_only=prev_warn)
