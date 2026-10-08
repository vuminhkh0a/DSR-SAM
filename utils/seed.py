import faulthandler
import os
import random
import signal
import sys
import warnings

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

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
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cv2.setRNGSeed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def worker_init_fn(worker_id, seed=42):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    cv2.setNumThreads(0)
    torch.set_num_threads(1)


def get_generator(seed=42):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
