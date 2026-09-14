import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import random
import warnings
import numpy as np
import torch
import cv2
import torch.backends.cudnn as cudnn

# Silence torch / CUDA determinism warnings while keeping deterministic
# behavior fully enabled (no mechanism is changed).
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', message='.*[Dd]eterministic.*')
warnings.filterwarnings('ignore', message='.*[Cc][Uu][Dd][Aa].*')


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
