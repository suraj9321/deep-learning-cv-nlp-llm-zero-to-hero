"""Device selection and reproducibility helpers."""
import os
import random

import numpy as np
import torch


def get_device(verbose: bool = True) -> torch.device:
    """Pick the best available accelerator: CUDA (Colab/cloud) -> MPS (Apple Silicon) -> CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if verbose:
        print(f"Using device: {device}")
    return device


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch so runs are repeatable.

    Note: GPU kernels (especially on MPS) are not always bit-for-bit deterministic,
    so expect tiny differences between runs even with the same seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and MPS generators
    os.environ["PYTHONHASHSEED"] = str(seed)


def fast_dev_run() -> bool:
    """True when FAST_DEV_RUN=1: notebooks shrink epochs/data so they finish in seconds (used for checks)."""
    return os.environ.get("FAST_DEV_RUN") == "1"
