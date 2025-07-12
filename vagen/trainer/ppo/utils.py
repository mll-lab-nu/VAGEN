"""Utility helpers for PPO training (VAGEN).

Currently provides:
    • seed_everything(seed: int) – set deterministic seed for Python `random`,
      NumPy, torch (CPU & CUDA).  Also sets `PYTHONHASHSEED` and forces
      deterministic cuDNN behaviour.

The helper is intentionally lightweight so it can be imported both in driver
code and inside Ray worker actors without introducing circular dependencies.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch

__all__ = ["seed_everything"]


def seed_everything(seed: int, *, deterministic: bool = True, warn: bool = True) -> None:
    """Seed *all* common PRNGs to make experiment deterministic.

    Parameters
    ----------
    seed : int
        The random seed to set.
    deterministic : bool, default True
        If *True* will apply extra flags to make cuDNN deterministic
        (slower but reproducible).  Can be disabled if performance is
        preferred over bit-wise reproducibility.
    warn : bool, default True
        If *True* prints a short message when called multiple times.
    """
    # Prevent accidental reseeding in the same process unless user opts out.
    if hasattr(seed_everything, "_seeded") and seed_everything._seeded and warn:
        print("[seed_everything] WARNING: reseeding the same Python process.")

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    seed_everything._seeded = True  # type: ignore[attr-defined]
