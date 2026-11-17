"""Apply the configured random seed to the local runtime."""

from __future__ import annotations

import os
import random


def apply_configured_seed() -> int | None:
    """Seed Python, NumPy, and Torch from ``SOUNDLAB_SEED``.

    An empty or unset variable leaves the runtime unseeded and returns ``None``.
    """
    raw = os.getenv("SOUNDLAB_SEED")
    if raw is None or raw.strip() == "":
        return None
    seed = int(raw)
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        np = None
    else:
        np.random.seed(seed)
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    return seed
