from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Prefer CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
