from __future__ import annotations

import numpy as np


def count_windows(sequence_length: int, tin: int, tout: int, stride: int = 1) -> int:
    """Return the number of valid direct-forecast windows."""
    max_start = sequence_length - (tin + tout)
    if max_start < 0:
        return 0
    return (max_start // stride) + 1


def make_windows(
    vil: np.ndarray,
    tin: int,
    tout: int,
    stride: int = 1,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Create sliding windows from a (T, H, W) sequence."""
    if vil.ndim != 3:
        raise ValueError(f"Expected (T, H, W), got {vil.shape}")

    total_steps = vil.shape[0]
    max_start = total_steps - (tin + tout)
    if max_start < 0:
        return []

    windows: list[tuple[np.ndarray, np.ndarray, int]] = []
    for t0 in range(0, max_start + 1, stride):
        x = vil[t0 : t0 + tin]
        y = vil[t0 + tin : t0 + tin + tout]
        windows.append((x, y, t0))
    return windows
