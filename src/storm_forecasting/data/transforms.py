from __future__ import annotations

import numpy as np


def normalize_unit_scale(arr: np.ndarray, divisor: float = 255.0) -> np.ndarray:
    return arr.astype(np.float32) / float(divisor)
