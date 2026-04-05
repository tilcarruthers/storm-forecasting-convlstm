from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def synthetic_h5(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic.h5"
    with h5py.File(path, "w") as handle:
        for idx in range(3):
            grp = handle.create_group(f"S{idx:03d}")
            arr = np.random.randint(0, 256, size=(32, 32, 36), dtype=np.uint8)
            grp.create_dataset("vil", data=arr)
    return path


@pytest.fixture()
def synthetic_index() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["S000", "S001", "S002"],
            "img_type": ["vil", "vil", "vil"],
            "event_type": ["storm", "storm", "storm"],
        }
    )
