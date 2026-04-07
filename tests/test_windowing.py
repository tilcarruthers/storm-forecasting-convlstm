import numpy as np
from storm_forecasting.data.windowing import count_windows, make_windows


def test_count_windows_matches_expected() -> None:
    assert count_windows(sequence_length=36, tin=12, tout=12, stride=1) == 13


def test_make_windows_shapes() -> None:
    vil = np.zeros((36, 32, 32), dtype=np.float32)
    windows = make_windows(vil, tin=12, tout=12, stride=1)
    x, y, t0 = windows[0]
    assert x.shape == (12, 32, 32)
    assert y.shape == (12, 32, 32)
    assert t0 == 0
