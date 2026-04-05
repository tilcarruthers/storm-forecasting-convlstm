from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def split_ids(
    event_ids: Iterable[str],
    test_frac: float = 0.30,
    val_frac_of_train: float = 0.20,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Create non-overlapping storm-wise train / val / test splits."""
    event_ids = [str(event_id).strip() for event_id in event_ids]
    rng = np.random.default_rng(seed)
    rng.shuffle(event_ids)

    n_total = len(event_ids)
    n_test = int(round(n_total * test_frac))
    test_ids = event_ids[:n_test]
    trainval_ids = event_ids[n_test:]

    n_val = int(round(len(trainval_ids) * val_frac_of_train))
    val_ids = trainval_ids[:n_val]
    train_ids = trainval_ids[n_val:]

    return train_ids, val_ids, test_ids


def assert_non_overlapping_splits(
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
) -> None:
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    assert not (train_set & val_set), "Train and validation splits overlap."
    assert not (train_set & test_set), "Train and test splits overlap."
    assert not (val_set & test_set), "Validation and test splits overlap."
