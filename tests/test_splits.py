from storm_forecasting.data.splits import assert_non_overlapping_splits, split_ids


def test_split_ids_non_overlapping() -> None:
    ids = [f"S{i:03d}" for i in range(10)]
    train_ids, val_ids, test_ids = split_ids(ids, test_frac=0.3, val_frac_of_train=0.2, seed=42)
    assert_non_overlapping_splits(train_ids, val_ids, test_ids)
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(ids)
