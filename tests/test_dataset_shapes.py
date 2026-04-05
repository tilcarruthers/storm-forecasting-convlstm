from storm_forecasting.data.dataset import VILSeq2SeqDataset


def test_dataset_item_shapes(synthetic_h5, synthetic_index) -> None:
    ds = VILSeq2SeqDataset(
        df_index=synthetic_index,
        event_ids=["S000"],
        h5_path=synthetic_h5,
        tin=12,
        tout=12,
        stride=1,
        use_sliding_windows=True,
        use_crops=False,
        crop_size=None,
        mode="train",
    )
    x, y, meta = ds[0]
    assert x.shape == (12, 1, 32, 32)
    assert y.shape == (12, 1, 32, 32)
    assert meta["event_id"] == "S000"
