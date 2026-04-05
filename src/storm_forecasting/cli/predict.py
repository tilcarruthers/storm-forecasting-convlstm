from __future__ import annotations

import argparse
from pathlib import Path

from storm_forecasting.config import load_config
from storm_forecasting.data.dataset import VILSeq2SeqDataset
from storm_forecasting.data.io import load_index_csv
from storm_forecasting.data.splits import split_ids
from storm_forecasting.evaluation.qualitative import (
    plot_compact_panel,
    plot_example,
    predict_example,
    save_pred_gt_gif,
)
from storm_forecasting.models.seq2seq_unet import ConvLSTMUNetSeq2Seq
from storm_forecasting.paths import ensure_dir
from storm_forecasting.training.checkpoints import load_checkpoint, load_model_state
from storm_forecasting.utils.device import get_device
from storm_forecasting.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate qualitative predictions.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.")
    parser.add_argument("--index", type=int, default=0, help="Dataset index to visualise.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    config = load_config(args.config)
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    outputs_cfg = config["outputs"]

    device = get_device()

    df_index = load_index_csv(data_cfg["index_csv"], event_col=data_cfg["event_col"])
    all_ids = df_index[data_cfg["event_col"]].dropna().astype(str).str.strip().unique().tolist()
    train_ids, val_ids, test_ids = split_ids(
        all_ids,
        test_frac=float(data_cfg["test_frac"]),
        val_frac_of_train=float(data_cfg["val_frac_of_train"]),
        seed=int(config["seed"]),
    )

    split_map = {"train": train_ids, "val": val_ids, "test": test_ids}
    dataset = VILSeq2SeqDataset(
        df_index=df_index,
        event_ids=split_map[args.split],
        h5_path=data_cfg["h5_path"],
        tin=int(data_cfg["tin"]),
        tout=int(data_cfg["tout"]),
        stride=int(data_cfg["stride"]),
        use_sliding_windows=bool(data_cfg["use_sliding_windows"]),
        use_crops=False,
        crop_size=data_cfg.get("crop_size"),
        mode=args.split,
        normalize_divisor=float(data_cfg["normalize_divisor"]),
        event_col=data_cfg["event_col"],
        dataset_key=data_cfg["dataset_key"],
    )

    model = ConvLSTMUNetSeq2Seq(
        base=int(model_cfg["base"]),
        bottleneck_ch=int(model_cfg["bottleneck_ch"]),
        tin=int(data_cfg["tin"]),
        tout=int(data_cfg["tout"]),
        dropout_p=float(model_cfg.get("dropout_p", 0.0)),
    ).to(device)

    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    load_model_state(model, checkpoint)

    x, y, y_hat, meta = predict_example(
        model=model,
        dataset=dataset,
        index=args.index,
        device=device,
        amp_enabled=bool(train_cfg["amp"]),
    )

    pred_dir = ensure_dir(Path(outputs_cfg["predictions_dir"]) / outputs_cfg["run_name"])
    example_path = pred_dir / f"example_{args.split}_{args.index}.png"
    compact_path = pred_dir / f"compact_{args.split}_{args.index}.png"
    gif_path = pred_dir / f"pred_vs_gt_{args.split}_{args.index}.gif"

    plot_example(x, y, y_hat, save_path=example_path)
    plot_compact_panel(y, y_hat, save_path=compact_path)
    save_pred_gt_gif(y, y_hat, gif_path)

    print("Saved qualitative outputs:")
    print(f"  example: {example_path}")
    print(f"  compact: {compact_path}")
    print(f"  gif: {gif_path}")
    print(f"  meta: {meta}")


if __name__ == "__main__":
    main()
