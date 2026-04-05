from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from storm_forecasting.config import load_config
from storm_forecasting.data.dataset import VILSeq2SeqDataset, build_dataloader
from storm_forecasting.data.io import load_index_csv
from storm_forecasting.data.splits import split_ids
from storm_forecasting.evaluation.horizon_metrics import mae_per_horizon, plot_horizon_metric
from storm_forecasting.evaluation.metrics import compute_metrics
from storm_forecasting.models.seq2seq_unet import ConvLSTMUNetSeq2Seq
from storm_forecasting.paths import ensure_dir
from storm_forecasting.training.checkpoints import load_checkpoint, load_model_state
from storm_forecasting.utils.device import get_device
from storm_forecasting.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained storm forecasting model.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path. Defaults to outputs/checkpoints/<run_name>/best.pt",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate_streaming(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp_enabled: bool,
    compute_ssim: bool,
    weighted_mae_cfg: dict | None,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    weight_sum = 0.0
    model.eval()

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled and device.type == "cuda"):
            y_hat = model(x)

        batch_metrics = compute_metrics(
            y_hat.cpu(),
            y.cpu(),
            compute_ssim=compute_ssim,
            weighted_mae_cfg=weighted_mae_cfg,
        )
        batch_weight = float(x.size(0))
        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_weight
        weight_sum += batch_weight

    return {key: value / max(weight_sum, 1.0) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    setup_logging()

    config = load_config(args.config)
    data_cfg = config["data"]
    train_cfg = config["training"]
    eval_cfg = config["evaluation"]
    model_cfg = config["model"]
    outputs_cfg = config["outputs"]

    device = get_device()

    checkpoint_path = args.checkpoint or str(
        Path(outputs_cfg["checkpoints_dir"]) / outputs_cfg["run_name"] / "best.pt"
    )

    df_index = load_index_csv(data_cfg["index_csv"], event_col=data_cfg["event_col"])
    all_ids = df_index[data_cfg["event_col"]].dropna().astype(str).str.strip().unique().tolist()
    _, _, test_ids = split_ids(
        all_ids,
        test_frac=float(data_cfg["test_frac"]),
        val_frac_of_train=float(data_cfg["val_frac_of_train"]),
        seed=int(config["seed"]),
    )

    test_ds = VILSeq2SeqDataset(
        df_index=df_index,
        event_ids=test_ids,
        h5_path=data_cfg["h5_path"],
        tin=int(data_cfg["tin"]),
        tout=int(data_cfg["tout"]),
        stride=int(data_cfg["stride"]),
        use_sliding_windows=bool(data_cfg["use_sliding_windows"]),
        use_crops=False,
        crop_size=data_cfg.get("crop_size"),
        mode="test",
        normalize_divisor=float(data_cfg["normalize_divisor"]),
        event_col=data_cfg["event_col"],
        dataset_key=data_cfg["dataset_key"],
    )
    test_loader = build_dataloader(
        test_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
    )

    model = ConvLSTMUNetSeq2Seq(
        base=int(model_cfg["base"]),
        bottleneck_ch=int(model_cfg["bottleneck_ch"]),
        tin=int(data_cfg["tin"]),
        tout=int(data_cfg["tout"]),
        dropout_p=float(model_cfg.get("dropout_p", 0.0)),
    ).to(device)

    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    load_model_state(model, checkpoint)

    metrics = evaluate_streaming(
        model=model,
        loader=test_loader,
        device=device,
        amp_enabled=bool(train_cfg["amp"]),
        compute_ssim=bool(eval_cfg.get("compute_ssim", False)),
        weighted_mae_cfg=eval_cfg.get("weighted_mae") if bool(eval_cfg.get("compute_weighted_mae", False)) else None,
    )

    horizons = mae_per_horizon(
        model=model,
        loader=test_loader,
        device=device,
        amp_enabled=bool(train_cfg["amp"]),
        tout=int(data_cfg["tout"]),
    )

    metrics_dir = ensure_dir(outputs_cfg["metrics_dir"])
    figures_dir = ensure_dir(outputs_cfg["figures_dir"])
    metrics_path = metrics_dir / f"{outputs_cfg['run_name']}_eval.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plot_horizon_metric(
        horizons,
        save_path=figures_dir / f"{outputs_cfg['run_name']}_mae_per_horizon.png",
    )

    print("Evaluation metrics:", metrics)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
