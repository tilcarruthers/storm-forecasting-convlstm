from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from storm_forecasting.config import load_config, save_flat_config_csv, save_resolved_config
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
    parser.add_argument(
        "--device", type=str, default=None, help="Optional device override, e.g. cpu or cuda"
    )
    parser.add_argument(
        "--num-workers", type=int, default=None, help="Optional dataloader worker override"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Optional dataloader batch size override"
    )
    parser.add_argument(
        "--max-batches", type=int, default=None, help="Evaluate only the first N batches"
    )
    parser.add_argument(
        "--save-config-artifacts",
        action="store_true",
        help="Save resolved YAML and a flattened CSV of the run config alongside metrics.",
    )
    return parser.parse_args()


def _resolve_device(device_arg: str | None) -> torch.device:
    return torch.device(device_arg) if device_arg else get_device()


def _write_eval_summary_csv(metrics: dict[str, float], save_path: str | Path) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in sorted(metrics.items()):
            writer.writerow([key, value])
    return save_path


@torch.no_grad()
def evaluate_streaming(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp_enabled: bool,
    compute_ssim: bool,
    weighted_mae_cfg: dict | None,
    max_batches: int | None = None,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    weight_sum = 0.0
    model.eval()

    total = len(loader)
    if max_batches is not None:
        total = min(total, max_batches)

    for batch_idx, (x, y, _) in enumerate(tqdm(loader, total=total, desc="Evaluate", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type=device.type, enabled=amp_enabled and device.type == "cuda"
        ):
            y_hat = model(x)

        batch_metrics = compute_metrics(
            y_hat.detach().cpu(),
            y.detach().cpu(),
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

    device = _resolve_device(args.device)
    batch_size = int(args.batch_size or train_cfg["batch_size"])
    num_workers = int(train_cfg["num_workers"] if args.num_workers is None else args.num_workers)

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
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Num test events: {len(test_ids)}")
    print(f"Num test samples: {len(test_ds)}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Max batches: {args.max_batches}")

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
        weighted_mae_cfg=eval_cfg.get("weighted_mae")
        if bool(eval_cfg.get("compute_weighted_mae", False))
        else None,
        max_batches=args.max_batches,
    )

    horizons = mae_per_horizon(
        model=model,
        loader=test_loader,
        device=device,
        amp_enabled=bool(train_cfg["amp"]),
        tout=int(data_cfg["tout"]),
        max_batches=args.max_batches,
    )

    metrics_dir = ensure_dir(outputs_cfg["metrics_dir"])
    figures_dir = ensure_dir(outputs_cfg["figures_dir"])
    metrics_json_path = metrics_dir / f"{outputs_cfg['run_name']}_eval.json"
    metrics_csv_path = metrics_dir / f"{outputs_cfg['run_name']}_eval.csv"
    horizon_csv_path = metrics_dir / f"{outputs_cfg['run_name']}_mae_per_horizon.csv"
    metrics_json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_eval_summary_csv(metrics, metrics_csv_path)

    with horizon_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["forecast_step", "mae"])
        for idx, value in enumerate(horizons, start=1):
            writer.writerow([idx, float(value)])

    plot_horizon_metric(
        horizons,
        save_path=figures_dir / f"{outputs_cfg['run_name']}_mae_per_horizon.png",
    )

    if args.save_config_artifacts:
        save_resolved_config(
            config, metrics_dir / f"{outputs_cfg['run_name']}_resolved_config.yaml"
        )
        save_flat_config_csv(config, metrics_dir / f"{outputs_cfg['run_name']}_resolved_config.csv")

    print("Evaluation metrics:", metrics)
    print(f"Saved metrics JSON to {metrics_json_path}")
    print(f"Saved metrics CSV to {metrics_csv_path}")
    print(f"Saved horizon CSV to {horizon_csv_path}")


if __name__ == "__main__":
    main()
