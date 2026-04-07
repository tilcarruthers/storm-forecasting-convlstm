from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from storm_forecasting.config import load_config, save_flat_config_csv, save_resolved_config
from storm_forecasting.data.dataset import VILSeq2SeqDataset, build_dataloader
from storm_forecasting.data.io import load_index_csv, validate_index_against_h5
from storm_forecasting.data.splits import assert_non_overlapping_splits, split_ids
from storm_forecasting.models.seq2seq_unet import ConvLSTMUNetSeq2Seq
from storm_forecasting.paths import ensure_dir
from storm_forecasting.seed import seed_everything
from storm_forecasting.training.engine import evaluate_regression_metrics, fit
from storm_forecasting.training.losses import get_loss
from storm_forecasting.training.optim import build_optimizer, build_scheduler
from storm_forecasting.utils.device import get_device
from storm_forecasting.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the baseline storm forecasting model.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML.")
    parser.add_argument(
        "--device", type=str, default=None, help="Optional device override, e.g. cpu or cuda"
    )
    parser.add_argument(
        "--num-workers", type=int, default=None, help="Optional dataloader worker override"
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override")
    parser.add_argument(
        "--save-config-artifacts",
        action="store_true",
        help="Save resolved YAML and a flattened CSV of the run config alongside metrics.",
    )
    return parser.parse_args()


def _resolve_device(device_arg: str | None) -> torch.device:
    return torch.device(device_arg) if device_arg else get_device()


def _write_metrics_csv(metrics: dict[str, float], save_path: str | Path) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in sorted(metrics.items()):
            writer.writerow([key, value])
    return save_path


def main() -> None:
    args = parse_args()
    setup_logging()

    config = load_config(args.config)
    seed_everything(int(config["seed"]))

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    outputs_cfg = config["outputs"]

    device = _resolve_device(args.device)
    batch_size = int(args.batch_size or train_cfg["batch_size"])
    num_workers = int(train_cfg["num_workers"] if args.num_workers is None else args.num_workers)

    df_index = load_index_csv(data_cfg["index_csv"], event_col=data_cfg["event_col"])
    validation = validate_index_against_h5(
        df_index, data_cfg["h5_path"], event_col=data_cfg["event_col"]
    )
    if validation["missing"]:
        raise ValueError(f"Some event IDs were missing from HDF5: {validation['missing'][:10]}")

    all_ids = df_index[data_cfg["event_col"]].dropna().astype(str).str.strip().unique().tolist()
    train_ids, val_ids, test_ids = split_ids(
        all_ids,
        test_frac=float(data_cfg["test_frac"]),
        val_frac_of_train=float(data_cfg["val_frac_of_train"]),
        seed=int(config["seed"]),
    )
    assert_non_overlapping_splits(train_ids, val_ids, test_ids)

    if bool(train_cfg.get("use_all_data", False)):
        train_ids = sorted(set(train_ids + val_ids + test_ids))
        val_ids = []
        test_ids = []

    train_ds = VILSeq2SeqDataset(
        df_index=df_index,
        event_ids=train_ids,
        h5_path=data_cfg["h5_path"],
        tin=int(data_cfg["tin"]),
        tout=int(data_cfg["tout"]),
        stride=int(data_cfg["stride"]),
        use_sliding_windows=bool(data_cfg["use_sliding_windows"]),
        use_crops=bool(data_cfg["use_crops"]),
        crop_size=data_cfg.get("crop_size"),
        mode="train",
        normalize_divisor=float(data_cfg["normalize_divisor"]),
        event_col=data_cfg["event_col"],
        dataset_key=data_cfg["dataset_key"],
    )

    val_loader = None
    if val_ids:
        val_ds = VILSeq2SeqDataset(
            df_index=df_index,
            event_ids=val_ids,
            h5_path=data_cfg["h5_path"],
            tin=int(data_cfg["tin"]),
            tout=int(data_cfg["tout"]),
            stride=int(data_cfg["stride"]),
            use_sliding_windows=bool(data_cfg["use_sliding_windows"]),
            use_crops=False,
            crop_size=data_cfg.get("crop_size"),
            mode="val",
            normalize_divisor=float(data_cfg["normalize_divisor"]),
            event_col=data_cfg["event_col"],
            dataset_key=data_cfg["dataset_key"],
        )
        val_loader = build_dataloader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    train_loader = build_dataloader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    print(f"Device: {device}")
    print(
        f"Train events: {len(train_ids)} | Val events: {len(val_ids)} | Test events: {len(test_ids)}"
    )
    print(f"Train samples: {len(train_ds)}")
    if val_ids:
        print(f"Val samples: {len(val_ds)}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")

    model = ConvLSTMUNetSeq2Seq(
        base=int(model_cfg["base"]),
        bottleneck_ch=int(model_cfg["bottleneck_ch"]),
        tin=int(data_cfg["tin"]),
        tout=int(data_cfg["tout"]),
        dropout_p=float(model_cfg.get("dropout_p", 0.0)),
    ).to(device)

    criterion = get_loss(
        name=train_cfg["loss"]["name"],
        kwargs=train_cfg["loss"].get("kwargs", {}),
    )
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)
    scaler = torch.amp.GradScaler(enabled=bool(train_cfg["amp"]) and device.type == "cuda")

    run_dir = ensure_dir(Path(outputs_cfg["checkpoints_dir"]) / outputs_cfg["run_name"])
    results = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=int(train_cfg["epochs"]),
        output_dir=run_dir,
        config=config,
        scaler=scaler,
        scheduler=scheduler,
        amp_enabled=bool(train_cfg["amp"]),
        grad_accum=int(train_cfg["grad_accum"]),
        clip_grad_norm=train_cfg.get("clip_grad_norm"),
        log_every=int(train_cfg["log_every"]),
        early_patience=train_cfg.get("early_patience"),
        min_delta=float(train_cfg.get("min_delta", 1e-6)),
    )

    metrics: dict[str, object] = {"train": results}
    if test_ids:
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
        checkpoint = torch.load(results["best_path"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate_regression_metrics(
            model=model,
            loader=test_loader,
            device=device,
            amp_enabled=bool(train_cfg["amp"]),
        )
        metrics["test"] = test_metrics
        print("Test metrics:", test_metrics)

    metrics_dir = ensure_dir(outputs_cfg["metrics_dir"])
    metrics_json_path = metrics_dir / f"{outputs_cfg['run_name']}.json"
    metrics_csv_path = metrics_dir / f"{outputs_cfg['run_name']}.csv"
    metrics_json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if "test" in metrics and isinstance(metrics["test"], dict):
        _write_metrics_csv(metrics["test"], metrics_csv_path)

    if args.save_config_artifacts:
        save_resolved_config(
            config, metrics_dir / f"{outputs_cfg['run_name']}_resolved_config.yaml"
        )
        save_flat_config_csv(config, metrics_dir / f"{outputs_cfg['run_name']}_resolved_config.csv")

    print(f"Saved metrics JSON to {metrics_json_path}")
    if metrics_csv_path.exists():
        print(f"Saved metrics CSV to {metrics_csv_path}")


if __name__ == "__main__":
    main()
