from __future__ import annotations

import argparse

from storm_forecasting.config import load_config
from storm_forecasting.data.io import (
    download_dataset_files,
    filter_vil_events,
    infer_event_col,
    infer_img_type_col,
    load_events_metadata,
    save_index_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a VIL-only index CSV for Task 1.")
    parser.add_argument("--config", type=str, default=None, help="Optional experiment config YAML.")
    parser.add_argument("--events-csv", type=str, default=None, help="Path to raw events.csv.")
    parser.add_argument("--output-csv", type=str, default=None, help="Path to VIL-only output CSV.")
    parser.add_argument(
        "--download", action="store_true", help="Download train.h5 and events.csv first."
    )
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--local-dir", type=str, default=None)
    parser.add_argument("--img-type-col", type=str, default=None)
    parser.add_argument("--vil-value", type=str, default="vil")
    return parser.parse_args()


def resolve_args(args: argparse.Namespace) -> dict[str, str | None]:
    resolved = {
        "events_csv": args.events_csv,
        "output_csv": args.output_csv,
        "repo_id": args.repo_id,
        "local_dir": args.local_dir,
        "img_type_col": args.img_type_col,
        "event_col": None,
    }

    if args.config:
        config = load_config(args.config)
        data_cfg = config["data"]
        resolved["events_csv"] = resolved["events_csv"] or data_cfg.get("events_csv")
        resolved["output_csv"] = resolved["output_csv"] or data_cfg.get("index_csv")
        resolved["repo_id"] = resolved["repo_id"] or data_cfg.get("repo_id")
        resolved["local_dir"] = resolved["local_dir"] or data_cfg.get("data_dir")
        resolved["img_type_col"] = resolved["img_type_col"] or data_cfg.get("img_type_col")
        resolved["event_col"] = data_cfg.get("event_col")

    if resolved["events_csv"] is None or resolved["output_csv"] is None:
        raise ValueError("events_csv and output_csv must be provided directly or via --config")

    if args.download and (resolved["repo_id"] is None or resolved["local_dir"] is None):
        raise ValueError("repo_id and local_dir must be provided when using --download")

    return resolved


def main() -> None:
    args = parse_args()
    resolved = resolve_args(args)

    if args.download:
        downloaded = download_dataset_files(
            repo_id=str(resolved["repo_id"]), local_dir=str(resolved["local_dir"])
        )
        print("Downloaded or reused cached dataset files:")
        for name, path in downloaded.items():
            print(f"  - {name}: {path}")

    events_df = load_events_metadata(str(resolved["events_csv"]))
    img_type_col = resolved["img_type_col"] or infer_img_type_col(events_df)
    event_col = resolved["event_col"] or infer_event_col(events_df)

    vil_df = filter_vil_events(events_df, img_type_col=str(img_type_col), vil_value=args.vil_value)
    out_path = save_index_csv(vil_df, str(resolved["output_csv"]))

    print(f"Saved VIL-only index to {out_path.resolve()}")
    print(f"Rows written: {len(vil_df)}")
    print(f"Unique events: {vil_df[event_col].astype(str).nunique()}")
    print(f"Filter used: {img_type_col} == {args.vil_value!r}")


if __name__ == "__main__":
    main()
