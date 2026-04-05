from __future__ import annotations

import argparse

from storm_forecasting.data.io import (
    download_dataset_files,
    filter_vil_events,
    load_events_metadata,
    save_index_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a VIL-only index CSV for Task 1.")
    parser.add_argument("--events-csv", type=str, required=True, help="Path to raw events.csv.")
    parser.add_argument("--output-csv", type=str, required=True, help="Path to VIL-only output CSV.")
    parser.add_argument("--download", action="store_true", help="Download train.h5 and events.csv first.")
    parser.add_argument("--repo-id", type=str, default="benmoseley/ese-dl-2025-26-group-project")
    parser.add_argument("--local-dir", type=str, default="data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.download:
        download_dataset_files(repo_id=args.repo_id, local_dir=args.local_dir)

    events_df = load_events_metadata(args.events_csv)
    vil_df = filter_vil_events(events_df, img_type_col="img_type", vil_value="vil")
    out_path = save_index_csv(vil_df, args.output_csv)
    print(f"Saved VIL-only index to {out_path.resolve()}")


if __name__ == "__main__":
    main()
