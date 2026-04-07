#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Export it before running this script."
  echo 'Example: export HF_TOKEN="your_token_here"'
  exit 1
fi

python -m storm_forecasting.cli.make_dataset_index \
  --config configs/experiments/baseline_reproduction.yaml \
  --download
