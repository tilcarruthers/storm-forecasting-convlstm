#!/usr/bin/env bash
set -euo pipefail

python -m storm_forecasting.cli.evaluate   --config configs/experiments/baseline_reproduction.yaml
