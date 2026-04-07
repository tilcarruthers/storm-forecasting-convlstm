# Storm Forecasting with ConvLSTM Seq2Seq

A clean, reproducible PyTorch refactor of an Imperial College MSc storm forecasting coursework project. The task is to use 12 past VIL (Vertically Integrated Liquid) radar frames to predict the next 12 frames at a 5-minute cadence.

The original coursework objective was to optimise **MAE / L1 loss** because that was the competition metric. This refactor keeps that baseline intact, then creates a modular codebase that makes it straightforward to add more careful evaluation later, including weighted MAE, SSIM, and a simple Monte Carlo dropout uncertainty analysis.

## Why this repository exists

The notebook version worked for the coursework and competition, but it mixed together:

- EDA
- data indexing
- HDF5 I/O
- dataset construction
- model definition
- training loops
- evaluation and plotting

This repository extracts those concerns into a proper Python package so the project can be used as a portfolio piece for ML research and ML engineering roles.

## Problem setup

- **Input:** 12 historical VIL frames
- **Output:** 12 future VIL frames
- **Cadence:** 5 minutes per frame
- **Task type:** direct multi-step spatiotemporal forecasting
- **Model family:** ConvLSTM bottleneck U-Net seq2seq model in PyTorch

## Modelling decisions

### Direct multi-step forecasting instead of autoregressive rollout

This project predicts the next 12 frames in one forward pass rather than feeding predictions back into the model autoregressively. That design choice was deliberate:

- storm evolution changes materially across the forecast horizon
- repeated autoregressive rollout can compound error
- a direct objective better matches the downstream evaluation setting for the coursework

### MAE / L1 as the original optimisation target

The original competition was scored with MAE, so the baseline model is trained with L1 loss. That is preserved here to keep the benchmark honest.

### Future evaluation extensions

Weighted MAE, SSIM, and Monte Carlo dropout are included as modular evaluation utilities so they can be investigated without rewriting the whole repo. They should be interpreted as **supplementary analysis**, not as retroactive changes to the original benchmark.

## Repository layout

```text
storm-forecasting-convlstm/
├─ configs/
├─ data/
├─ docs/
├─ notebooks/
├─ outputs/
├─ reports/
├─ scripts/
├─ src/storm_forecasting/
└─ tests/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Data bootstrap

The raw coursework dataset is hosted on Hugging Face and requires authentication.

### Option 1: environment variable

```bash
export HF_TOKEN="your_token_here"
```

### Option 2: Hugging Face CLI login

```bash
huggingface-cli login
```

Then bootstrap the data and build the VIL-only index:

```bash
make download-data
```

or:

```bash
./scripts/bootstrap_data.sh
```

This downloads or reuses cached copies of:

- `data/events.csv`
- `data/train.h5`

and builds:

- `data/vil_events.csv`

No raw data or secrets are committed to the repository.

## Core commands

### Train baseline

```bash
python -m storm_forecasting.cli.train \
  --config configs/experiments/baseline_reproduction.yaml \
  --save-config-artifacts
```

### Evaluate baseline

```bash
python -m storm_forecasting.cli.evaluate \
  --config configs/experiments/baseline_reproduction.yaml \
  --checkpoint outputs/checkpoints/best_competition_model.pt \
  --save-config-artifacts
```

### Predict qualitative examples

```bash
python -m storm_forecasting.cli.predict \
  --config configs/experiments/baseline_reproduction.yaml \
  --checkpoint outputs/checkpoints/baseline_reproduction/best.pt \
  --index 0
```

## Useful CLI arguments

### `storm_forecasting.cli.make_dataset_index`

- `--config`: load paths and dataset metadata from an experiment config
- `--download`: download raw files before building the index
- `--repo-id`: override the Hugging Face dataset repo
- `--local-dir`: override the download location
- `--events-csv`: explicit path to raw metadata CSV
- `--output-csv`: explicit path to the generated VIL-only index
- `--img-type-col`: override the modality column used for filtering
- `--vil-value`: override the value treated as VIL

### `storm_forecasting.cli.train`

- `--config`: experiment config
- `--device`: override device selection, e.g. `cpu` or `cuda`
- `--num-workers`: dataloader worker override
- `--batch-size`: dataloader batch-size override
- `--save-config-artifacts`: save the fully resolved config as YAML and a flattened CSV table

### `storm_forecasting.cli.evaluate`

- `--config`: experiment config
- `--checkpoint`: checkpoint to evaluate
- `--device`: override device selection
- `--num-workers`: dataloader worker override
- `--batch-size`: dataloader batch-size override
- `--max-batches`: smoke-test the evaluation loop on a small subset
- `--save-config-artifacts`: save the fully resolved config as YAML and a flattened CSV table

## Outputs and reproducibility

Training and evaluation can now save:

- resolved config YAML files
- flattened config CSV tables
- JSON metric summaries
- CSV metric summaries
- per-horizon MAE CSV files
- qualitative figures under `outputs/figures/`

This makes runs easier to inspect and compare without going back to notebook state.

## What is implemented now

- HDF5 loading for per-storm VIL arrays
- sliding-window sequence generation
- storm-wise train/val/test split
- lazy PyTorch dataset
- ConvLSTM bottleneck U-Net seq2seq model
- MAE baseline training
- overall MAE / MSE / RMSE evaluation
- per-horizon error curves
- qualitative panels and GIF generation
- optional weighted MAE / SSIM / MC dropout utilities
- reproducible data bootstrap from Hugging Face
- config export to YAML / CSV for train and eval runs

## What this project does **not** claim

This repository should not be framed as:

- an operational forecasting platform
- a production geospatial MLOps system
- a calibrated probabilistic weather service
- a full remote sensing stack

It is a clean research-engineering refactor of a spatiotemporal forecasting project using VIL imagery in PyTorch.

## Immediate next steps

- run full baseline evaluation on desktop / GPU hardware
- add baseline result figures and tables to this README
- evaluate weighted MAE and SSIM on the saved competition checkpoint
- then run controlled retraining experiments if those metrics prove informative
- add GitHub Actions CI alongside the existing tests and pre-commit hooks

## Suggested first workflow on a fresh clone

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
export HF_TOKEN="your_token_here"
make download-data
pytest
python -m storm_forecasting.cli.evaluate \
  --config configs/experiments/baseline_reproduction.yaml \
  --checkpoint outputs/checkpoints/best_competition_model.pt \
  --save-config-artifacts
```
