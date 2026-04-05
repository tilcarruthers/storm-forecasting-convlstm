# Methodology notes

This repository is a refactor of an MSc coursework / competition project for 12-step storm nowcasting on VIL imagery.

## Task framing
- Input: 12 past VIL frames
- Output: next 12 VIL frames
- Cadence: 5 minutes
- Baseline objective: MAE / L1, aligned with competition scoring

## Model choice
The model is a ConvLSTM bottleneck U-Net:
- shared 2D encoder across input frames
- ConvLSTM encoder over latent sequence
- zero-input ConvLSTM decoder for future latent states
- U-Net style decoder back to image space

## Why direct forecasting
The project uses direct multi-output prediction instead of autoregressive rollout because storm evolution changes significantly across the horizon and autoregressive feedback can accumulate error.

## Evaluation extensions
The codebase includes optional weighted MAE, SSIM, and Monte Carlo dropout utilities for future analysis. Those should be treated as extensions rather than retroactive redefinitions of the original benchmark.
