from __future__ import annotations

import torch
import torch.nn as nn


class WeightedMAELoss(nn.Module):
    """Piecewise weighted MAE using target intensity thresholds."""

    def __init__(
        self,
        thresholds: tuple[float, ...] = (0.25, 0.5, 0.75),
        weights: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0),
    ) -> None:
        super().__init__()
        if len(weights) != len(thresholds) + 1:
            raise ValueError("weights must be exactly one longer than thresholds")
        self.thresholds = thresholds
        self.weights = weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        abs_err = (pred - target).abs()
        weight_map = torch.full_like(target, self.weights[0])

        for threshold, weight in zip(self.thresholds, self.weights[1:]):
            weight_map = torch.where(target >= threshold, torch.full_like(weight_map, weight), weight_map)

        return (abs_err * weight_map).sum() / weight_map.sum().clamp_min(1e-12)


def get_loss(name: str, kwargs: dict | None = None) -> nn.Module:
    kwargs = kwargs or {}
    normalized = name.strip().lower()

    if normalized in {"mae", "l1", "l1loss"}:
        return nn.L1Loss()
    if normalized in {"weighted_mae", "weighted-l1"}:
        return WeightedMAELoss(**kwargs)

    raise ValueError(f"Unsupported loss name: {name}")
