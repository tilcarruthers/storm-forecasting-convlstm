from __future__ import annotations

from typing import Any

import torch


def build_optimizer(model: torch.nn.Module, training_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    name = str(training_cfg.get("optimizer", "adamw")).lower()
    lr = float(training_cfg["lr"])
    weight_decay = float(training_cfg.get("weight_decay", 0.0))

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_cfg: dict[str, Any],
) -> torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    name = training_cfg.get("scheduler")
    if name is None:
        return None

    normalized = str(name).lower()
    if normalized in {"reduce_on_plateau", "plateau"}:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(training_cfg.get("lr_factor", 0.5)),
            patience=int(training_cfg.get("lr_patience", 2)),
        )

    raise ValueError(f"Unsupported scheduler: {name}")
