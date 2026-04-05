from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from storm_forecasting.training.checkpoints import save_checkpoint


def _amp_enabled(device: torch.device, requested: bool) -> bool:
    return requested and device.type == "cuda"


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    model.eval()
    total = 0.0
    n_samples = 0

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=_amp_enabled(device, amp_enabled)):
            y_hat = model(x)
            loss = criterion(y_hat, y)

        total += float(loss.detach()) * x.size(0)
        n_samples += x.size(0)

    return total / max(n_samples, 1)


@torch.no_grad()
def evaluate_regression_metrics(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, float]:
    model.eval()
    mae_total = 0.0
    mse_total = 0.0
    n_samples = 0

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=_amp_enabled(device, amp_enabled)):
            y_hat = model(x)

        mae_total += float(F.l1_loss(y_hat, y, reduction="mean")) * x.size(0)
        mse_total += float(F.mse_loss(y_hat, y, reduction="mean")) * x.size(0)
        n_samples += x.size(0)

    mae = mae_total / max(n_samples, 1)
    mse = mse_total / max(n_samples, 1)
    rmse = float(np.sqrt(mse))
    return {"mae": mae, "mse": mse, "rmse": rmse}


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    criterion: torch.nn.Module,
    device: torch.device,
    grad_accum: int = 1,
    amp_enabled: bool = False,
    clip_grad_norm: float | None = None,
    log_every: int = 50,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total = 0.0
    n_samples = 0

    progress = tqdm(enumerate(loader), total=len(loader), desc="train", leave=False)
    for step, (x, y, _) in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=_amp_enabled(device, amp_enabled)):
            y_hat = model(x)
            loss = criterion(y_hat, y) / grad_accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        should_step = (step + 1) % grad_accum == 0
        if should_step:
            if clip_grad_norm is not None:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        loss_item = float(loss.detach()) * grad_accum
        total += loss_item * x.size(0)
        n_samples += x.size(0)

        if (step + 1) % log_every == 0:
            progress.set_postfix({"mae": f"{loss_item:.4f}"})

    if len(loader) > 0 and len(loader) % grad_accum != 0:
        if clip_grad_norm is not None:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total / max(n_samples, 1)


def fit(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader | None,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    output_dir: str | Path,
    config: dict[str, Any],
    scaler: torch.amp.GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | torch.optim.lr_scheduler._LRScheduler | None = None,
    amp_enabled: bool = False,
    grad_accum: int = 1,
    clip_grad_norm: float | None = None,
    log_every: int = 50,
    early_patience: int | None = None,
    min_delta: float = 1e-6,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }
    best_val = float("inf")
    best_metrics: dict[str, Any] = {}
    bad_epochs = 0

    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            device=device,
            grad_accum=grad_accum,
            amp_enabled=amp_enabled,
            clip_grad_norm=clip_grad_norm,
            log_every=log_every,
        )

        history["train_loss"].append(train_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history["lr"].append(current_lr)

        if val_loader is not None:
            val_loss = evaluate_loss(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                amp_enabled=amp_enabled,
            )
        else:
            val_loss = train_loss

        history["val_loss"].append(val_loss)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        epoch_metrics = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            config=config,
            metrics=epoch_metrics,
        )

        improved = val_loss < (best_val - min_delta)
        if improved:
            best_val = val_loss
            best_metrics = epoch_metrics
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                config=config,
                metrics=epoch_metrics,
            )
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(
            f"Epoch {epoch:02d}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f} | "
            f"lr={current_lr:.2e}"
        )

        if early_patience is not None and bad_epochs >= early_patience:
            print(f"Early stopping triggered after {bad_epochs} non-improving epochs.")
            break

    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    return {
        "history": history,
        "best_val_loss": best_val,
        "best_path": str(best_path),
        "last_path": str(last_path),
        "best_metrics": best_metrics,
    }
