from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download


def download_dataset_files(
    repo_id: str,
    local_dir: str | Path,
    filenames: Iterable[str] = ("train.h5", "events.csv"),
    token: str | None = None,
) -> list[Path]:
    """Download raw dataset files from the Hugging Face Hub."""
    token = token or os.getenv("HF_TOKEN")
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for filename in filenames:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(local_dir),
            token=token,
        )
        outputs.append(Path(path))
    return outputs


def load_events_metadata(
    path: str | Path,
    parse_dates: tuple[str, ...] = ("start_utc",),
) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    for col in parse_dates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def filter_vil_events(
    df: pd.DataFrame,
    img_type_col: str = "img_type",
    vil_value: str = "vil",
) -> pd.DataFrame:
    if img_type_col not in df.columns:
        raise KeyError(f"Missing image type column: {img_type_col}")
    out = df[df[img_type_col] == vil_value].copy()
    return out.reset_index(drop=True)


def save_index_csv(df: pd.DataFrame, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def load_index_csv(path: str | Path, event_col: str = "id") -> pd.DataFrame:
    df = pd.read_csv(path)
    if event_col not in df.columns:
        raise KeyError(f"Missing event column: {event_col}")
    df[event_col] = df[event_col].astype(str).str.strip()
    return df


def list_h5_event_ids(h5_path: str | Path) -> list[str]:
    with h5py.File(h5_path, "r") as handle:
        return sorted(handle.keys())


def get_event_length(h5_path: str | Path, event_id: str, dataset_key: str = "vil") -> int:
    with h5py.File(h5_path, "r") as handle:
        arr = handle[str(event_id)][dataset_key]
        shape = arr.shape
    if len(shape) != 3:
        raise ValueError(f"Expected 3D array for {event_id}, got shape {shape}")
    if shape[0] == shape[1]:
        return int(shape[2])  # (H, W, T)
    return int(shape[0])  # already (T, H, W)


def load_vil_array_from_h5(
    h5_path: str | Path,
    event_id: str,
    dataset_key: str = "vil",
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Load one storm sequence as (T, H, W)."""
    with h5py.File(h5_path, "r") as handle:
        arr = handle[str(event_id)][dataset_key][:]

    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D VIL array, got {arr.shape} for event {event_id}")

    # Notebook data was stored as (H, W, T)
    if arr.shape[0] == arr.shape[1]:
        arr = np.transpose(arr, (2, 0, 1))

    return arr.astype(dtype)


def validate_index_against_h5(
    df_index: pd.DataFrame,
    h5_path: str | Path,
    event_col: str = "id",
) -> dict[str, list[str]]:
    csv_ids = df_index[event_col].dropna().astype(str).str.strip().unique().tolist()
    h5_ids = set(list_h5_event_ids(h5_path))
    missing = [event_id for event_id in csv_ids if event_id not in h5_ids]
    present = [event_id for event_id in csv_ids if event_id in h5_ids]
    return {"present": present, "missing": missing}
