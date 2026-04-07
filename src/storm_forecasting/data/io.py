from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

DEFAULT_DATA_FILENAMES = ("train.h5", "events.csv")
DEFAULT_IMG_TYPE_CANDIDATES = ("img_type", "data_type", "product", "dataset")
DEFAULT_EVENT_COL_CANDIDATES = ("id", "event_id")


def _resolve_hf_token(token: str | None = None) -> str | None:
    return token or os.getenv("HF_TOKEN")


def download_dataset_files(
    repo_id: str,
    local_dir: str | Path,
    filenames: Iterable[str] = DEFAULT_DATA_FILENAMES,
    token: str | None = None,
) -> dict[str, Path]:
    """Download raw dataset files from the Hugging Face Hub.

    This function is idempotent: repeated runs reuse the local cache and only fetch
    files if they are missing or stale according to the Hugging Face cache.
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    token = _resolve_hf_token(token)

    outputs: dict[str, Path] = {}
    for filename in filenames:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(local_dir),
            token=token,
        )
        outputs[filename] = Path(path)
    return outputs


def load_events_metadata(
    path: str | Path,
    parse_dates: tuple[str, ...] = ("start_utc",),
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"events metadata not found: {path}")

    df = pd.read_csv(path)
    for col in parse_dates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def infer_event_col(
    df: pd.DataFrame, candidates: Iterable[str] = DEFAULT_EVENT_COL_CANDIDATES
) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"Could not find an event id column. Tried: {tuple(candidates)}")


def infer_img_type_col(
    df: pd.DataFrame, candidates: Iterable[str] = DEFAULT_IMG_TYPE_CANDIDATES
) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"Could not find an image type column. Tried: {tuple(candidates)}")


def filter_vil_events(
    df: pd.DataFrame,
    img_type_col: str | None = None,
    vil_value: str = "vil",
) -> pd.DataFrame:
    img_type_col = img_type_col or infer_img_type_col(df)
    out = df[df[img_type_col].astype(str).str.strip().str.lower() == vil_value.lower()].copy()
    return out.reset_index(drop=True)


def save_index_csv(df: pd.DataFrame, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def load_index_csv(path: str | Path, event_col: str = "id") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Index CSV not found: {path}. Run the dataset bootstrap step first to create it."
        )
    df = pd.read_csv(path)
    if event_col not in df.columns:
        raise KeyError(f"Missing event column: {event_col}")
    df[event_col] = df[event_col].astype(str).str.strip()
    return df


def list_h5_event_ids(h5_path: str | Path) -> list[str]:
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 data file not found: {h5_path}")
    with h5py.File(h5_path, "r") as handle:
        return sorted(handle.keys())


def get_event_length(h5_path: str | Path, event_id: str, dataset_key: str = "vil") -> int:
    with h5py.File(h5_path, "r") as handle:
        arr = handle[str(event_id)][dataset_key]
        shape = arr.shape
    if len(shape) != 3:
        raise ValueError(f"Expected 3D array for {event_id}, got shape {shape}")
    if shape[0] == shape[1]:
        return int(shape[2])
    return int(shape[0])


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
