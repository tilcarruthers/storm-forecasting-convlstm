from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from storm_forecasting.data.windowing import count_windows


@dataclass(frozen=True)
class SampleIndex:
    event_id: str
    t0: int


class VILSeq2SeqDataset(Dataset):
    """Lazy HDF5-backed dataset for direct multi-step forecasting."""

    def __init__(
        self,
        df_index: pd.DataFrame,
        event_ids: list[str],
        h5_path: str | Path,
        tin: int,
        tout: int,
        stride: int = 1,
        use_sliding_windows: bool = True,
        use_crops: bool = False,
        crop_size: int | None = None,
        mode: str = "train",
        normalize_divisor: float = 255.0,
        event_col: str = "id",
        dataset_key: str = "vil",
    ) -> None:
        self.df_index = df_index.copy()
        self.event_ids = [str(event_id).strip() for event_id in event_ids]
        self.h5_path = str(h5_path)
        self.tin = tin
        self.tout = tout
        self.stride = stride
        self.use_sliding_windows = use_sliding_windows
        self.use_crops = use_crops
        self.crop_size = crop_size
        self.mode = mode
        self.normalize_divisor = normalize_divisor
        self.event_col = event_col
        self.dataset_key = dataset_key

        self._h5: h5py.File | None = None
        self.samples = self._build_sample_index()

    def _open_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self) -> None:
        self.close()

    @staticmethod
    def _shape_to_sequence_length(shape: tuple[int, ...], event_id: str) -> int:
        if len(shape) != 3:
            raise ValueError(f"Expected 3D sequence for {event_id}, got {shape}")
        if shape[0] == shape[1]:
            return int(shape[2])
        return int(shape[0])

    def _build_sample_index(self) -> list[SampleIndex]:
        samples: list[SampleIndex] = []
        with h5py.File(self.h5_path, "r") as handle:
            for event_id in self.event_ids:
                shape = handle[str(event_id)][self.dataset_key].shape
                n_steps = self._shape_to_sequence_length(shape, event_id)
                if self.use_sliding_windows:
                    n_windows = count_windows(n_steps, self.tin, self.tout, self.stride)
                    if n_windows == 0:
                        continue
                    for t0 in range(0, max(0, n_steps - (self.tin + self.tout)) + 1, self.stride):
                        samples.append(SampleIndex(event_id=event_id, t0=t0))
                else:
                    if n_steps < self.tin + self.tout:
                        continue
                    samples.append(SampleIndex(event_id=event_id, t0=0))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _crop(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.use_crops or self.crop_size is None:
            return x, y

        h, w = x.shape[1], x.shape[2]
        crop = self.crop_size
        if crop >= h or crop >= w:
            return x, y

        if self.mode == "train":
            top = np.random.randint(0, h - crop + 1)
            left = np.random.randint(0, w - crop + 1)
        else:
            top = (h - crop) // 2
            left = (w - crop) // 2

        return (
            x[:, top : top + crop, left : left + crop],
            y[:, top : top + crop, left : left + crop],
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | str]]:
        sample = self.samples[idx]
        handle = self._open_h5()
        arr = handle[str(sample.event_id)][self.dataset_key][:]

        vil = np.asarray(arr)
        if vil.ndim != 3:
            raise ValueError(f"Expected 3D data, got {vil.shape} for event {sample.event_id}")
        if vil.shape[0] == vil.shape[1]:
            vil = np.transpose(vil, (2, 0, 1))

        x = vil[sample.t0 : sample.t0 + self.tin].astype(np.float32)
        y = vil[sample.t0 + self.tin : sample.t0 + self.tin + self.tout].astype(np.float32)

        x, y = self._crop(x, y)
        x = x / self.normalize_divisor
        y = y / self.normalize_divisor

        x_tensor = torch.from_numpy(x).unsqueeze(1).float()
        y_tensor = torch.from_numpy(y).unsqueeze(1).float()

        meta = {"event_id": sample.event_id, "t0": sample.t0}
        return x_tensor, y_tensor, meta


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
