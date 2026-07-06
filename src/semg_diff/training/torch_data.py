from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class EMGWindowDataset(Dataset):
    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        env_ids: np.ndarray | None = None,
    ) -> None:
        if windows.ndim != 3:
            raise ValueError(f"Expected windows shape (n, time, channels), got {windows.shape}")
        self.windows = windows.astype(np.float32, copy=False)
        self.labels = labels.astype(np.int64, copy=False)
        self.env_ids = (
            env_ids.astype(np.int64, copy=False) if env_ids is not None else np.zeros(len(labels), dtype=np.int64)
        )

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.windows[idx].T.copy())
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        env = torch.tensor(self.env_ids[idx], dtype=torch.long)
        return x, y, env
