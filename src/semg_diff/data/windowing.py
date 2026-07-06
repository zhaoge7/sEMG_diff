from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from semg_diff.data.db2 import DB2Recording


@dataclass(frozen=True)
class WindowConfig:
    sampling_rate_hz: int = 2000
    window_ms: int = 200
    step_ms: int = 100
    label_majority_threshold: float = 0.90
    include_rest: bool = False

    @property
    def window_samples(self) -> int:
        return int(round(self.sampling_rate_hz * self.window_ms / 1000.0))

    @property
    def step_samples(self) -> int:
        return int(round(self.sampling_rate_hz * self.step_ms / 1000.0))


def majority_label(labels: np.ndarray) -> tuple[int, float]:
    values, counts = np.unique(labels, return_counts=True)
    idx = int(np.argmax(counts))
    return int(values[idx]), float(counts[idx] / labels.size)


def slice_recording_windows(
    recording: DB2Recording,
    emg: np.ndarray,
    cfg: WindowConfig,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Slice one recording into windows and metadata rows."""
    if emg.shape[0] != recording.labels.shape[0]:
        raise ValueError("EMG and label lengths differ")

    win = cfg.window_samples
    step = cfg.step_samples
    if win <= 0 or step <= 0:
        raise ValueError(f"Invalid window/step samples: {win}/{step}")
    if emg.shape[0] < win:
        return np.empty((0, win, emg.shape[1]), dtype=np.float32), pd.DataFrame()

    windows: list[np.ndarray] = []
    rows: list[dict[str, int | float]] = []

    for start in range(0, emg.shape[0] - win + 1, step):
        end = start + win
        label, ratio = majority_label(recording.labels[start:end])
        if label == 0 and not cfg.include_rest:
            continue
        if ratio < cfg.label_majority_threshold:
            continue

        repetition, _ = majority_label(recording.repetitions[start:end])
        windows.append(emg[start:end].astype(np.float32, copy=False))
        rows.append(
            {
                "subject": recording.subject,
                "exercise": recording.exercise,
                "repetition": repetition,
                "label": label,
                "start": start,
                "end": end,
                "majority_ratio": ratio,
                "env_id": recording.subject,
                "class_id": label,
                "sample_weight": 1.0,
            }
        )

    if not windows:
        return np.empty((0, win, emg.shape[1]), dtype=np.float32), pd.DataFrame(rows)
    return np.stack(windows).astype(np.float32), pd.DataFrame(rows)


def concatenate_window_sets(
    parts: list[tuple[np.ndarray, pd.DataFrame]],
) -> tuple[np.ndarray, pd.DataFrame]:
    arrays = [arr for arr, meta in parts if arr.shape[0] > 0]
    metas = [meta for arr, meta in parts if arr.shape[0] > 0 and not meta.empty]
    if not arrays:
        raise ValueError("No windows were generated")
    windows = np.concatenate(arrays, axis=0).astype(np.float32)
    metadata = pd.concat(metas, ignore_index=True)
    return windows, metadata
