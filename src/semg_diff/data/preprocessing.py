from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class FilterConfig:
    sampling_rate_hz: int = 2000
    bandpass_hz: tuple[float, float] = (20.0, 450.0)
    notch_hz: float | None = 50.0
    notch_quality: float = 30.0
    enabled: bool = True
    rectify: bool = False


def preprocess_emg(emg: np.ndarray, cfg: FilterConfig) -> np.ndarray:
    """Apply per-channel filtering and optional rectification."""
    x = np.asarray(emg, dtype=np.float32)
    if not cfg.enabled:
        return np.abs(x) if cfg.rectify else x.copy()

    nyquist = cfg.sampling_rate_hz / 2.0
    low, high = cfg.bandpass_hz
    sos = signal.butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    filtered = signal.sosfiltfilt(sos, x, axis=0).astype(np.float32)

    if cfg.notch_hz:
        b, a = signal.iirnotch(cfg.notch_hz / nyquist, cfg.notch_quality)
        filtered = signal.filtfilt(b, a, filtered, axis=0).astype(np.float32)

    if cfg.rectify:
        filtered = np.abs(filtered)
    return filtered


@dataclass
class ChannelStandardScaler:
    mean_: np.ndarray
    scale_: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.scale_


def fit_source_scaler(windows: np.ndarray, source_mask: np.ndarray):
    """Fit a channel-wise scaler on source-training windows only."""
    if windows.ndim != 3:
        raise ValueError(f"Expected windows as (n, time, channels), got {windows.shape}")
    source_windows = windows[source_mask]
    if source_windows.size == 0:
        raise ValueError("Cannot fit scaler: source mask selected no windows")
    flat = source_windows.reshape(-1, source_windows.shape[-1])
    try:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(flat)
        return scaler
    except ModuleNotFoundError:
        mean = flat.mean(axis=0, keepdims=True)
        scale = flat.std(axis=0, keepdims=True)
        scale[scale == 0] = 1.0
        return ChannelStandardScaler(mean_=mean, scale_=scale)


def transform_windows_with_scaler(windows: np.ndarray, scaler) -> np.ndarray:
    n, t, c = windows.shape
    transformed = scaler.transform(windows.reshape(-1, c)).reshape(n, t, c)
    return transformed.astype(np.float32)
