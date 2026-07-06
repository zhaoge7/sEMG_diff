from __future__ import annotations

import numpy as np


def mean_absolute_value(x: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(x), axis=1)


def root_mean_square(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(x), axis=1))


def waveform_length(x: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(np.diff(x, axis=1)), axis=1)


def zero_crossings(x: np.ndarray, threshold: float) -> np.ndarray:
    left = x[:, :-1, :]
    right = x[:, 1:, :]
    sign_change = (left * right) < 0
    enough_delta = np.abs(left - right) >= threshold
    return np.sum(sign_change & enough_delta, axis=1)


def slope_sign_changes(x: np.ndarray, threshold: float) -> np.ndarray:
    prev = x[:, :-2, :]
    mid = x[:, 1:-1, :]
    nxt = x[:, 2:, :]
    left = mid - prev
    right = mid - nxt
    return np.sum((left * right > 0) & ((np.abs(left) >= threshold) | (np.abs(right) >= threshold)), axis=1)


def variance(x: np.ndarray) -> np.ndarray:
    return np.var(x, axis=1, ddof=1)


def willison_amplitude(x: np.ndarray, threshold: float) -> np.ndarray:
    return np.sum(np.abs(np.diff(x, axis=1)) >= threshold, axis=1)


FEATURE_FUNCS = {
    "mav": lambda x, cfg: mean_absolute_value(x),
    "rms": lambda x, cfg: root_mean_square(x),
    "wl": lambda x, cfg: waveform_length(x),
    "zc": lambda x, cfg: zero_crossings(x, float(cfg.get("zc_threshold", 0.0005))),
    "ssc": lambda x, cfg: slope_sign_changes(x, float(cfg.get("ssc_threshold", 0.0005))),
    "var": lambda x, cfg: variance(x),
    "wamp": lambda x, cfg: willison_amplitude(x, float(cfg.get("wamp_threshold", 0.005))),
}


def extract_time_domain_features(
    windows: np.ndarray,
    enabled: list[str],
    feature_cfg: dict[str, float] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Extract channel-wise time-domain features from (n, time, channels) windows."""
    if windows.ndim != 3:
        raise ValueError(f"Expected windows shape (n, time, channels), got {windows.shape}")
    cfg = feature_cfg or {}
    features: list[np.ndarray] = []
    names: list[str] = []
    channels = windows.shape[-1]
    for feature_name in enabled:
        if feature_name not in FEATURE_FUNCS:
            raise KeyError(f"Unknown time-domain feature: {feature_name}")
        values = FEATURE_FUNCS[feature_name](windows, cfg)
        features.append(values)
        names.extend([f"ch{ch + 1}_{feature_name}" for ch in range(channels)])
    return np.concatenate(features, axis=1).astype(np.float32), names
