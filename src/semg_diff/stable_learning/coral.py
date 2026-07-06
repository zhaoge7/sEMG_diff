from __future__ import annotations

import numpy as np


def _covariance(x: np.ndarray, eps: float) -> np.ndarray:
    cov = np.cov(x, rowvar=False)
    return cov + np.eye(cov.shape[0], dtype=cov.dtype) * eps


def _matrix_power_symmetric(matrix: np.ndarray, power: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, 1e-12, None)
    return (vectors * np.power(values, power)) @ vectors.T


def coral_align_source_to_target(
    source_x: np.ndarray,
    target_x: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Unsupervised CORAL transform using source features and unlabeled target features."""
    source_mean = source_x.mean(axis=0, keepdims=True)
    target_mean = target_x.mean(axis=0, keepdims=True)
    centered_source = source_x - source_mean
    source_cov = _covariance(centered_source, eps)
    target_cov = _covariance(target_x - target_mean, eps)
    transform = _matrix_power_symmetric(source_cov, -0.5) @ _matrix_power_symmetric(target_cov, 0.5)
    return (centered_source @ transform + target_mean).astype(np.float32)
