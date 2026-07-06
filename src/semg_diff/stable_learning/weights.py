from __future__ import annotations

import numpy as np


def class_environment_balance_weights(
    labels: np.ndarray,
    env_ids: np.ndarray,
    max_weight: float = 10.0,
) -> np.ndarray:
    """Balance each class across source environments using inverse class-env frequency.

    The weight is learned from source data only by callers. Within each gesture class,
    each environment contributes the same total weight, which reduces domination by
    subjects with more windows while preserving class totals.
    """
    labels = np.asarray(labels)
    env_ids = np.asarray(env_ids)
    if labels.shape[0] != env_ids.shape[0]:
        raise ValueError("labels and env_ids must have the same length")

    weights = np.ones(labels.shape[0], dtype=np.float32)
    for label in np.unique(labels):
        class_mask = labels == label
        class_count = int(class_mask.sum())
        envs = np.unique(env_ids[class_mask])
        if class_count == 0 or envs.size == 0:
            continue
        target_env_mass = class_count / envs.size
        for env in envs:
            mask = class_mask & (env_ids == env)
            count = int(mask.sum())
            if count > 0:
                weights[mask] = target_env_mass / count

    weights = weights / np.mean(weights)
    return np.clip(weights, 1.0 / max_weight, max_weight).astype(np.float32)
