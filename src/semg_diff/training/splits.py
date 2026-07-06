from __future__ import annotations

import numpy as np


def loso_subjects(subjects: np.ndarray) -> list[int]:
    return sorted(np.unique(subjects).astype(int).tolist())


def source_validation_subjects(
    source_subjects: list[int],
    val_fraction: float,
    seed: int,
) -> list[int]:
    if len(source_subjects) < 3:
        return [source_subjects[-1]]
    rng = np.random.default_rng(seed)
    shuffled = np.array(source_subjects, dtype=int)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(source_subjects) * val_fraction)))
    n_val = min(n_val, len(source_subjects) - 1)
    return sorted(shuffled[:n_val].astype(int).tolist())


def loso_masks(
    subjects: np.ndarray,
    target_subject: int,
    val_fraction: float = 0.15,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_subjects = sorted(np.unique(subjects).astype(int).tolist())
    source_subjects = [s for s in all_subjects if s != target_subject]
    if not source_subjects:
        raise ValueError("LOSO requires at least two subjects")
    val_subjects = source_validation_subjects(source_subjects, val_fraction, seed + target_subject)
    train_subjects = [s for s in source_subjects if s not in val_subjects]
    train_mask = np.isin(subjects, train_subjects)
    val_mask = np.isin(subjects, val_subjects)
    test_mask = subjects == target_subject
    return train_mask, val_mask, test_mask


def within_subject_repetition_masks(
    subjects: np.ndarray,
    repetitions: np.ndarray,
    subject: int,
    test_repetitions: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    subject_mask = subjects == subject
    test_mask = subject_mask & np.isin(repetitions, test_repetitions)
    train_mask = subject_mask & ~np.isin(repetitions, test_repetitions)
    return train_mask, test_mask
