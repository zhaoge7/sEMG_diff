from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

REQUIRED_FIELDS = ("emg", "stimulus", "restimulus", "repetition", "rerepetition")
FILE_RE = re.compile(r"S(?P<subject>\d+)_E(?P<exercise>\d+)_A(?P<acquisition>\d+)\.mat$")


@dataclass(frozen=True)
class DB2File:
    path: Path
    subject: int
    exercise: int
    acquisition: int


@dataclass
class DB2Recording:
    emg: np.ndarray
    stimulus: np.ndarray
    restimulus: np.ndarray
    repetition: np.ndarray
    rerepetition: np.ndarray
    subject: int
    exercise: int
    path: Path

    @property
    def labels(self) -> np.ndarray:
        return self.restimulus

    @property
    def repetitions(self) -> np.ndarray:
        return self.rerepetition


def discover_db2_files(
    raw_dir: str | Path,
    subjects: list[int] | None = None,
    exercises: list[int] | None = None,
) -> list[DB2File]:
    raw_path = Path(raw_dir)
    subject_set = set(subjects) if subjects else None
    exercise_set = set(exercises) if exercises else None
    files: list[DB2File] = []

    for path in raw_path.glob("DB2_s*/S*_E*_A*.mat"):
        match = FILE_RE.match(path.name)
        if not match:
            continue
        subject = int(match.group("subject"))
        exercise = int(match.group("exercise"))
        acquisition = int(match.group("acquisition"))
        if subject_set is not None and subject not in subject_set:
            continue
        if exercise_set is not None and exercise not in exercise_set:
            continue
        files.append(DB2File(path=path, subject=subject, exercise=exercise, acquisition=acquisition))

    return sorted(files, key=lambda f: (f.subject, f.exercise, f.acquisition))


def expected_file(subject: int, exercise: int, raw_dir: str | Path) -> Path:
    return Path(raw_dir) / f"DB2_s{subject}" / f"S{subject}_E{exercise}_A1.mat"


def load_db2_recording(db2_file: DB2File | str | Path) -> DB2Recording:
    if isinstance(db2_file, DB2File):
        file_info = db2_file
    else:
        path = Path(db2_file)
        match = FILE_RE.match(path.name)
        if not match:
            raise ValueError(f"Cannot infer DB2 subject/exercise from filename: {path}")
        file_info = DB2File(
            path=path,
            subject=int(match.group("subject")),
            exercise=int(match.group("exercise")),
            acquisition=int(match.group("acquisition")),
        )

    mat = loadmat(file_info.path)
    missing = [field for field in REQUIRED_FIELDS if field not in mat]
    if missing:
        raise KeyError(f"{file_info.path} is missing required fields: {missing}")

    emg = np.asarray(mat["emg"], dtype=np.float32)
    if emg.ndim != 2:
        raise ValueError(f"Expected emg to be 2D, got shape {emg.shape} in {file_info.path}")

    def flat_int(name: str) -> np.ndarray:
        values = np.asarray(mat[name]).reshape(-1)
        if values.shape[0] != emg.shape[0]:
            raise ValueError(
                f"Field {name} length {values.shape[0]} does not match emg length "
                f"{emg.shape[0]} in {file_info.path}"
            )
        return values.astype(np.int16, copy=False)

    return DB2Recording(
        emg=emg,
        stimulus=flat_int("stimulus"),
        restimulus=flat_int("restimulus"),
        repetition=flat_int("repetition"),
        rerepetition=flat_int("rerepetition"),
        subject=file_info.subject,
        exercise=file_info.exercise,
        path=file_info.path,
    )


def check_db2_integrity(
    raw_dir: str | Path,
    subjects: list[int],
    exercises: list[int],
    expected_channels: int = 12,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject in subjects:
        for exercise in exercises:
            path = expected_file(subject, exercise, raw_dir)
            row: dict[str, object] = {
                "subject": subject,
                "exercise": exercise,
                "path": str(path),
                "exists": path.exists(),
                "ok": False,
                "samples": 0,
                "channels": 0,
                "gestures": "",
                "repetitions": "",
                "error": "",
            }
            if not path.exists():
                row["error"] = "missing file"
                rows.append(row)
                continue
            try:
                rec = load_db2_recording(path)
                row["samples"] = int(rec.emg.shape[0])
                row["channels"] = int(rec.emg.shape[1])
                row["gestures"] = ",".join(map(str, sorted(set(rec.restimulus.tolist()))))
                row["repetitions"] = ",".join(map(str, sorted(set(rec.rerepetition.tolist()))))
                row["ok"] = rec.emg.shape[1] == expected_channels
                if not row["ok"]:
                    row["error"] = f"expected {expected_channels} channels"
            except Exception as exc:
                row["error"] = str(exc)
            rows.append(row)
    return pd.DataFrame(rows)


def dataset_statistics(
    raw_dir: str | Path,
    subjects: list[int],
    exercises: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for file_info in discover_db2_files(raw_dir, subjects=subjects, exercises=exercises):
        rec = load_db2_recording(file_info)
        labels, label_counts = np.unique(rec.restimulus, return_counts=True)
        active = rec.restimulus > 0
        active_labels = sorted(np.unique(rec.restimulus[active]).astype(int).tolist())
        active_repetitions = sorted(np.unique(rec.rerepetition[active]).astype(int).tolist())
        rows.append(
            {
                "subject": rec.subject,
                "exercise": rec.exercise,
                "samples": int(rec.emg.shape[0]),
                "seconds": float(rec.emg.shape[0] / 2000.0),
                "channels": int(rec.emg.shape[1]),
                "min_label": int(labels.min()),
                "max_label": int(labels.max()),
                "num_labels_including_rest": int(labels.size),
                "active_gestures": ",".join(map(str, active_labels)),
                "active_repetitions": ",".join(map(str, active_repetitions)),
                "rest_fraction": float(label_counts[labels == 0][0] / rec.emg.shape[0])
                if np.any(labels == 0)
                else 0.0,
            }
        )
    return pd.DataFrame(rows)
