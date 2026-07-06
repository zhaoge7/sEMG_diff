#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from semg_diff.config import ensure_project_dirs, load_config
from semg_diff.data.db2 import discover_db2_files, load_db2_recording
from semg_diff.data.preprocessing import FilterConfig, preprocess_emg
from semg_diff.data.windowing import WindowConfig, concatenate_window_sets, slice_recording_windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Ninapro DB2 window cache from .mat files.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--subjects", nargs="*", type=int, default=None)
    parser.add_argument("--exercises", nargs="*", type=int, default=None)
    parser.add_argument("--out", default=None, help="Output .npz path.")
    parser.add_argument("--no-filter", action="store_true", help="Skip bandpass/notch filtering.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)

    dataset = cfg["dataset"]
    project = cfg["project"]
    preprocessing = cfg["preprocessing"]
    windowing = cfg["windowing"]

    subjects = args.subjects or list(map(int, dataset["subjects"]))
    exercises = args.exercises or list(map(int, dataset["exercises"]))
    files = discover_db2_files(project["raw_db2_dir"], subjects=subjects, exercises=exercises)
    if not files:
        raise SystemExit("No DB2 files found for the requested subjects/exercises")

    filter_cfg = FilterConfig(
        sampling_rate_hz=int(dataset["sampling_rate_hz"]),
        bandpass_hz=tuple(map(float, preprocessing["bandpass_hz"])),
        notch_hz=float(preprocessing["notch_hz"]) if preprocessing.get("notch_hz") else None,
        notch_quality=float(preprocessing.get("notch_quality", 30)),
        enabled=bool(preprocessing.get("filter", True)) and not args.no_filter,
        rectify=bool(preprocessing.get("rectify", False)),
    )
    window_cfg = WindowConfig(
        sampling_rate_hz=int(dataset["sampling_rate_hz"]),
        window_ms=int(windowing["window_ms"]),
        step_ms=int(windowing["step_ms"]),
        label_majority_threshold=float(windowing["label_majority_threshold"]),
        include_rest=bool(dataset.get("include_rest", False)),
    )

    parts = []
    for file_info in tqdm(files, desc="windowing DB2"):
        recording = load_db2_recording(file_info)
        emg = preprocess_emg(recording.emg, filter_cfg)
        parts.append(slice_recording_windows(recording, emg, window_cfg))

    windows, metadata = concatenate_window_sets(parts)
    dtype = np.dtype(windowing.get("save_dtype", "float32"))
    windows = windows.astype(dtype, copy=False)

    out_path = Path(
        args.out
        or Path(project["processed_dir"])
        / f"db2_E{'-'.join(map(str, exercises))}_S{'-'.join(map(str, subjects[:3]))}"
        f"{'_all' if len(subjects) > 3 else ''}_{window_cfg.window_ms}ms.npz"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        windows=windows,
        labels=metadata["label"].to_numpy(np.int16),
        subjects=metadata["subject"].to_numpy(np.int16),
        exercises=metadata["exercise"].to_numpy(np.int16),
        repetitions=metadata["repetition"].to_numpy(np.int16),
        env_ids=metadata["env_id"].to_numpy(np.int16),
        starts=metadata["start"].to_numpy(np.int64),
        ends=metadata["end"].to_numpy(np.int64),
        sample_weights=metadata["sample_weight"].to_numpy(np.float32),
        metadata_csv=metadata.to_csv(index=False),
    )
    metadata_path = out_path.with_suffix(".metadata.csv")
    metadata.to_csv(metadata_path, index=False)

    print(f"Windows: {windows.shape}")
    print(f"Labels: {metadata['label'].nunique()} classes")
    print(f"Subjects: {metadata['subject'].nunique()}")
    print(f"Wrote: {out_path}")
    print(f"Wrote: {metadata_path}")


if __name__ == "__main__":
    main()
