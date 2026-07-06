#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from semg_diff.config import ensure_project_dirs, load_config
from semg_diff.data.db2 import check_db2_integrity, dataset_statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Ninapro DB2 .mat file integrity.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)

    dataset = cfg["dataset"]
    project = cfg["project"]
    out_dir = Path(args.out_dir or project["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    integrity = check_db2_integrity(
        project["raw_db2_dir"],
        subjects=list(map(int, dataset["subjects"])),
        exercises=list(map(int, dataset["exercises"])),
        expected_channels=int(dataset["emg_channels"]),
    )
    stats = dataset_statistics(
        project["raw_db2_dir"],
        subjects=list(map(int, dataset["subjects"])),
        exercises=list(map(int, dataset["exercises"])),
    )

    integrity_path = out_dir / "db2_integrity.csv"
    stats_path = out_dir / "db2_statistics.csv"
    integrity.to_csv(integrity_path, index=False)
    stats.to_csv(stats_path, index=False)

    ok = bool(integrity["ok"].all())
    print(f"Integrity rows: {len(integrity)}")
    print(f"All files OK: {ok}")
    print(f"Wrote: {integrity_path}")
    print(f"Wrote: {stats_path}")
    if not ok:
        print(integrity.loc[~integrity["ok"], ["subject", "exercise", "error"]].to_string(index=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
