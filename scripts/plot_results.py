#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot subject metrics and confusion matrices.")
    parser.add_argument("--fold-results", nargs="+", required=True)
    parser.add_argument("--prediction-files", nargs="*", default=None)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def method_name(frame: pd.DataFrame) -> pd.Series:
    return frame["model"].astype(str) + "/" + frame["stable_method"].astype(str)


def safe_name(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("\\", "_")
    )

def plot_fold_metrics(data: pd.DataFrame, out_dir: Path) -> None:
    data = data.copy()
    data["method"] = method_name(data)
    sns.set_theme(style="whitegrid")
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        if metric not in data:
            continue
        plt.figure(figsize=(12, 5))
        sns.barplot(data=data, x="target_subject", y=metric, hue="method")
        plt.xlabel("Target subject")
        plt.ylabel(metric.replace("_", " ").title())
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_by_subject.png", dpi=180)
        plt.close()

        plt.figure(figsize=(7, 5))
        sns.boxplot(data=data, x="method", y=metric)
        plt.xlabel("")
        plt.ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_boxplot.png", dpi=180)
        plt.close()


def discover_prediction_files(fold_result_paths: list[Path], explicit: list[str] | None) -> list[Path]:
    if explicit:
        return [Path(p) for p in explicit]
    paths: list[Path] = []
    for fold_path in fold_result_paths:
        candidate = fold_path.parent / "predictions.csv"
        if candidate.exists():
            paths.append(candidate)
    return paths


def plot_confusion_matrices(prediction_paths: list[Path], out_dir: Path) -> None:
    for path in prediction_paths:
        pred = pd.read_csv(path)
        if not {"model", "stable_method", "y_true", "y_pred"}.issubset(pred.columns):
            continue
        pred["method"] = method_name(pred)
        for method, group in pred.groupby("method"):
            labels = sorted(set(group["y_true"].tolist()) | set(group["y_pred"].tolist()))
            cm = confusion_matrix(group["y_true"], group["y_pred"], labels=labels)
            cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
            size = max(6, min(16, len(labels) * 0.45))
            plt.figure(figsize=(size, size))
            sns.heatmap(
                cm_norm,
                cmap="viridis",
                vmin=0,
                vmax=1,
                xticklabels=labels,
                yticklabels=labels,
                square=True,
                cbar_kws={"label": "Row-normalized count"},
            )
            plt.xlabel("Predicted label")
            plt.ylabel("True label")
            plt.title(method)
            plt.tight_layout()
            stem = safe_name(f"{path.parent.name}_{method}_confusion_matrix")
            plt.savefig(out_dir / f"{stem}.png", dpi=180)
            plt.close()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_paths = [Path(path) for path in args.fold_results]
    frames = []
    for path in fold_paths:
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    plot_fold_metrics(data, out_dir)

    prediction_paths = discover_prediction_files(fold_paths, args.prediction_files)
    plot_confusion_matrices(prediction_paths, out_dir)
    print(f"Wrote plots to: {out_dir}")


if __name__ == "__main__":
    main()
