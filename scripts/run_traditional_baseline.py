#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from semg_diff.config import ensure_project_dirs, get_seed, load_config
from semg_diff.evaluation.metrics import (
    classification_metrics,
    confusion_matrix_frame,
    per_class_f1,
    subject_stability_metrics,
)
from semg_diff.features.time_domain import extract_time_domain_features
from semg_diff.models.traditional import build_classifier, fit_classifier
from semg_diff.stable_learning.weights import class_environment_balance_weights
from semg_diff.training.splits import loso_masks, loso_subjects, within_subject_repetition_masks
from semg_diff.utils import set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run traditional ML baselines on DB2 window cache.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--windows", required=True, help="Window cache .npz from build_db2_windows.py")
    parser.add_argument("--protocol", choices=["loso", "within_subject"], default=None)
    parser.add_argument("--classifiers", nargs="*", default=None)
    parser.add_argument("--stable-weights", action="store_true")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def load_windows(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files if key != "metadata_csv"}


def fit_predict_fold(
    clf_name: str,
    clf_cfg: dict,
    features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    sample_weights: np.ndarray | None,
) -> np.ndarray:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(features[train_mask])
    x_test = scaler.transform(features[test_mask])
    model = build_classifier(clf_name, clf_cfg)
    fit_classifier(model, x_train, labels[train_mask], sample_weight=sample_weights)
    return model.predict(x_test)


def run_loso(
    cfg: dict,
    cache: dict[str, np.ndarray],
    classifiers: list[str],
    stable_weights: bool,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cfg = cfg["features"]["time_domain"]
    features, feature_names = extract_time_domain_features(
        cache["windows"],
        enabled=list(feature_cfg["enabled"]),
        feature_cfg=feature_cfg,
    )
    pd.Series(feature_names, name="feature").to_csv(out_dir / "time_domain_features.csv", index=False)

    labels = cache["labels"].astype(int)
    subjects = cache["subjects"].astype(int)
    env_ids = cache["env_ids"].astype(int)
    rows: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []
    split_cfg = cfg["splits"]

    for clf_name in classifiers:
        clf_cfg = dict(cfg["models"]["traditional"])
        clf_cfg["random_state"] = get_seed(cfg)
        for target_subject in tqdm(loso_subjects(subjects), desc=f"{clf_name} LOSO"):
            train_mask, val_mask, test_mask = loso_masks(
                subjects,
                target_subject=target_subject,
                val_fraction=float(split_cfg.get("val_subject_fraction", 0.15)),
                seed=get_seed(cfg),
            )
            train_or_val_mask = train_mask | val_mask
            weights = None
            method = "standard"
            if stable_weights:
                weights = class_environment_balance_weights(
                    labels[train_or_val_mask],
                    env_ids[train_or_val_mask],
                    max_weight=float(cfg["stable_learning"]["sample_reweighting"].get("max_weight", 10.0)),
                )
                method = "class_environment_balance"

            y_pred = fit_predict_fold(
                clf_name,
                clf_cfg,
                features,
                labels,
                train_or_val_mask,
                test_mask,
                weights,
            )
            y_true = labels[test_mask]
            metrics = classification_metrics(y_true, y_pred)
            rows.append(
                {
                    "protocol": "loso",
                    "model": clf_name,
                    "stable_method": method,
                    "target_subject": target_subject,
                    "n_train": int(train_or_val_mask.sum()),
                    "n_test": int(test_mask.sum()),
                    **metrics,
                }
            )
            all_predictions.append(
                pd.DataFrame(
                    {
                        "model": clf_name,
                        "stable_method": method,
                        "target_subject": target_subject,
                        "y_true": y_true,
                        "y_pred": y_pred,
                    }
                )
            )

    results = pd.DataFrame(rows)
    predictions = pd.concat(all_predictions, ignore_index=True)
    return results, predictions


def run_within_subject(
    cfg: dict,
    cache: dict[str, np.ndarray],
    classifiers: list[str],
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cfg = cfg["features"]["time_domain"]
    features, feature_names = extract_time_domain_features(
        cache["windows"],
        enabled=list(feature_cfg["enabled"]),
        feature_cfg=feature_cfg,
    )
    pd.Series(feature_names, name="feature").to_csv(out_dir / "time_domain_features.csv", index=False)

    labels = cache["labels"].astype(int)
    subjects = cache["subjects"].astype(int)
    repetitions = cache["repetitions"].astype(int)
    rows: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []
    test_reps = list(map(int, cfg["splits"].get("within_subject_test_repetitions", [2, 5])))

    for clf_name in classifiers:
        clf_cfg = dict(cfg["models"]["traditional"])
        clf_cfg["random_state"] = get_seed(cfg)
        for subject in tqdm(loso_subjects(subjects), desc=f"{clf_name} within"):
            train_mask, test_mask = within_subject_repetition_masks(
                subjects,
                repetitions,
                subject=subject,
                test_repetitions=test_reps,
            )
            if not train_mask.any() or not test_mask.any():
                continue
            y_pred = fit_predict_fold(
                clf_name,
                clf_cfg,
                features,
                labels,
                train_mask,
                test_mask,
                sample_weights=None,
            )
            y_true = labels[test_mask]
            rows.append(
                {
                    "protocol": "within_subject",
                    "model": clf_name,
                    "stable_method": "standard",
                    "target_subject": subject,
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                    **classification_metrics(y_true, y_pred),
                }
            )
            all_predictions.append(
                pd.DataFrame(
                    {
                        "model": clf_name,
                        "stable_method": "standard",
                        "target_subject": subject,
                        "y_true": y_true,
                        "y_pred": y_pred,
                    }
                )
            )

    results = pd.DataFrame(rows)
    predictions = pd.concat(all_predictions, ignore_index=True)
    return results, predictions


def write_reports(results: pd.DataFrame, predictions: pd.DataFrame, out_dir: Path) -> None:
    results.to_csv(out_dir / "fold_results.csv", index=False)
    summary_rows = []
    for (model, method), group in results.groupby(["model", "stable_method"]):
        summary_rows.append({"model": model, "stable_method": method, **subject_stability_metrics(group)})
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)

    predictions.to_csv(out_dir / "predictions.csv", index=False)
    for (model, method), group in predictions.groupby(["model", "stable_method"]):
        tag = f"{model}_{method}".replace("/", "_")
        per_class_f1(group["y_true"].to_numpy(), group["y_pred"].to_numpy()).to_csv(
            out_dir / f"{tag}_per_class_f1.csv",
            index=False,
        )
        confusion_matrix_frame(group["y_true"].to_numpy(), group["y_pred"].to_numpy()).to_csv(
            out_dir / f"{tag}_confusion_matrix.csv"
        )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)
    set_random_seed(get_seed(cfg))

    cache = load_windows(args.windows)
    protocol = args.protocol or cfg["splits"].get("protocol", "loso")
    classifiers = args.classifiers or list(cfg["models"]["traditional"]["classifiers"])
    stable_suffix = "_stable_weights" if args.stable_weights else ""
    out_dir = Path(args.out_dir or Path(cfg["project"]["results_dir"]) / f"{protocol}_traditional{stable_suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if protocol == "loso":
        results, predictions = run_loso(cfg, cache, classifiers, args.stable_weights, out_dir)
    else:
        results, predictions = run_within_subject(cfg, cache, classifiers, out_dir)

    write_reports(results, predictions, out_dir)
    print(results.to_string(index=False))
    print(f"Wrote reports to: {out_dir}")


if __name__ == "__main__":
    main()
