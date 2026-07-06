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
from semg_diff.stable_learning.coral import coral_align_source_to_target
from semg_diff.training.splits import loso_masks, loso_subjects
from semg_diff.utils import set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unsupervised CORAL + traditional classifier baseline.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--windows", required=True)
    parser.add_argument("--classifier", default="linear_svm")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)
    set_random_seed(get_seed(cfg))
    out_dir = Path(args.out_dir or Path(cfg["project"]["results_dir"]) / f"loso_coral_{args.classifier}")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.windows, allow_pickle=True)
    windows = data["windows"].astype(np.float32)
    labels = data["labels"].astype(int)
    subjects = data["subjects"].astype(int)
    feature_cfg = cfg["features"]["time_domain"]
    features, _feature_names = extract_time_domain_features(
        windows,
        enabled=list(feature_cfg["enabled"]),
        feature_cfg=feature_cfg,
    )

    rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for target_subject in tqdm(loso_subjects(subjects), desc=f"CORAL {args.classifier} LOSO"):
        train_mask, val_mask, test_mask = loso_masks(
            subjects,
            target_subject,
            val_fraction=float(cfg["splits"].get("val_subject_fraction", 0.15)),
            seed=get_seed(cfg),
        )
        source_mask = train_mask | val_mask
        scaler = StandardScaler()
        source_x = scaler.fit_transform(features[source_mask])
        target_x = scaler.transform(features[test_mask])
        aligned_source_x = coral_align_source_to_target(source_x, target_x)
        model = build_classifier(args.classifier, cfg["models"]["traditional"])
        fit_classifier(model, aligned_source_x, labels[source_mask])
        y_pred = model.predict(target_x)
        y_true = labels[test_mask]
        rows.append(
            {
                "protocol": "loso_unsupervised_target_adaptation",
                "model": args.classifier,
                "stable_method": "coral_unlabeled_target",
                "target_subject": target_subject,
                "n_train": int(source_mask.sum()),
                "n_unlabeled_target": int(test_mask.sum()),
                "n_test": int(test_mask.sum()),
                **classification_metrics(y_true, y_pred),
            }
        )
        predictions.append(
            pd.DataFrame(
                {
                    "model": args.classifier,
                    "stable_method": "coral_unlabeled_target",
                    "target_subject": target_subject,
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            )
        )

    results = pd.DataFrame(rows)
    pred_df = pd.concat(predictions, ignore_index=True)
    results.to_csv(out_dir / "fold_results.csv", index=False)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame([subject_stability_metrics(results)]).to_csv(out_dir / "summary.csv", index=False)
    per_class_f1(pred_df["y_true"].to_numpy(), pred_df["y_pred"].to_numpy()).to_csv(
        out_dir / "per_class_f1.csv",
        index=False,
    )
    confusion_matrix_frame(pred_df["y_true"].to_numpy(), pred_df["y_pred"].to_numpy()).to_csv(
        out_dir / "confusion_matrix.csv"
    )
    print(results.to_string(index=False))
    print("Note: CORAL uses unlabeled target-subject features, so this is unsupervised domain adaptation.")
    print(f"Wrote reports to: {out_dir}")


if __name__ == "__main__":
    main()
