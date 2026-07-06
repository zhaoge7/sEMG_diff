#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from semg_diff.config import ensure_project_dirs, get_seed, load_config
from semg_diff.evaluation.metrics import (
    classification_metrics,
    confusion_matrix_frame,
    per_class_f1,
    subject_stability_metrics,
)
from semg_diff.models.cnn import SimpleEMGCNN
from semg_diff.training.cnn_trainer import TrainConfig, predict_model, train_model
from semg_diff.training.splits import loso_masks, loso_subjects
from semg_diff.training.torch_data import EMGWindowDataset
from semg_diff.utils import set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 1D-CNN or Stable-CNN LOSO baseline.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--windows", required=True)
    parser.add_argument("--stable-risk", action="store_true")
    parser.add_argument("--target-subjects", nargs="*", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def load_windows(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files if key != "metadata_csv"}


def transform_by_train_scaler(
    windows: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train = windows[train_mask]
    val = windows[val_mask]
    test = windows[test_mask]
    n_train, t, c = train.shape
    scaler.fit(train.reshape(-1, c))
    return (
        scaler.transform(train.reshape(-1, c)).reshape(n_train, t, c).astype(np.float32),
        scaler.transform(val.reshape(-1, c)).reshape(val.shape[0], t, c).astype(np.float32),
        scaler.transform(test.reshape(-1, c)).reshape(test.shape[0], t, c).astype(np.float32),
    )


def make_loader(windows: np.ndarray, labels: np.ndarray, env_ids: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        EMGWindowDataset(windows, labels, env_ids),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)
    set_random_seed(get_seed(cfg))

    cache = load_windows(args.windows)
    windows = cache["windows"].astype(np.float32)
    raw_labels = cache["labels"].astype(int)
    subjects = cache["subjects"].astype(int)
    env_ids = cache["env_ids"].astype(int)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(raw_labels).astype(np.int64)

    cnn_cfg = cfg["models"]["cnn"]
    stable_lambda = float(cfg["stable_learning"]["stable_risk"].get("lambda", 0.5)) if args.stable_risk else 0.0
    method = "stable_risk" if args.stable_risk else "standard"
    out_dir = Path(args.out_dir or Path(cfg["project"]["results_dir"]) / f"loso_cnn_{method}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(label_encoder.classes_, name="original_label").to_csv(out_dir / "label_mapping.csv", index_label="class_id")

    target_subjects = args.target_subjects or loso_subjects(subjects)
    rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    split_cfg = cfg["splits"]

    for target_subject in tqdm(target_subjects, desc=f"CNN {method} LOSO"):
        train_mask, val_mask, test_mask = loso_masks(
            subjects,
            target_subject=target_subject,
            val_fraction=float(split_cfg.get("val_subject_fraction", 0.15)),
            seed=get_seed(cfg),
        )
        train_x, val_x, test_x = transform_by_train_scaler(windows, train_mask, val_mask, test_mask)
        train_y, val_y, test_y = labels[train_mask], labels[val_mask], labels[test_mask]
        train_env, val_env, test_env = env_ids[train_mask], env_ids[val_mask], env_ids[test_mask]

        model = SimpleEMGCNN(
            in_channels=train_x.shape[-1],
            num_classes=len(label_encoder.classes_),
            hidden_channels=list(cnn_cfg.get("hidden_channels", [64, 128, 128])),
            dropout=float(cnn_cfg.get("dropout", 0.30)),
        )
        train_loader = make_loader(train_x, train_y, train_env, int(cnn_cfg["batch_size"]), shuffle=True)
        val_loader = make_loader(val_x, val_y, val_env, int(cnn_cfg["batch_size"]), shuffle=False)
        test_loader = make_loader(test_x, test_y, test_env, int(cnn_cfg["batch_size"]), shuffle=False)
        checkpoint_path = out_dir / f"subject_{target_subject:02d}.pt"
        history = train_model(
            model,
            train_loader,
            val_loader,
            TrainConfig(
                epochs=int(args.epochs or cnn_cfg["epochs"]),
                learning_rate=float(cnn_cfg["learning_rate"]),
                weight_decay=float(cnn_cfg["weight_decay"]),
                patience=int(cnn_cfg["patience"]),
                stable_lambda=stable_lambda,
                device=args.device,
            ),
            checkpoint_path=checkpoint_path,
        )
        history.to_csv(out_dir / f"subject_{target_subject:02d}_history.csv", index=False)

        encoded_true, encoded_pred = predict_model(model, test_loader, device=args.device)
        y_true = label_encoder.inverse_transform(encoded_true)
        y_pred = label_encoder.inverse_transform(encoded_pred)
        rows.append(
            {
                "protocol": "loso",
                "model": "simple_1d_cnn",
                "stable_method": method,
                "target_subject": target_subject,
                "n_train": int(train_mask.sum()),
                "n_val": int(val_mask.sum()),
                "n_test": int(test_mask.sum()),
                **classification_metrics(y_true, y_pred),
            }
        )
        predictions.append(
            pd.DataFrame(
                {
                    "model": "simple_1d_cnn",
                    "stable_method": method,
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
    print(f"Wrote reports to: {out_dir}")


if __name__ == "__main__":
    main()
