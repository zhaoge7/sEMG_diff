from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC


class WeightedLDAClassifier:
    """Linear discriminant analysis with optional per-sample weights."""

    def __init__(self, reg: float = 1e-6) -> None:
        self.reg = float(reg)

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        if sample_weight is None:
            sample_weight = np.ones(y.shape[0], dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if x.shape[0] != y.shape[0] or y.shape[0] != sample_weight.shape[0]:
            raise ValueError("x, y and sample_weight must have matching first dimensions")

        classes = np.unique(y)
        means = []
        priors = []
        covariance = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
        total_weight = float(sample_weight.sum())
        if total_weight <= 0:
            raise ValueError("sample_weight sum must be positive")

        for cls in classes:
            mask = y == cls
            class_x = x[mask]
            class_w = sample_weight[mask]
            class_weight = float(class_w.sum())
            if class_weight <= 0:
                continue
            mean = np.average(class_x, axis=0, weights=class_w)
            centered = class_x - mean
            covariance += (centered * class_w[:, None]).T @ centered
            means.append(mean)
            priors.append(class_weight / total_weight)

        self.classes_ = classes
        self.means_ = np.vstack(means)
        self.priors_ = np.asarray(priors, dtype=np.float64)
        denom = max(total_weight - len(classes), 1.0)
        covariance = covariance / denom
        trace_scale = float(np.trace(covariance) / max(covariance.shape[0], 1))
        covariance += np.eye(covariance.shape[0]) * self.reg * max(trace_scale, 1.0)
        self.precision_ = np.linalg.pinv(covariance)
        self.coef_ = self.means_ @ self.precision_
        self.intercept_ = -0.5 * np.sum(self.coef_ * self.means_, axis=1) + np.log(self.priors_)
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) @ self.coef_.T + self.intercept_

    def predict(self, x: np.ndarray) -> np.ndarray:
        scores = self.decision_function(x)
        return self.classes_[np.argmax(scores, axis=1)]


def build_classifier(name: str, cfg: dict[str, Any] | None = None):
    cfg = cfg or {}
    if name == "lda":
        return WeightedLDAClassifier(reg=float(cfg.get("lda_reg", 1e-6)))
    if name == "linear_svm":
        return LinearSVC(
            C=float(cfg.get("C", 1.0)),
            class_weight=cfg.get("class_weight"),
            dual="auto",
            max_iter=int(cfg.get("max_iter", 5000)),
        )
    if name == "rbf_svm":
        return SVC(
            C=float(cfg.get("C", 10.0)),
            gamma=cfg.get("gamma", "scale"),
            kernel="rbf",
            class_weight=cfg.get("class_weight"),
        )
    if name == "random_forest":
        rf_cfg = cfg.get("random_forest", cfg)
        return RandomForestClassifier(
            n_estimators=int(rf_cfg.get("n_estimators", 300)),
            max_depth=rf_cfg.get("max_depth"),
            n_jobs=int(rf_cfg.get("n_jobs", -1)),
            random_state=int(rf_cfg.get("random_state", 2026)),
            class_weight=rf_cfg.get("class_weight"),
        )
    raise KeyError(f"Unknown classifier: {name}")


def fit_classifier(model, x, y, sample_weight=None):
    if sample_weight is not None:
        try:
            return model.fit(x, y, sample_weight=sample_weight)
        except TypeError as exc:
            raise TypeError(f"{type(model).__name__} does not support sample_weight") from exc
    return model.fit(x, y)
