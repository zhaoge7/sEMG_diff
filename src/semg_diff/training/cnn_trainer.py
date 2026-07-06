from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 6
    stable_lambda: float = 0.0
    device: str = "auto"


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def stable_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    env_ids: torch.Tensor,
    stable_lambda: float,
) -> torch.Tensor:
    per_sample = nn.functional.cross_entropy(logits, labels, reduction="none")
    if stable_lambda <= 0:
        return per_sample.mean()

    env_losses = []
    for env in torch.unique(env_ids):
        mask = env_ids == env
        if mask.any():
            env_losses.append(per_sample[mask].mean())
    if len(env_losses) <= 1:
        return per_sample.mean()
    stacked = torch.stack(env_losses)
    return stacked.mean() + stable_lambda * torch.var(stacked, unbiased=False)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    stable_lambda: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []

    for x, y, env in loader:
        x = x.to(device)
        y = y.to(device)
        env = env.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = stable_cross_entropy_loss(logits, y, env, stable_lambda if training else 0.0)
            if training:
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        y_true.append(y.detach().cpu().numpy())
        y_pred.append(torch.argmax(logits, dim=1).detach().cpu().numpy())

    truth = np.concatenate(y_true) if y_true else np.array([], dtype=int)
    pred = np.concatenate(y_pred) if y_pred else np.array([], dtype=int)
    return {
        "loss": float(np.mean(losses)) if losses else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)) if truth.size else np.nan,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    checkpoint_path: str | Path,
) -> pd.DataFrame:
    device = resolve_device(cfg.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float | int]] = []
    best_score = -np.inf
    bad_epochs = 0
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, cfg.stable_lambda)
        val_metrics = run_epoch(model, val_loader, None, device, 0.0)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "val_loss": val_metrics["loss"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        }
        history.append(row)

        score = float(val_metrics["balanced_accuracy"])
        if score > best_score:
            best_score = score
            bad_epochs = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "score": score}, checkpoint_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return pd.DataFrame(history)


def predict_model(model: nn.Module, loader: DataLoader, device: str = "auto") -> tuple[np.ndarray, np.ndarray]:
    torch_device = resolve_device(device)
    model.to(torch_device)
    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    with torch.no_grad():
        for x, y, _env in loader:
            logits = model(x.to(torch_device))
            y_true.append(y.numpy())
            y_pred.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)
