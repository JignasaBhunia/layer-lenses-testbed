"""Training utilities for standard ReLU MLP models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from first_experiment.relu_mlp import ReLUMLP
from first_experiment.training import set_seed


@dataclass(frozen=True)
class ReLUTrainConfig:
    """Training config for single-phase ReLU MLP experiments."""

    epochs: int = 200
    lr: float = 1e-3
    batch_size: int = 1024
    seed: int = 365
    device: str = "cpu"
    snapshot_epochs: tuple[int, ...] = (0,)
    weight_decay: float = 0.0
    lr_scheduler_eta_min_ratio: float = 0.2
    show_progress: bool = False


def _labels_pm1_to_binary(y: torch.Tensor) -> torch.Tensor:
    if not torch.all((y == -1) | (y == 1)):
        raise ValueError("Expected labels in {-1, +1}.")
    return (y + 1.0) / 2.0


def evaluate_relu_mlp(
    *,
    model: nn.Module,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    device: str = "cpu",
) -> dict[str, int | float]:
    """Evaluate log loss and zero-one loss for a ReLU MLP classifier."""
    model = model.to(torch.device(device))
    model.eval()
    x = torch.from_numpy(x_eval).float().to(device)
    y_pm1 = torch.from_numpy(y_eval).float().to(device)
    y_bin = _labels_pm1_to_binary(y_pm1)

    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        logits = model(x)
        log_loss = float(criterion(logits, y_bin).detach().cpu())
        preds_pm1 = torch.where(logits >= 0, 1.0, -1.0)
        zero_one_loss = float((preds_pm1 != y_pm1).float().mean().detach().cpu())

    return {
        "log_loss": log_loss,
        "zero_one_loss": zero_one_loss,
        "accuracy": 1.0 - zero_one_loss,
        "num_samples": int(x_eval.shape[0]),
    }


def train_relu_mlp(
    *,
    model: ReLUMLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: ReLUTrainConfig,
) -> dict[str, object]:
    """Train ReLU MLP with Adam + BCE-with-logits and cosine annealing LR."""
    set_seed(config.seed)
    device = torch.device(config.device)
    model = model.to(device)

    x = torch.from_numpy(x_train).float().to(device)
    y_pm1 = torch.from_numpy(y_train).float().to(device)
    y_bin = _labels_pm1_to_binary(y_pm1)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=max(1, config.epochs),
        eta_min=config.lr * config.lr_scheduler_eta_min_ratio,
    )

    num_samples = x.shape[0]
    losses: list[float] = []
    lr_history: list[float] = []
    checkpoint_snapshots: dict[int, dict[str, torch.Tensor]] = {}

    epoch_iter = range(config.epochs + 1)
    if config.show_progress:
        epoch_iter = tqdm(epoch_iter, desc="Training ReLU MLP", leave=False)

    for epoch in epoch_iter:
        if epoch in config.snapshot_epochs:
            checkpoint_snapshots[epoch] = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        permutation = torch.randperm(num_samples, device=device)
        if epoch == config.epochs:
            break

        running_loss = 0.0
        seen = 0
        for start in range(0, num_samples, config.batch_size):
            end = min(start + config.batch_size, num_samples)
            idx = permutation[start:end]

            optimizer.zero_grad()
            logits = model(x[idx])
            loss = criterion(logits, y_bin[idx])
            loss.backward()
            optimizer.step()

            batch_n = end - start
            running_loss += float(loss.detach().cpu()) * batch_n
            seen += batch_n

        losses.append(running_loss / max(1, seen))
        lr_history.append(float(optimizer.param_groups[0]["lr"]))
        scheduler.step()
        if config.show_progress:
            epoch_iter.set_postfix(loss=f"{losses[-1]:.4f}")

    return {
        "model": model,
        "epoch_losses": losses,
        "lr_history": lr_history,
        "checkpoint_snapshots": checkpoint_snapshots,
    }
