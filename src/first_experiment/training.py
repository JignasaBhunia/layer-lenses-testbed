"""Training utilities for DLGN models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from first_experiment.dlgn import DLGNSF


@dataclass(frozen=True)
class TrainConfig:
    """Minimal training config for milestone 1."""

    epochs: int = 200
    lr: float = 2e-3
    batch_size: int = 1024
    seed: int = 365
    device: str = "cpu"
    snapshot_epochs: tuple[int, ...] = (0,)
    show_progress: bool = True
    weight_decay_gating: float = 0.0
    weight_decay_value: float = 0.0


def set_seed(seed: int) -> None:
    """Set deterministic seeds for NumPy and PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _labels_pm1_to_binary(y: torch.Tensor) -> torch.Tensor:
    if not torch.all((y == -1) | (y == 1)):
        raise ValueError("Expected labels in {-1, +1}.")
    return (y + 1.0) / 2.0


def evaluate_dlgn_sf(
    *,
    model: DLGNSF,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    device: str = "cpu",
) -> dict[str, int | float]:
    """Evaluate log loss and zero-one loss for DLGN-SF on a dataset."""
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
        accuracy = 1.0 - zero_one_loss

    return {
        "log_loss": log_loss,
        "zero_one_loss": zero_one_loss,
        "accuracy": accuracy,
        "num_samples": int(x_eval.shape[0]),
    }


def train_dlgn_sf(
    *,
    model: DLGNSF,
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: TrainConfig,
) -> dict[str, object]:
    """Train DLGN-SF with Adam and BCE-with-logits loss.

    Returns a dictionary with the trained model, epoch losses, and snapshots.
    Snapshots store effective gating weights at selected epochs.
    """
    set_seed(config.seed)
    device = torch.device(config.device)
    model = model.to(device)

    x = torch.from_numpy(x_train).float().to(device)
    y_pm1 = torch.from_numpy(y_train).float().to(device)
    y_bin = _labels_pm1_to_binary(y_pm1)

    criterion = nn.BCEWithLogitsLoss()

    gating_params = list(model.gating_layers.parameters())
    value_params = list(model.value_layers.parameters())
    if getattr(model, "first_value_vector", None) is not None:
        value_params.append(model.first_value_vector)

    optimizer = torch.optim.Adam(
        [
            {"params": gating_params, "weight_decay": config.weight_decay_gating},
            {"params": value_params, "weight_decay": config.weight_decay_value},
        ],
        lr=config.lr,
    )

    num_samples = x.shape[0]
    losses: list[float] = []
    snapshots: dict[int, list[torch.Tensor]] = {}

    epoch_iter = range(config.epochs + 1)
    if config.show_progress:
        epoch_iter = tqdm(epoch_iter, desc="Training DLGN-SF", leave=False)

    for epoch in epoch_iter:
        if epoch in config.snapshot_epochs:
            snapshots[epoch] = [
                w.cpu().clone() for w in model.effective_gating_weights()
            ]
        if epoch == config.epochs:
            break

        permutation = torch.randperm(num_samples, device=device)
        running_loss = 0.0
        seen = 0

        for start in range(0, num_samples, config.batch_size):
            end = min(start + config.batch_size, num_samples)
            idx = permutation[start:end]
            xb = x[idx]
            yb = y_bin[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            batch_n = end - start
            running_loss += float(loss.detach().cpu()) * batch_n
            seen += batch_n

        epoch_loss = running_loss / max(1, seen)
        losses.append(epoch_loss)
        if config.show_progress:
            epoch_iter.set_postfix(loss=f"{epoch_loss:.4f}")

    return {
        "model": model,
        "epoch_losses": losses,
        "snapshots": snapshots,
    }
