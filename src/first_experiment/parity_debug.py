"""Helpers to diagnose numerical parity between two model implementations.

These utilities are intended for notebook-driven debugging (legacy notebook
model vs. clean package model). They check:
1) forward-output parity on a fixed batch,
2) one optimizer-step parity with a supplied loss function.
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import torch
from torch import nn


def _clone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return a detached CPU clone of a state dict."""
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def forward_parity_stats(
    *,
    model_a: nn.Module,
    model_b: nn.Module,
    x: torch.Tensor,
    logits_a_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    logits_b_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
) -> dict[str, float]:
    """Compute forward-output mismatch statistics between two models.

    Args:
        model_a: First model under comparison.
        model_b: Second model under comparison.
        x: Input batch tensor.
        logits_a_fn: Callable extracting 1D logits from ``model_a``.
        logits_b_fn: Callable extracting 1D logits from ``model_b``.
    """
    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        logits_a = logits_a_fn(model_a, x).detach()
        logits_b = logits_b_fn(model_b, x).detach()

    if logits_a.shape != logits_b.shape:
        raise ValueError(
            f"Logit shape mismatch: {tuple(logits_a.shape)} vs {tuple(logits_b.shape)}."
        )

    diff = (logits_a - logits_b).reshape(-1)
    abs_diff = diff.abs()
    return {
        "max_abs_diff": float(abs_diff.max().cpu()),
        "mean_abs_diff": float(abs_diff.mean().cpu()),
        "rmse": float(torch.sqrt(torch.mean(diff * diff)).cpu()),
    }


def one_step_parity_stats(
    *,
    model_a: nn.Module,
    model_b: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    logits_a_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    logits_b_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    loss_a_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    loss_b_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    lr: float = 2e-3,
    weight_decay: float = 0.0,
) -> dict[str, float]:
    """Run one AdamW step for each model and compare parameter deltas.

    Notes:
    - The function does not mutate the caller's models.
    - Both models must expose matching state-dict keys and parameter shapes.
    """
    work_a = copy.deepcopy(model_a).train()
    work_b = copy.deepcopy(model_b).train()

    opt_a = torch.optim.AdamW(work_a.parameters(), lr=lr, weight_decay=weight_decay)
    opt_b = torch.optim.AdamW(work_b.parameters(), lr=lr, weight_decay=weight_decay)

    before_a = _clone_state_dict(work_a.state_dict())
    before_b = _clone_state_dict(work_b.state_dict())

    opt_a.zero_grad()
    loss_a = loss_a_fn(logits_a_fn(work_a, x), y)
    loss_a.backward()
    opt_a.step()

    opt_b.zero_grad()
    loss_b = loss_b_fn(logits_b_fn(work_b, x), y)
    loss_b.backward()
    opt_b.step()

    after_a = _clone_state_dict(work_a.state_dict())
    after_b = _clone_state_dict(work_b.state_dict())

    keys_a = set(before_a)
    keys_b = set(before_b)
    if keys_a != keys_b:
        raise ValueError("State dict keys differ between models.")

    step_abs_diffs: list[torch.Tensor] = []
    cross_abs_diffs: list[torch.Tensor] = []
    for key in sorted(keys_a):
        delta_a = (after_a[key] - before_a[key]).reshape(-1)
        delta_b = (after_b[key] - before_b[key]).reshape(-1)
        if delta_a.shape != delta_b.shape:
            raise ValueError(f"Parameter shape mismatch for key '{key}'.")
        step_abs_diffs.append((delta_a - delta_b).abs())
        cross_abs_diffs.append((after_a[key] - after_b[key]).abs().reshape(-1))

    all_step_abs = torch.cat(step_abs_diffs)
    all_cross_abs = torch.cat(cross_abs_diffs)
    return {
        "loss_a": float(loss_a.detach().cpu()),
        "loss_b": float(loss_b.detach().cpu()),
        "loss_abs_diff": float(abs(loss_a.detach().cpu() - loss_b.detach().cpu())),
        "max_abs_step_diff": float(all_step_abs.max().cpu()),
        "mean_abs_step_diff": float(all_step_abs.mean().cpu()),
        "max_abs_param_diff_after_step": float(all_cross_abs.max().cpu()),
    }
