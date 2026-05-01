"""Standard fully-connected ReLU network for binary classification."""

from __future__ import annotations

import torch
from torch import nn


class ReLUMLP(nn.Module):
    """Simple MLP that returns scalar logits of shape (batch,)."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dims: list[int],
        bias: bool = False,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty.")

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.hidden_layers = nn.ModuleList()

        prev = input_dim
        for width in hidden_dims:
            self.hidden_layers.append(nn.Linear(prev, width, bias=bias))
            prev = width
        self.output_layer = nn.Linear(prev, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.hidden_layers:
            h = torch.relu(layer(h))
        return self.output_layer(h).squeeze(-1)
