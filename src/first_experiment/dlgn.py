"""DLGN model definitions.

Milestone 1 implements only DLGN_SF (shallow-features parameterization),
which corresponds to the "DLGN" model used in Paper 2.
"""

from __future__ import annotations

import torch
from torch import nn


class DLGNSF(nn.Module):
    """Deep Linearly Gated Network (shallow-features variant).

    Gating layer `l` always reads from the raw input `x`:
        gate_l(x) = sigmoid(beta * (V_l x))
    Value network has two modes:
    - `value_input_mode="ones"` (default): h_0 = 1.
    - `value_input_mode="x"`: paper-faithful deep linear value network with h_0 = x.

    Both modes use the same value-network parameters. The only difference is
    whether the initial hidden input is raw `x` or an all-ones tensor.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dims: list[int],
        beta: float = 30.0,
        bias: bool = False,
        value_input_mode: str = "ones",
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty for DLGNSF.")
        if value_input_mode not in {"ones", "x"}:
            raise ValueError(
                "value_input_mode must be one of {'ones', 'x'}, "
                f"got {value_input_mode!r}."
            )

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.beta = float(beta)
        self.value_input_mode = value_input_mode

        # Gating (DLGN-SF): each hidden layer gate is parameterized directly from x.
        self.gating_layers = nn.ModuleList(
            [nn.Linear(input_dim, width, bias=bias) for width in hidden_dims]
        )

        # Value network: same parameterization for both input modes.
        value_dims = [input_dim, *hidden_dims, 1]
        self.value_layers = nn.ModuleList(
            [
                nn.Linear(value_dims[i], value_dims[i + 1], bias=bias)
                for i in range(len(value_dims) - 1)
            ]
        )

    def effective_gating_weights(self) -> list[torch.Tensor]:
        """Return effective gating weights V^l for each hidden layer."""
        return [layer.weight.detach().clone() for layer in self.gating_layers]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (batch,)."""
        if self.value_input_mode == "ones":
            h = torch.ones_like(x)
        else:
            h = x
        for i, gate_layer in enumerate(self.gating_layers):
            gate_scores = gate_layer(x)
            gate_values = torch.sigmoid(self.beta * gate_scores)
            h = self.value_layers[i](h) * gate_values
        logits = 2*self.value_layers[-1](h).squeeze(-1) # mult by 2 to mimic the [-z,z] with cross entropy loss behaviour in legacy notebook
        return logits
