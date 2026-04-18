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
    - `value_input_mode="ones"` (default): h_0 = 1 and the first value
      transform is directly parameterized by a vector u_1 of shape (M,).
      This removes redundant parameters from an MxD matrix that would be
      used only via row sums under h_0=1.
    - `value_input_mode="x"`: paper-faithful deep linear value network with h_0 = x.
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

        if self.value_input_mode == "ones":
            self.first_value_vector = nn.Parameter(torch.empty(hidden_dims[0]))
            nn.init.normal_(self.first_value_vector, mean=0.0, std=1.0)

            # Remaining value layers: hidden-to-hidden plus final hidden-to-1.
            value_dims = [*hidden_dims, 1]
            self.value_layers = nn.ModuleList(
                [
                    nn.Linear(value_dims[i], value_dims[i + 1], bias=bias)
                    for i in range(len(value_dims) - 1)
                ]
            )
        else:
            self.first_value_vector = None
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
            # h_0 = 1, first value transform is directly u_1.
            h = self.first_value_vector.unsqueeze(0).expand(x.shape[0], -1)

            gate0 = torch.sigmoid(self.beta * self.gating_layers[0](x))
            h = h * gate0

            for i in range(1, len(self.gating_layers)):
                gate_values = torch.sigmoid(self.beta * self.gating_layers[i](x))
                h = self.value_layers[i - 1](h) * gate_values
            logits = self.value_layers[-1](h).squeeze(-1)
            return logits

        h = x
        for i, gate_layer in enumerate(self.gating_layers):
            gate_scores = gate_layer(x)
            gate_values = torch.sigmoid(self.beta * gate_scores)
            h = self.value_layers[i](h) * gate_values
        logits = self.value_layers[-1](h).squeeze(-1)
        return logits
