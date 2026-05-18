import math

import torch

from layer_lenses.relu_mlp import ReLUMLP


def test_relu_mlp_uses_gaussian_init_with_default_uniform_variance() -> None:
    torch.manual_seed(123)
    fan_in = 128
    model = ReLUMLP(input_dim=fan_in, hidden_dims=[4096], bias=True)
    weights = model.hidden_layers[0].weight.detach()

    expected_var = 1.0 / (3.0 * fan_in)
    sample_var = float(weights.var(unbiased=False))
    default_uniform_bound = 1.0 / math.sqrt(fan_in)

    assert abs(sample_var - expected_var) / expected_var < 0.05
    assert bool((weights.abs() > default_uniform_bound).any())
