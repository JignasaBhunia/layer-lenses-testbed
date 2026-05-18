import numpy as np
import torch

from layer_lenses.relu_mlp import ReLUMLP
from layer_lenses.relu_training import (
    ReLUTrainConfig,
    evaluate_relu_mlp,
    train_relu_mlp,
)


def test_train_relu_mlp_smoke() -> None:
    x = np.random.randn(128, 10).astype(np.float32)
    y = np.where(np.random.randn(128) > 0, 1, -1).astype(np.int8)
    model = ReLUMLP(input_dim=10, hidden_dims=[16, 16], bias=False)

    out = train_relu_mlp(
        model=model,
        x_train=x,
        y_train=y,
        config=ReLUTrainConfig(
            epochs=4,
            lr=1e-3,
            batch_size=32,
            seed=123,
            snapshot_epochs=(0, 2, 4),
        ),
    )
    assert len(out["epoch_losses"]) == 4
    assert set(out["checkpoint_snapshots"].keys()) == {0, 2, 4}


def test_train_relu_mlp_sgd_smoke() -> None:
    x = np.random.randn(128, 10).astype(np.float32)
    y = np.where(np.random.randn(128) > 0, 1, -1).astype(np.int8)
    model = ReLUMLP(input_dim=10, hidden_dims=[16, 16], bias=False)

    out = train_relu_mlp(
        model=model,
        x_train=x,
        y_train=y,
        config=ReLUTrainConfig(
            epochs=4,
            lr=1e-2,
            batch_size=32,
            seed=456,
            snapshot_epochs=(0, 2, 4),
            optimizer="sgd",
        ),
    )
    assert len(out["epoch_losses"]) == 4
    assert set(out["checkpoint_snapshots"].keys()) == {0, 2, 4}


def test_first_layer_gradient_projection_keeps_sgd_update_orthogonal() -> None:
    x = np.random.randn(128, 10).astype(np.float32)
    y = np.where(np.random.randn(128) > 0, 1, -1).astype(np.int8)
    model = ReLUMLP(input_dim=10, hidden_dims=[16, 16], bias=False)
    projection_vector = np.zeros(10, dtype=np.float32)
    projection_vector[0] = 1.0

    out = train_relu_mlp(
        model=model,
        x_train=x,
        y_train=y,
        config=ReLUTrainConfig(
            epochs=1,
            lr=1e-2,
            batch_size=x.shape[0],
            seed=789,
            snapshot_epochs=(0, 1),
            optimizer="sgd",
            first_layer_grad_orthogonal_to=projection_vector,
        ),
    )
    w0_start = out["checkpoint_snapshots"][0]["hidden_layers.0.weight"]
    w0_end = out["checkpoint_snapshots"][1]["hidden_layers.0.weight"]
    delta = w0_end - w0_start

    assert torch.allclose(delta @ torch.from_numpy(projection_vector), torch.zeros(16), atol=1e-7)


def test_parameter_update_mask_freezes_individual_entries() -> None:
    x = np.ones((64, 4), dtype=np.float32)
    y = np.ones(64, dtype=np.int8)
    model = ReLUMLP(input_dim=4, hidden_dims=[3], bias=False)
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(0.1)

    mask = {name: torch.ones_like(param) for name, param in model.named_parameters()}
    mask["hidden_layers.0.weight"][0, 0] = 0

    out = train_relu_mlp(
        model=model,
        x_train=x,
        y_train=y,
        config=ReLUTrainConfig(
            epochs=1,
            lr=1e-1,
            batch_size=x.shape[0],
            seed=101,
            snapshot_epochs=(0, 1),
            optimizer="sgd",
            parameter_update_mask=mask,
        ),
    )
    w0_start = out["checkpoint_snapshots"][0]["hidden_layers.0.weight"]
    w0_end = out["checkpoint_snapshots"][1]["hidden_layers.0.weight"]

    assert w0_end[0, 0] == w0_start[0, 0]
    assert not torch.allclose(w0_end[0, 1], w0_start[0, 1])


def test_evaluate_relu_mlp_metrics_shape() -> None:
    x = np.random.randn(64, 10).astype(np.float32)
    y = np.where(np.random.randn(64) > 0, 1, -1).astype(np.int8)
    model = ReLUMLP(input_dim=10, hidden_dims=[16, 16], bias=False)
    with torch.no_grad():
        _ = model(torch.from_numpy(x))
    metrics = evaluate_relu_mlp(model=model, x_eval=x, y_eval=y)
    assert set(metrics.keys()) == {"log_loss", "zero_one_loss", "accuracy", "num_samples"}
    assert metrics["num_samples"] == 64
