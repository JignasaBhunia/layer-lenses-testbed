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


def test_evaluate_relu_mlp_metrics_shape() -> None:
    x = np.random.randn(64, 10).astype(np.float32)
    y = np.where(np.random.randn(64) > 0, 1, -1).astype(np.int8)
    model = ReLUMLP(input_dim=10, hidden_dims=[16, 16], bias=False)
    with torch.no_grad():
        _ = model(torch.from_numpy(x))
    metrics = evaluate_relu_mlp(model=model, x_eval=x, y_eval=y)
    assert set(metrics.keys()) == {"log_loss", "zero_one_loss", "accuracy", "num_samples"}
    assert metrics["num_samples"] == 64
