import numpy as np
import torch

from first_experiment.dlgn import DLGNSF
from first_experiment.odt import generate_cob_odt_data
from first_experiment.training import TrainConfig, evaluate_dlgn_sf, train_dlgn_sf


def test_dlgn_sf_forward_shape() -> None:
    model = DLGNSF(input_dim=10, hidden_dims=[6, 6], beta=10.0, bias=False)
    x = torch.randn(7, 10)
    logits = model(x)
    assert logits.shape == (7,)


def test_dlgn_sf_forward_shape_x_mode() -> None:
    model = DLGNSF(
        input_dim=10, hidden_dims=[6, 6], beta=10.0, bias=False, value_input_mode="x"
    )
    x = torch.randn(7, 10)
    logits = model(x)
    assert logits.shape == (7,)


def test_dlgn_sf_invalid_value_input_mode_raises() -> None:
    try:
        DLGNSF(input_dim=10, hidden_dims=[6, 6], value_input_mode="bad")
        assert False, "Expected ValueError for invalid value_input_mode."
    except ValueError:
        pass


def test_train_dlgn_sf_smoke_loss_decreases() -> None:
    x, y, _, _ = generate_cob_odt_data(
        num_data=512,
        dim=10,
        depth=3,
        seed=365,
        threshold=0.0,
    )
    model = DLGNSF(input_dim=10, hidden_dims=[8, 8], beta=10.0, bias=False)
    config = TrainConfig(
        epochs=20,
        lr=2e-3,
        batch_size=128,
        seed=365,
        device="cpu",
        snapshot_epochs=(0, 20),
    )
    out = train_dlgn_sf(model=model, x_train=x, y_train=y, config=config)
    losses = out["epoch_losses"]
    assert isinstance(losses, list)
    assert len(losses) == config.epochs
    assert losses[-1] < losses[0]

    snapshots = out["snapshots"]
    assert 0 in snapshots and 20 in snapshots
    assert len(snapshots[0]) == len(model.hidden_dims)


def test_evaluate_dlgn_sf_reports_metrics() -> None:
    x, y, _, _ = generate_cob_odt_data(
        num_data=256,
        dim=10,
        depth=3,
        seed=42,
        threshold=0.0,
    )
    model = DLGNSF(input_dim=10, hidden_dims=[8, 8], beta=10.0, bias=False)
    cfg = TrainConfig(
        epochs=5,
        lr=2e-3,
        batch_size=64,
        seed=42,
        device="cpu",
        show_progress=False,
    )
    out = train_dlgn_sf(model=model, x_train=x, y_train=y, config=cfg)
    trained_model = out["model"]

    metrics = evaluate_dlgn_sf(model=trained_model, x_eval=x, y_eval=y, device="cpu")
    assert set(metrics.keys()) == {"log_loss", "zero_one_loss", "accuracy", "num_samples"}
    assert metrics["num_samples"] == x.shape[0]
    assert metrics["log_loss"] >= 0.0
    assert 0.0 <= metrics["zero_one_loss"] <= 1.0
    assert 0.0 <= metrics["accuracy"] <= 1.0
