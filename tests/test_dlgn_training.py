import numpy as np
import torch

from first_experiment.dlgn import DLGNSF
from first_experiment.odt import generate_cob_odt_data
from first_experiment.training import (
    TrainConfig,
    effective_gating_weights_from_checkpoint,
    evaluate_dlgn_sf,
    train_dlgn_sf,
)


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


def test_dlgn_sf_same_parameter_count_across_modes() -> None:
    model_ones = DLGNSF(
        input_dim=10, hidden_dims=[6, 6], beta=10.0, bias=False, value_input_mode="ones"
    )
    model_x = DLGNSF(
        input_dim=10, hidden_dims=[6, 6], beta=10.0, bias=False, value_input_mode="x"
    )
    n_ones = sum(p.numel() for p in model_ones.parameters())
    n_x = sum(p.numel() for p in model_x.parameters())
    assert n_ones == n_x


def test_dlgn_sf_invalid_value_input_mode_raises() -> None:
    try:
        DLGNSF(input_dim=10, hidden_dims=[6, 6], value_input_mode="bad")
        assert False, "Expected ValueError for invalid value_input_mode."
    except ValueError:
        pass


def test_dlgn_sf_init_weight_scales_affect_initialization() -> None:
    torch.manual_seed(123)
    base = DLGNSF(
        input_dim=10,
        hidden_dims=[6, 6],
        beta=10.0,
        bias=False,
        gating_weight_scale=1.0,
        value_weight_scale=1.0,
    )
    torch.manual_seed(123)
    scaled = DLGNSF(
        input_dim=10,
        hidden_dims=[6, 6],
        beta=10.0,
        bias=False,
        gating_weight_scale=2.0,
        value_weight_scale=0.5,
    )

    for layer_base, layer_scaled in zip(base.gating_layers, scaled.gating_layers):
        assert torch.allclose(layer_scaled.weight, 2.0 * layer_base.weight)
    for layer_base, layer_scaled in zip(base.value_layers, scaled.value_layers):
        assert torch.allclose(layer_scaled.weight, 0.5 * layer_base.weight)


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

    ckpt = out["checkpoint_snapshots"]
    assert set(ckpt.keys()) == {0, 20}
    assert set(ckpt[0].keys()) == set(out["model"].state_dict().keys())

    gw0 = effective_gating_weights_from_checkpoint(out["model"], ckpt[0])
    assert len(gw0) == len(model.hidden_dims)
    gw20 = effective_gating_weights_from_checkpoint(out["model"], ckpt[20])
    assert all(w.shape == gw0[i].shape for i, w in enumerate(gw20))


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
