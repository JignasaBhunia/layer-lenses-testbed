"""Single-seed regime runners for multipath DLGN experiments."""

from __future__ import annotations

import copy

import numpy as np
import torch

from layer_lenses.dlgn import DLGNSF
from layer_lenses.experiment_config import (
    EvalMetrics,
    ExperimentConfig,
    PruningSummary,
    RegimeResult,
    SeedRunResult,
)
from layer_lenses.odt import generate_cob_odt_data
from layer_lenses.training import evaluate_dlgn_sf, set_seed, train_dlgn_sf


def _build_model(config: ExperimentConfig) -> DLGNSF:
    return DLGNSF(
        input_dim=config.data.dim,
        hidden_dims=list(config.model.hidden_dims),
        beta=config.model.beta,
        bias=config.model.bias,
        value_input_mode=config.model.value_input_mode,
        gating_weight_scale=config.model.gating_weight_scale,
        value_weight_scale=config.model.value_weight_scale,
    )


def _to_eval_metrics(metrics: dict[str, int | float]) -> EvalMetrics:
    return EvalMetrics(
        log_loss=float(metrics["log_loss"]),
        zero_one_loss=float(metrics["zero_one_loss"]),
        accuracy=float(metrics["accuracy"]),
        num_samples=int(metrics["num_samples"]),
    )


def _evaluate_regime(
    *,
    model: DLGNSF,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    device: str,
) -> tuple[EvalMetrics, EvalMetrics]:
    train_metrics = _to_eval_metrics(
        evaluate_dlgn_sf(model=model, x_eval=x_train, y_eval=y_train, device=device)
    )
    test_metrics = _to_eval_metrics(
        evaluate_dlgn_sf(model=model, x_eval=x_eval, y_eval=y_eval, device=device)
    )
    return train_metrics, test_metrics


def _zero_low_norm_gating_rows(
    model: DLGNSF,
    *,
    threshold: float,
) -> tuple[tuple[int, ...], tuple[torch.Tensor, ...]]:
    rows_zeroed: list[int] = []
    frozen_masks: list[torch.Tensor] = []
    with torch.no_grad():
        for layer in model.gating_layers:
            row_norms = layer.weight.norm(dim=1)
            keep_mask = row_norms >= threshold
            layer.weight[~keep_mask, :] = 0.0
            row_mask = keep_mask.float().unsqueeze(1).to(layer.weight.device)
            rows_zeroed.append(int((~keep_mask).sum().item()))
            frozen_masks.append(row_mask)
    return tuple(rows_zeroed), tuple(frozen_masks)


def _rows_still_zero(model: DLGNSF, row_masks: tuple[torch.Tensor, ...]) -> bool:
    with torch.no_grad():
        for layer, row_mask in zip(model.gating_layers, row_masks):
            frozen_rows = row_mask.squeeze(1) == 0
            if frozen_rows.any() and not torch.allclose(
                layer.weight[frozen_rows], torch.zeros_like(layer.weight[frozen_rows]), atol=1e-10
            ):
                return False
    return True


def run_one_phase(
    *,
    config: ExperimentConfig,
    model: DLGNSF,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
) -> RegimeResult:
    train_cfg = config.train_config_one_phase()
    out = train_dlgn_sf(model=model, x_train=x_train, y_train=y_train, config=train_cfg)
    trained = out["model"]
    train_metrics, test_metrics = _evaluate_regime(
        model=trained,
        x_train=x_train,
        y_train=y_train,
        x_eval=x_eval,
        y_eval=y_eval,
        device=config.device,
    )
    return RegimeResult(
        regime="one_phase",
        seed_bundle=config.seeds,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        epoch_losses=tuple(float(x) for x in out["epoch_losses"]),
        lr_history=tuple(float(x) for x in out["lr_history"]),
    )


def run_two_phase_nozero(
    *,
    config: ExperimentConfig,
    phase1_model_base: DLGNSF,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
) -> RegimeResult:
    train_cfg = config.train_config_phase2_nozero()
    out = train_dlgn_sf(
        model=copy.deepcopy(phase1_model_base),
        x_train=x_train,
        y_train=y_train,
        config=train_cfg,
    )
    trained = out["model"]
    train_metrics, test_metrics = _evaluate_regime(
        model=trained,
        x_train=x_train,
        y_train=y_train,
        x_eval=x_eval,
        y_eval=y_eval,
        device=config.device,
    )
    return RegimeResult(
        regime="two_phase_nozero",
        seed_bundle=config.seeds,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        epoch_losses=tuple(float(x) for x in out["epoch_losses"]),
        lr_history=tuple(float(x) for x in out["lr_history"]),
    )


def run_two_phase_zeroed(
    *,
    config: ExperimentConfig,
    phase1_model_base: DLGNSF,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
) -> RegimeResult:
    model = copy.deepcopy(phase1_model_base)
    rows_zeroed, row_masks = _zero_low_norm_gating_rows(
        model, threshold=config.gating_norm_zero_threshold
    )
    train_cfg = config.train_config_phase2_zeroed()
    out = train_dlgn_sf(model=model, x_train=x_train, y_train=y_train, config=train_cfg)
    trained = out["model"]
    train_metrics, test_metrics = _evaluate_regime(
        model=trained,
        x_train=x_train,
        y_train=y_train,
        x_eval=x_eval,
        y_eval=y_eval,
        device=config.device,
    )
    pruning_summary = PruningSummary(
        threshold=config.gating_norm_zero_threshold,
        rows_zeroed_per_layer=rows_zeroed,
        total_rows_zeroed=int(sum(rows_zeroed)),
        rows_still_zero_after_training=_rows_still_zero(trained, row_masks),
    )
    return RegimeResult(
        regime="two_phase_zeroed",
        seed_bundle=config.seeds,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        epoch_losses=tuple(float(x) for x in out["epoch_losses"]),
        lr_history=tuple(float(x) for x in out["lr_history"]),
        pruning_summary=pruning_summary,
    )


def run_all_regimes_for_seed(config: ExperimentConfig) -> SeedRunResult:
    """Run one-phase and both two-phase variants for a single master seed."""
    x, y, _, meta = generate_cob_odt_data(
        num_data=config.data.num_data,
        dim=config.data.dim,
        depth=config.data.depth,
        seed=config.seeds.data_seed,
        threshold=config.data.threshold,
    )
    x_train = x[: config.data.n_train]
    y_train = y[: config.data.n_train]
    x_eval = x[config.data.n_train :]
    y_eval = y[config.data.n_train :]

    set_seed(config.seeds.init_seed)
    phase1_model = _build_model(config)
    phase1_out = train_dlgn_sf(
        model=phase1_model,
        x_train=x_train,
        y_train=y_train,
        config=config.train_config_phase1(),
    )
    phase1_model_base = phase1_out["model"]

    set_seed(config.seeds.init_seed)
    one_phase_model = _build_model(config)

    one_phase = run_one_phase(
        config=config,
        model=one_phase_model,
        x_train=x_train,
        y_train=y_train,
        x_eval=x_eval,
        y_eval=y_eval,
    )
    two_nozero = run_two_phase_nozero(
        config=config,
        phase1_model_base=phase1_model_base,
        x_train=x_train,
        y_train=y_train,
        x_eval=x_eval,
        y_eval=y_eval,
    )
    two_zeroed = run_two_phase_zeroed(
        config=config,
        phase1_model_base=phase1_model_base,
        x_train=x_train,
        y_train=y_train,
        x_eval=x_eval,
        y_eval=y_eval,
    )
    return SeedRunResult(
        experiment_name=config.experiment_name,
        seed_bundle=config.seeds,
        train_size=int(x_train.shape[0]),
        eval_size=int(x_eval.shape[0]),
        regimes={
            one_phase.regime: one_phase,
            two_nozero.regime: two_nozero,
            two_zeroed.regime: two_zeroed,
        },
        data_meta={k: int(v) if isinstance(v, np.integer) else v for k, v in meta.items()},
    )
