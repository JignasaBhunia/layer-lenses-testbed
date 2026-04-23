from first_experiment.analysis import compute_delta_vs_baseline
from first_experiment.experiment_config import DataConfig, ExperimentConfig, ModelConfig
from first_experiment.multiseed import run_multiseed


def _tiny_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="test_multipath_analysis",
        data=DataConfig(dim=10, depth=3, n_train=96, threshold=0.0),
        model=ModelConfig(hidden_dims=(8, 8), beta=10.0),
        total_epochs=4,
        phase1_epochs=2,
        lr=2e-3,
        batch_size=48,
        snapshot_stride=2,
        show_progress=False,
        gating_norm_zero_threshold=10.0,
    )


def test_compute_delta_vs_baseline_shape() -> None:
    payload = run_multiseed(
        base_config=_tiny_config(),
        master_seeds=[101, 102],
        write_artifacts=False,
    )
    summary_df = payload["summary_df"]
    delta_df = compute_delta_vs_baseline(summary_df, baseline_regime="one_phase")

    assert not delta_df.empty
    assert set(delta_df["baseline_regime"].unique()) == {"one_phase"}
    assert set(delta_df.columns) == {
        "regime",
        "split",
        "metric",
        "baseline_regime",
        "mean_delta_vs_baseline",
    }
