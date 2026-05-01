from pathlib import Path

from layer_lenses.analysis import aggregate_descriptive_stats
from layer_lenses.experiment_config import DataConfig, ExperimentConfig, ModelConfig
from layer_lenses.multiseed import run_multiseed


def _tiny_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="test_multipath",
        data=DataConfig(dim=10, depth=3, n_train=128, threshold=0.0),
        model=ModelConfig(hidden_dims=(8, 8), beta=10.0),
        total_epochs=6,
        phase1_epochs=3,
        lr=2e-3,
        batch_size=64,
        snapshot_stride=3,
        show_progress=False,
        gating_norm_zero_threshold=10.0,
    )


def test_run_multiseed_produces_all_regimes_and_seed_fairness() -> None:
    cfg = _tiny_config()
    payload = run_multiseed(base_config=cfg, master_seeds=[11, 12], write_artifacts=False)
    seed_results = payload["seed_results"]
    assert len(seed_results) == 2

    for seed_result in seed_results:
        assert set(seed_result.regimes.keys()) == {
            "one_phase",
            "two_phase_nozero",
            "two_phase_zeroed",
        }
        for regime in seed_result.regimes.values():
            assert regime.seed_bundle.master_seed == seed_result.seed_bundle.master_seed
            assert regime.seed_bundle.data_seed == seed_result.seed_bundle.data_seed
            assert regime.seed_bundle.init_seed == seed_result.seed_bundle.init_seed


def test_two_phase_zeroed_reports_pruning_invariants() -> None:
    cfg = _tiny_config().with_master_seed(21)
    payload = run_multiseed(base_config=cfg, master_seeds=[21], write_artifacts=False)
    seed_result = payload["seed_results"][0]
    zeroed = seed_result.regimes["two_phase_zeroed"]
    assert zeroed.pruning_summary is not None
    assert zeroed.pruning_summary.total_rows_zeroed > 0
    assert zeroed.pruning_summary.rows_still_zero_after_training


def test_descriptive_summary_schema_and_counts() -> None:
    cfg = _tiny_config()
    payload = run_multiseed(base_config=cfg, master_seeds=[31, 32], write_artifacts=False)
    summary = aggregate_descriptive_stats(payload["seed_results"])
    expected_columns = {
        "regime",
        "split",
        "metric",
        "n",
        "mean",
        "std",
        "sem",
        "ci95_low",
        "ci95_high",
    }
    assert expected_columns.issubset(set(summary.columns))
    assert summary["n"].nunique() == 1
    assert int(summary["n"].iloc[0]) == 2


def test_multiseed_artifacts_written(tmp_path: Path) -> None:
    cfg = _tiny_config()
    payload = run_multiseed(
        base_config=cfg,
        master_seeds=[41],
        write_artifacts=True,
        results_dir=tmp_path,
        run_tag="testrun",
    )
    run_root = payload["run_root"]
    assert (run_root / "manifest.json").exists()
    assert (run_root / "summary_descriptive.csv").exists()
    assert (run_root / "summary_delta_vs_one_phase.csv").exists()
    assert (run_root / "seeds" / "master_seed=41" / "result.json").exists()
