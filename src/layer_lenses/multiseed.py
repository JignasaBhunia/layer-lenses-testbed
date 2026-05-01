"""API-first multi-seed orchestration for multipath DLGN."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from layer_lenses.analysis import aggregate_descriptive_stats, compute_delta_vs_baseline
from layer_lenses.experiment_config import ExperimentConfig, SeedRunResult
from layer_lenses.experiment_runner import run_all_regimes_for_seed
from layer_lenses.io import ensure_run_root, write_run_manifest, write_seed_result, write_summary_rows


def run_multiseed(
    *,
    base_config: ExperimentConfig,
    master_seeds: list[int],
    write_artifacts: bool = False,
    results_dir: str | Path = "results",
    run_tag: str | None = None,
) -> dict[str, object]:
    """Run all regimes over many master seeds and optionally persist artifacts."""
    seed_results: list[SeedRunResult] = []
    for master_seed in master_seeds:
        cfg = base_config.with_master_seed(master_seed)
        seed_results.append(run_all_regimes_for_seed(cfg))

    summary_df = aggregate_descriptive_stats(seed_results)
    delta_df = compute_delta_vs_baseline(summary_df, baseline_regime="one_phase")
    payload: dict[str, object] = {
        "config": asdict(base_config),
        "master_seeds": master_seeds,
        "seed_results": seed_results,
        "summary_df": summary_df,
        "delta_df": delta_df,
    }

    if write_artifacts:
        run_root = ensure_run_root(
            experiment_name=base_config.experiment_name, results_dir=results_dir, run_tag=run_tag
        )
        write_run_manifest(
            run_root=run_root,
            config=base_config,
            extra_meta={"master_seeds": master_seeds},
        )
        for result in seed_results:
            write_seed_result(run_root=run_root, seed_result=result)
        write_summary_rows(
            run_root=run_root,
            rows=summary_df.to_dict(orient="records"),
            filename="summary_descriptive.csv",
        )
        write_summary_rows(
            run_root=run_root,
            rows=delta_df.to_dict(orient="records"),
            filename="summary_delta_vs_one_phase.csv",
        )
        payload["run_root"] = run_root

    return payload
