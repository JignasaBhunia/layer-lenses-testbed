"""Descriptive statistics for multipath multi-seed results."""

from __future__ import annotations

import math

import pandas as pd

from first_experiment.experiment_config import RegimeResult, SeedRunResult


_METRIC_NAMES = ("log_loss", "zero_one_loss", "accuracy")
_Z95 = 1.959963984540054


def _extract_metric_rows(seed_results: list[SeedRunResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed_result in seed_results:
        for regime_name, regime in seed_result.regimes.items():
            rows.extend(_rows_for_regime(regime_name=regime_name, regime=regime))
    return rows


def _rows_for_regime(*, regime_name: str, regime: RegimeResult) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for split_name, metrics in (
        ("train", regime.train_metrics),
        ("test", regime.test_metrics),
    ):
        for metric_name in _METRIC_NAMES:
            out.append(
                {
                    "master_seed": regime.seed_bundle.master_seed,
                    "regime": regime_name,
                    "split": split_name,
                    "metric": metric_name,
                    "value": float(getattr(metrics, metric_name)),
                }
            )
    return out


def aggregate_descriptive_stats(seed_results: list[SeedRunResult]) -> pd.DataFrame:
    """Compute mean/std/SEM/95% CI (normal approx) across seeds per metric."""
    raw_rows = _extract_metric_rows(seed_results)
    if not raw_rows:
        return pd.DataFrame(
            columns=[
                "regime",
                "split",
                "metric",
                "n",
                "mean",
                "std",
                "sem",
                "ci95_low",
                "ci95_high",
            ]
        )

    df = pd.DataFrame(raw_rows)
    grouped = df.groupby(["regime", "split", "metric"], sort=True)["value"]
    agg = grouped.agg(n="count", mean="mean", std=lambda s: s.std(ddof=1)).reset_index()
    agg["std"] = agg["std"].fillna(0.0)
    agg["sem"] = agg.apply(
        lambda row: float(row["std"]) / math.sqrt(int(row["n"])) if int(row["n"]) > 0 else 0.0,
        axis=1,
    )
    agg["ci95_low"] = agg["mean"] - (_Z95 * agg["sem"])
    agg["ci95_high"] = agg["mean"] + (_Z95 * agg["sem"])
    return agg[
        ["regime", "split", "metric", "n", "mean", "std", "sem", "ci95_low", "ci95_high"]
    ].sort_values(["split", "metric", "regime"], ignore_index=True)


def compute_delta_vs_baseline(
    summary_df: pd.DataFrame,
    *,
    baseline_regime: str = "one_phase",
) -> pd.DataFrame:
    """Compute mean deltas relative to baseline regime for each split/metric."""
    if summary_df.empty:
        return pd.DataFrame(
            columns=["regime", "split", "metric", "baseline_regime", "mean_delta_vs_baseline"]
        )

    baseline = summary_df[summary_df["regime"] == baseline_regime][["split", "metric", "mean"]]
    baseline = baseline.rename(columns={"mean": "baseline_mean"})
    merged = summary_df.merge(baseline, on=["split", "metric"], how="left")
    merged["mean_delta_vs_baseline"] = merged["mean"] - merged["baseline_mean"]
    out = merged[["regime", "split", "metric", "mean_delta_vs_baseline"]].copy()
    out["baseline_regime"] = baseline_regime
    return out[["regime", "split", "metric", "baseline_regime", "mean_delta_vs_baseline"]]
