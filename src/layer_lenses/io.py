"""Result IO helpers for experiment runs and summaries."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from layer_lenses.experiment_config import ExperimentConfig, SeedRunResult


def ensure_run_root(
    *,
    experiment_name: str,
    results_dir: str | Path = "results",
    run_tag: str | None = None,
) -> Path:
    """Create and return a deterministic run root for artifacts."""
    tag = run_tag or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_root = Path(results_dir) / experiment_name / tag
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def write_run_manifest(
    *,
    run_root: Path,
    config: ExperimentConfig,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Write top-level reproducibility manifest for a run group."""
    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment_name": config.experiment_name,
        "config": asdict(config),
        "extra_meta": extra_meta or {},
    }
    out_path = run_root / "manifest.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def write_seed_result(
    *,
    run_root: Path,
    seed_result: SeedRunResult,
) -> Path:
    """Write one seed's regime outputs to JSON."""
    out_dir = run_root / "seeds" / f"master_seed={seed_result.seed_bundle.master_seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "result.json"
    out_path.write_text(json.dumps(seed_result.to_dict(), indent=2), encoding="utf-8")
    return out_path


def write_summary_rows(
    *,
    run_root: Path,
    rows: list[dict[str, Any]],
    filename: str = "summary.csv",
) -> Path:
    """Write list-of-dicts summary to CSV in run root."""
    out_path = run_root / filename
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return out_path
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path
