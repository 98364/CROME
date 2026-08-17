"""E-CS5: wall-time, peak-allocation, and approximation scaling benchmark."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
import tracemalloc
from typing import Any

import numpy as np

from crome_identification.certification import certify_overlap_target, decide_target
from crome_identification.seeding import make_seed_bundle

from ._common import load_exp_config, mode_reps, save_raw_and_summary


@dataclass(frozen=True)
class ScalingSpec:
    n_trajectories: int
    parameter_dim: int
    events_per_trajectory: int
    endpoints_per_trajectory: int

    @property
    def n_rows(self) -> int:
        return self.n_trajectories * self.endpoints_per_trajectory

    @property
    def key(self) -> str:
        return (
            f"n={self.n_trajectories}:p={self.parameter_dim}:"
            f"events={self.events_per_trajectory}:endpoints={self.endpoints_per_trajectory}"
        )


def _scaling_specs(cfg: dict[str, Any], mode: str) -> list[ScalingSpec]:
    values: dict[str, ScalingSpec] = {}
    for n in cfg[f"n_trajectories_{mode}"]:
        for p in cfg[f"parameter_dims_{mode}"]:
            spec = ScalingSpec(int(n), int(p), int(cfg["fixed_events"]), int(cfg["fixed_endpoints"]))
            values[spec.key] = spec
    for events in cfg[f"events_per_trajectory_{mode}"]:
        for endpoints in cfg[f"endpoints_per_trajectory_{mode}"]:
            spec = ScalingSpec(
                int(cfg["fixed_n_trajectories"]), int(cfg["fixed_parameter_dim"]),
                int(events), int(endpoints),
            )
            values[spec.key] = spec
    return sorted(values.values(), key=lambda spec: (spec.n_rows, spec.parameter_dim, spec.events_per_trajectory))


def _build_design(spec: ScalingSpec, rng: np.random.Generator):
    started = perf_counter()
    design = np.zeros((spec.n_rows, spec.parameter_dim), dtype=float)
    for _ in range(spec.events_per_trajectory):
        design += rng.normal(size=design.shape)
    design /= math.sqrt(spec.events_per_trajectory)
    return design, perf_counter() - started


def _measure_certificate(design: np.ndarray, outcomes: np.ndarray, target: np.ndarray,
                         theta_radius: float, *, exact: bool,
                         approximation_budget: float, scientific_tolerance: float):
    gc.collect()
    tracemalloc.start()
    started = perf_counter()
    certificate = certify_overlap_target(
        design, outcomes, target, noise_radius=0.0, design_error_bound=0.0,
        theta_radius=theta_radius, approximation_error=0.0 if exact else approximation_budget,
        exact_design=exact,
        deterministic_uncertainty=not exact,
        target_id="scalar_target",
        provenance="CS05 deterministic implementation-scaling fixture",
    )
    decision = decide_target([certificate], scientific_tolerance=scientific_tolerance)
    runtime = perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "status": decision.status.value,
        "point_estimate": decision.point_estimate,
        "runtime_seconds": runtime,
        "peak_python_bytes": int(peak),
        "target_residual": certificate.diagnostics["target_residual"],
        "target_amplification": certificate.diagnostics["target_amplification"],
        "certificate_scope": decision.certificate_scope.value,
    }


def _run_spec(cfg: dict[str, Any], spec: ScalingSpec, rep: int, mode: str):
    bundle = make_seed_bundle(int(cfg["master_seed"]), "cs05", f"{mode}:{spec.key}", rep)
    rng = bundle.rng
    design, build_runtime = _build_design(spec, rng)
    theta = np.linspace(1.0, -0.5, spec.parameter_dim)
    target = np.zeros(spec.parameter_dim); target[0] = 1.0
    outcomes = design @ theta
    exact = _measure_certificate(
        design, outcomes, target, float(np.linalg.norm(theta)), exact=True,
        approximation_budget=float(cfg["approximation_budget"]),
        scientific_tolerance=float(cfg["scientific_tolerance"]),
    )
    approximate_rows = min(
        spec.n_rows,
        max(int(cfg["approximation_min_rows"]), int(cfg["approximation_row_multiplier"]) * spec.parameter_dim),
    )
    selected = np.sort(rng.choice(spec.n_rows, size=approximate_rows, replace=False))
    approximate = _measure_certificate(
        design[selected], outcomes[selected], target, float(np.linalg.norm(theta)), exact=False,
        approximation_budget=float(cfg["approximation_budget"]),
        scientific_tolerance=float(cfg["scientific_tolerance"]),
    )
    discrepancy = abs(float(exact["point_estimate"]) - float(approximate["point_estimate"]))
    return {
        "rep": rep, "spec_key": spec.key, "n_trajectories": spec.n_trajectories,
        "parameter_dim": spec.parameter_dim, "events_per_trajectory": spec.events_per_trajectory,
        "endpoints_per_trajectory": spec.endpoints_per_trajectory, "n_operator_rows": spec.n_rows,
        "design_build_runtime_seconds": build_runtime,
        "input_matrix_bytes": int(design.nbytes + outcomes.nbytes + target.nbytes),
        "approximate_rows": approximate_rows, "exact": exact, "approximate": approximate,
        "status_agreement": exact["status"] == approximate["status"],
        "target_discrepancy": discrepancy,
    }


def _slope(x: list[float], y: list[float]) -> float:
    x_array = np.asarray(x, dtype=float); y_array = np.asarray(y, dtype=float)
    positive = (x_array > 0) & (y_array > 0)
    if np.sum(positive) < 2:
        return 0.0
    return float(np.polyfit(np.log(x_array[positive]), np.log(y_array[positive]), 1)[0])


def _empirical_slopes(records: list[dict[str, Any]]):
    rows = [float(row["n_operator_rows"]) for row in records]
    exact_times = [float(row["exact"]["runtime_seconds"]) for row in records]
    memory = [float(row["input_matrix_bytes"] + row["exact"]["peak_python_bytes"]) for row in records]
    dims = [float(row["parameter_dim"]) for row in records]
    return {
        "runtime_vs_rows": _slope(rows, exact_times),
        "memory_vs_rows": _slope(rows, memory),
        "runtime_vs_parameter_dim": _slope(dims, exact_times),
        "interpretation": "descriptive log-log slopes on the configured finite grid; not asymptotic complexity theorems",
    }


def _finite(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int, np.integer)):
        return True
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def _gate(cfg: dict[str, Any], records: list[dict[str, Any]], expected: int):
    agreement = sum(row["status_agreement"] for row in records) / len(records)
    maximum_discrepancy = max(row["target_discrepancy"] for row in records)
    checks = {
        "cell_completeness": {"passed": len(records) == expected, "observed": len(records), "required": expected},
        "finite_runtime_memory": {"passed": _finite(records), "observed": _finite(records), "required": True},
        "status_agreement": {"passed": agreement >= float(cfg["gate_min_status_agreement"]), "observed": agreement, "required": float(cfg["gate_min_status_agreement"])},
        "approximation_budget": {"passed": maximum_discrepancy <= float(cfg["approximation_budget"]), "observed": maximum_discrepancy, "required": float(cfg["approximation_budget"])},
    }
    return {"passed": all(item["passed"] for item in checks.values()), "checks": checks}


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write_csv(records: list[dict[str, Any]], config_hash: str, path: Path):
    rows = [{
        "rep": row["rep"], "n_trajectories": row["n_trajectories"],
        "parameter_dim": row["parameter_dim"], "events_per_trajectory": row["events_per_trajectory"],
        "endpoints_per_trajectory": row["endpoints_per_trajectory"], "n_operator_rows": row["n_operator_rows"],
        "design_build_runtime_seconds": row["design_build_runtime_seconds"],
        "exact_runtime_seconds": row["exact"]["runtime_seconds"],
        "approximate_runtime_seconds": row["approximate"]["runtime_seconds"],
        "exact_peak_python_bytes": row["exact"]["peak_python_bytes"],
        "approximate_peak_python_bytes": row["approximate"]["peak_python_bytes"],
        "input_matrix_bytes": row["input_matrix_bytes"], "target_discrepancy": row["target_discrepancy"],
        "status_agreement": int(row["status_agreement"]), "config_sha256": config_hash,
    } for row in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def run(mode: str = "smoke", outdir: Path | None = None) -> dict[str, Any]:
    cfg = load_exp_config("cs05_scaling"); cfg["mode"] = mode
    specs = _scaling_specs(cfg, mode); n_reps = mode_reps(cfg, mode)
    warm_rng = np.random.default_rng(0)
    warm_design = warm_rng.normal(size=(64, 3)); warm_target = np.array([1.0, 0.0, 0.0])
    _measure_certificate(warm_design, warm_design @ np.array([1.0, 0.2, -0.1]), warm_target, 1.1,
                         exact=True, approximation_budget=float(cfg["approximation_budget"]),
                         scientific_tolerance=float(cfg["scientific_tolerance"]))
    records = [_run_spec(cfg, spec, rep, mode) for rep in range(n_reps) for spec in specs]
    config_hash = _hash_json(cfg); slopes = _empirical_slopes(records)
    keys = [(row["rep"], row["spec_key"]) for row in records]
    result = {
        "experiment": "cs05_scaling", "mode": mode, "master_seed": int(cfg["master_seed"]),
        "config_sha256": config_hash, "n_reps": n_reps, "n_cells": len(specs),
        "measurement_protocol": "single process; one warm-up excluded; perf_counter; tracemalloc peak plus explicit input bytes",
        "empirical_slopes": slopes, "gate": _gate(cfg, records, n_reps * len(specs)),
        "artifact_audit": {"unique_keys": len(keys) == len(set(keys)), "strict_json": True},
        "config": cfg, "replication_records": records,
    }
    json.dumps(result, allow_nan=False)
    if outdir is not None:
        outdir = Path(outdir); save_raw_and_summary(result, outdir / f"cs05_{mode}.json")
        source = outdir.parent / "source_data" / f"cs05_{mode}.csv" if outdir.name == "raw" else outdir / f"cs05_{mode}_source.csv"
        _write_csv(records, config_hash, source)
    return result


if __name__ == "__main__":
    print(json.dumps(run("smoke", Path("results/raw"))["gate"], indent=2))
