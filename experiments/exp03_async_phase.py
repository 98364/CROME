"""Experiment 3: lag-support geometry and sample-size scaling.

The main scan compares a continuous-uniform phase design with a structural
positive-gap design.  Aligned and discrete phases are retained only in the
single-design geometry fixture used by the legacy support figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from crome_identification.inference.boundary_regression import (
    one_sided_local_linear,
    select_bandwidth_group_kfold,
)
from crome_identification.observation.lag_support import first_forward_recurrence
from crome_identification.responses.equivalence import response_with_jump
from crome_identification.seeding import make_seed_bundle

from ._common import load_exp_config, mode_reps, save_raw_and_summary


def _sample_phases(design: str, n: int, Delta: float, rng: np.random.Generator) -> np.ndarray:
    if design == "aligned":
        return np.zeros(n)
    if design == "discrete":
        grid = np.array([0.0, 0.25, 0.5, 0.75]) * Delta
        return rng.choice(grid, size=n)
    if design == "continuous_uniform":
        return rng.uniform(0.0, Delta, size=n)
    if design == "gap_near_zero":
        # R = Delta - phase is supported on [0.2 Delta, Delta].
        return rng.uniform(0.0, 0.8 * Delta, size=n)
    raise ValueError(design)


def _trajectory_count(cfg: dict[str, Any], mode: str) -> int:
    fallback = max(_sample_size_grid(cfg, mode))
    return int(cfg.get(f"n_traj_{mode}", fallback))


def _sample_size_grid(cfg: dict[str, Any], mode: str) -> list[int]:
    defaults = {
        "smoke": [50, 100],
        "dev": [100, 250, 500],
        "main": [100, 250, 500, 1000, 2000],
    }
    return [int(value) for value in cfg.get(f"n_grid_{mode}", defaults[mode])]


def _simulate_one(
    *,
    design: str,
    n: int,
    Delta: float,
    J: float,
    sigma: float,
    bandwidth_grid: list[float],
    near_zero_threshold: float,
    master_seed: int,
    config_id: str,
    rep: int,
) -> dict[str, Any]:
    rng = make_seed_bundle(master_seed, "exp03", config_id, rep).rng
    phases = _sample_phases(design, n, Delta, rng)
    first_lags = np.asarray(
        [first_forward_recurrence(float(phase), Delta) for phase in phases],
        dtype=float,
    )

    lags_parts: list[np.ndarray] = []
    outcome_parts: list[np.ndarray] = []
    trajectory_parts: list[np.ndarray] = []
    for i, phase in enumerate(phases):
        obs_times = np.arange(1, 6, dtype=float) * Delta
        lags_i = obs_times - float(phase)
        lags_i = lags_i[lags_i > 0]
        outcomes_i = response_with_jump(lags_i, J) + rng.normal(0.0, sigma, size=lags_i.size)
        lags_parts.append(lags_i)
        outcome_parts.append(np.asarray(outcomes_i, dtype=float))
        trajectory_parts.append(np.full(lags_i.size, i, dtype=int))

    lags = np.concatenate(lags_parts) if lags_parts else np.array([], dtype=float)
    outcomes = np.concatenate(outcome_parts) if outcome_parts else np.array([], dtype=float)
    trajectory_ids = (
        np.concatenate(trajectory_parts) if trajectory_parts else np.array([], dtype=int)
    )
    q_min = float(np.min(lags)) if lags.size else np.inf
    n_near = int(np.sum(lags <= near_zero_threshold * Delta))
    applicable = design == "continuous_uniform"
    estimate = None
    bandwidth = None
    nonfinite = False
    if applicable:
        bandwidth = select_bandwidth_group_kfold(
            lags,
            outcomes,
            trajectory_ids,
            bandwidth_grid,
            n_splits=5,
        )
        fit = one_sided_local_linear(lags, outcomes, bandwidth)
        estimate = float(fit["estimate"])
        nonfinite = not np.isfinite(estimate)
        if nonfinite:
            estimate = None

    return {
        "rep": rep,
        "n": n,
        "design": design,
        "q_min": q_min,
        "n_near_zero": n_near,
        "n_lags": int(lags.size),
        "first_lags": first_lags,
        "estimation_applicable": applicable,
        "point_estimate": estimate,
        "bandwidth": bandwidth,
        "nonfinite": nonfinite,
        "true_J": J,
    }


def _distribution_summary(values: list[float]) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"mean": None, "median": None, "q05": None, "q25": None, "q75": None, "q95": None}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "q05": float(np.quantile(finite, 0.05)),
        "q25": float(np.quantile(finite, 0.25)),
        "q75": float(np.quantile(finite, 0.75)),
        "q95": float(np.quantile(finite, 0.95)),
    }


def _summarize_records(records: list[dict[str, Any]], J: float) -> dict[str, Any]:
    estimates = np.asarray(
        [item["point_estimate"] for item in records if item["point_estimate"] is not None],
        dtype=float,
    )
    applicable = bool(records[0]["estimation_applicable"])
    if estimates.size:
        errors = estimates - J
        estimator = {
            "bias": float(np.mean(errors)),
            "sd": float(np.std(estimates, ddof=1)) if estimates.size > 1 else 0.0,
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "estimate_distribution": _distribution_summary(estimates.tolist()),
        }
    else:
        estimator = None
    return {
        "n": int(records[0]["n"]),
        "n_reps": len(records),
        "estimation_applicable": applicable,
        "q_min": _distribution_summary([item["q_min"] for item in records]),
        "near_zero_count": _distribution_summary(
            [float(item["n_near_zero"]) for item in records]
        ),
        "bandwidth": _distribution_summary(
            [item["bandwidth"] for item in records if item["bandwidth"] is not None]
        ),
        "failure_frequency": (
            float(np.mean([item["nonfinite"] for item in records])) if applicable else None
        ),
        "estimator": estimator,
    }


def run(mode: str = "smoke", outdir: Path | None = None) -> dict[str, Any]:
    cfg = load_exp_config("lag_support")
    cfg["mode"] = mode
    Delta = float(cfg.get("Delta", 1.0))
    J = float(cfg.get("J_a", 1.0))
    sigma = float(cfg.get("sigma_meas", 0.05))
    master_seed = int(cfg.get("master_seed", 20260806))
    bandwidth_grid = [float(value) for value in cfg.get("bandwidth_grid", [0.1, 0.2, 0.4, 0.8])]
    threshold = float(cfg.get("near_zero_threshold", 0.2))

    # Backward-compatible, single-design geometry output for Fig. 2.
    geometry_n = _trajectory_count(cfg, mode)
    geometry: dict[str, Any] = {}
    for design in cfg.get(
        "phase_designs", ["aligned", "discrete", "continuous_uniform", "gap_near_zero"]
    ):
        record = _simulate_one(
            design=design,
            n=geometry_n,
            Delta=Delta,
            J=J,
            sigma=sigma,
            bandwidth_grid=bandwidth_grid,
            near_zero_threshold=threshold,
            master_seed=master_seed,
            config_id=f"geometry:{mode}:{design}",
            rep=0,
        )
        estimate = (
            {
                "point_estimate": record["point_estimate"],
                "h": record["bandwidth"],
                "q_min": record["q_min"],
                "bias_vs_J": (
                    record["point_estimate"] - J
                    if record["point_estimate"] is not None
                    else None
                ),
            }
            if record["estimation_applicable"]
            else {
                "point_estimate": None,
                "reason": "support gap; report identified set only",
                "q_min": record["q_min"],
            }
        )
        geometry[design] = {
            "q_min": record["q_min"],
            "n_near_zero": record["n_near_zero"],
            "n_lags": record["n_lags"],
            "first_lags": record["first_lags"],
            "estimate": estimate,
            "true_J": J,
        }

    sample_sizes = _sample_size_grid(cfg, mode)
    n_reps = mode_reps(cfg, mode)
    scan_designs = ["continuous_uniform", "gap_near_zero"]
    convergence: dict[str, list[dict[str, Any]]] = {name: [] for name in scan_designs}
    replication_records: list[dict[str, Any]] = []
    for design in scan_designs:
        for n in sample_sizes:
            records = [
                _simulate_one(
                    design=design,
                    n=n,
                    Delta=Delta,
                    J=J,
                    sigma=sigma,
                    bandwidth_grid=bandwidth_grid,
                    near_zero_threshold=threshold,
                    master_seed=master_seed,
                    config_id=f"scan:{design}:n={n}",
                    rep=rep,
                )
                for rep in range(n_reps)
            ]
            convergence[design].append(_summarize_records(records, J))
            for record in records:
                compact = dict(record)
                compact.pop("first_lags", None)
                replication_records.append(compact)

    summary = {
        "experiment": "exp03_async_phase",
        "mode": mode,
        "master_seed": master_seed,
        "Delta": Delta,
        "true_J": J,
        "sample_sizes": sample_sizes,
        "n_reps": n_reps,
        "config": cfg,
        "designs": geometry,
        "convergence": convergence,
        "replication_records": replication_records,
        "interpretation": (
            "Empirical sample-size scaling only; no optimal-rate or boundary-coverage claim. "
            "The positive-gap design has no point-identified jump estimator."
        ),
    }
    if outdir is not None:
        save_raw_and_summary(summary, Path(outdir) / f"exp03_{mode}.json")
    return summary


if __name__ == "__main__":
    print(run("smoke", Path("results/raw")))
