"""Experiment 4: sharp identified set + simultaneous confidence region."""

from __future__ import annotations

from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from crome_identification.identification.confidence_regions import confidence_region_for_jump
from crome_identification.identification.holder_sets import (
    multi_lag_intersection,
    single_lag_set,
)
from crome_identification.inference.multiplier_band import multiplier_simultaneous_band
from crome_identification.responses.equivalence import response_with_jump
from crome_identification.seeding import make_seed_bundle

from ._common import load_exp_config, mode_reps, save_json


def _least_favorable_boundary_response(
    lags: np.ndarray,
    J_true: float,
    L: float,
    alpha: float,
) -> np.ndarray:
    """Activate alternating lower/upper anchored Hölder constraints."""
    signs = np.where(np.arange(lags.size) % 2 == 0, 1.0, -1.0)
    return J_true + signs * L * np.power(lags, alpha)


def _interval_summary(interval) -> dict[str, float]:
    return {
        "lower": interval.lower,
        "upper": interval.upper,
        "width": interval.width,
    }


def _monte_carlo_se(probability: float, n_reps: int) -> float:
    return float(np.sqrt(probability * (1.0 - probability) / n_reps))


def _wilson_interval(successes: int, n_reps: int, *, level: float = 0.95) -> dict[str, Any]:
    """Wilson score interval, including non-degenerate all-success/all-failure cases."""
    if n_reps <= 0:
        raise ValueError("n_reps must be positive")
    if not 0 <= successes <= n_reps:
        raise ValueError("successes must lie between zero and n_reps")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between zero and one")

    probability = successes / n_reps
    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    z2_over_n = z**2 / n_reps
    denominator = 1.0 + z2_over_n
    center = (probability + z2_over_n / 2.0) / denominator
    half_width = (
        z
        * np.sqrt(
            probability * (1.0 - probability) / n_reps
            + z**2 / (4.0 * n_reps**2)
        )
        / denominator
    )
    return {
        "method": "wilson",
        "level": level,
        "lower": float(max(0.0, center - half_width)),
        "upper": float(min(1.0, center + half_width)),
    }


def _simulate_bands(
    *,
    name: str,
    r_true: np.ndarray,
    lags: np.ndarray,
    level: float,
    n_traj: int,
    n_reps: int,
    n_boot: int,
    master_seed: int,
    mode: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Simulate one set of bands that can be reused across sensitivity assumptions."""
    bands = []
    for rep in range(n_reps):
        bundle = make_seed_bundle(master_seed, "exp04", f"{mode}:{name}", rep)
        Y = r_true + bundle.rng.normal(0.0, 0.15, size=(n_traj, lags.size))
        band = multiplier_simultaneous_band(
            Y,
            level=level,
            n_boot=n_boot,
            rng=bundle.bootstrap_rng(0),
        )
        bands.append((np.asarray(band["ell"]), np.asarray(band["u"])))
    return bands


def _summarize_scenario(
    *,
    r_true: np.ndarray,
    bands: list[tuple[np.ndarray, np.ndarray]],
    lags: np.ndarray,
    J_true: float,
    L: float,
    alpha: float,
    eta: float,
    level: float,
    mc_confidence_level: float,
) -> dict[str, Any]:
    """Apply an identifying restriction to saved bands and summarize coverage."""
    if not bands:
        raise ValueError("bands must contain at least one Monte Carlo repetition")

    oracle_set = multi_lag_intersection(r_true, lags, L, alpha, eta=eta)
    point_hits = []
    set_hits = []
    widths = []

    for ell, u in bands:
        cr = confidence_region_for_jump(
            ell, u, lags, L, alpha, level=level, eta=eta
        ).interval
        nonempty = not cr.is_empty()
        point_hits.append(nonempty and cr.contains(J_true))
        set_hits.append(
            nonempty
            and cr.lower <= oracle_set.lower + 1e-12
            and cr.upper >= oracle_set.upper - 1e-12
        )
        widths.append(max(0.0, cr.width))

    n_reps = len(bands)
    point_successes = int(np.sum(point_hits))
    oracle_set_successes = int(np.sum(set_hits))
    point_coverage = point_successes / n_reps
    oracle_set_coverage = oracle_set_successes / n_reps
    return {
        "oracle_set": _interval_summary(oracle_set),
        "oracle_contains_J": oracle_set.contains(J_true),
        "point_coverage": float(point_coverage),
        "point_coverage_successes": point_successes,
        "point_coverage_mc_se": _monte_carlo_se(point_coverage, n_reps),
        "point_coverage_ci": _wilson_interval(
            point_successes,
            n_reps,
            level=mc_confidence_level,
        ),
        "oracle_set_coverage": float(oracle_set_coverage),
        "oracle_set_coverage_successes": oracle_set_successes,
        "oracle_set_coverage_mc_se": _monte_carlo_se(oracle_set_coverage, n_reps),
        "oracle_set_coverage_ci": _wilson_interval(
            oracle_set_successes,
            n_reps,
            level=mc_confidence_level,
        ),
        "mean_cr_width": float(np.mean(widths)),
    }


def run(mode: str = "smoke", outdir: Path | None = None) -> dict[str, Any]:
    cfg = load_exp_config("partial_id")
    L = float(cfg.get("L", 1.0))
    alpha = float(cfg.get("alpha", 1.0))
    eta = float(cfg.get("eta", 1.0))
    level = float(cfg.get("level", 0.95))
    mc_confidence_level = float(cfg.get("mc_confidence_level", 0.95))
    J_true = 1.0
    lags = np.asarray(cfg.get("fixed_lags", [0.1, 0.25, 0.5, 0.75, 1.0]), dtype=float)
    interior_response = response_with_jump(lags, J_true)
    boundary_response = _least_favorable_boundary_response(lags, J_true, L, alpha)

    oracle_set = multi_lag_intersection(interior_response, lags, L, alpha, eta=eta)
    single = single_lag_set(float(interior_response[-1]), float(lags[-1]), L, alpha)

    n_traj = mode_reps(cfg, mode, key_prefix="n_traj")
    n_reps = mode_reps(cfg, mode)
    n_boot = int(cfg.get("n_bootstrap", 999))

    simulation_args = {
        "lags": lags,
        "level": level,
        "n_traj": n_traj,
        "n_reps": n_reps,
        "n_boot": n_boot,
        "master_seed": int(cfg["master_seed"]),
        "mode": mode,
    }
    interior_bands = _simulate_bands(
        name="interior",
        r_true=interior_response,
        **simulation_args,
    )
    boundary_bands = _simulate_bands(
        name="least_favorable_boundary",
        r_true=boundary_response,
        **simulation_args,
    )
    summary_args = {
        "lags": lags,
        "J_true": J_true,
        "eta": eta,
        "level": level,
        "mc_confidence_level": mc_confidence_level,
    }
    interior = _summarize_scenario(
        r_true=interior_response,
        bands=interior_bands,
        L=L,
        alpha=alpha,
        **summary_args,
    )
    boundary = _summarize_scenario(
        r_true=boundary_response,
        bands=boundary_bands,
        L=L,
        alpha=alpha,
        **summary_args,
    )

    L_grid = [float(value) for value in cfg.get("L_grid", [L])]
    alpha_grid = [float(value) for value in cfg.get("alpha_grid", [alpha])]
    sensitivity_grid = [
        {
            "L": grid_L,
            "alpha": grid_alpha,
            **_summarize_scenario(
                r_true=interior_response,
                bands=interior_bands,
                L=grid_L,
                alpha=grid_alpha,
                **summary_args,
            ),
        }
        for grid_L in L_grid
        for grid_alpha in alpha_grid
    ]

    summary = {
        "experiment": "exp04_partial_id",
        "mode": mode,
        "L": L,
        "alpha": alpha,
        "eta": eta,
        "level": level,
        "mc_confidence_level": mc_confidence_level,
        "true_J": J_true,
        "oracle_set": _interval_summary(oracle_set),
        "oracle_contains_J": oracle_set.contains(J_true),
        "single_lag_width": single.width,
        "multi_lag_width": oracle_set.width,
        "multi_not_wider_than_single": oracle_set.width <= single.width + 1e-12,
        "coverage": interior["point_coverage"],
        "coverage_successes": interior["point_coverage_successes"],
        "coverage_ci": interior["point_coverage_ci"],
        "mean_cr_width": interior["mean_cr_width"],
        "point_coverage": interior["point_coverage"],
        "point_coverage_successes": interior["point_coverage_successes"],
        "point_coverage_mc_se": interior["point_coverage_mc_se"],
        "point_coverage_ci": interior["point_coverage_ci"],
        "oracle_set_coverage": interior["oracle_set_coverage"],
        "oracle_set_coverage_successes": interior["oracle_set_coverage_successes"],
        "oracle_set_coverage_mc_se": interior["oracle_set_coverage_mc_se"],
        "oracle_set_coverage_ci": interior["oracle_set_coverage_ci"],
        "scenarios": {
            "interior": interior,
            "least_favorable_boundary": boundary,
        },
        "sensitivity": {
            "dgp": "interior",
            "interpretation": (
                "The response DGP and simulated bands are fixed; only the identifying "
                "restriction (L, alpha) varies."
            ),
            "grid": sensitivity_grid,
        },
        "n_reps": n_reps,
        "n_traj": n_traj,
        "n_boot": n_boot,
    }
    if outdir is not None:
        save_json(summary, Path(outdir) / f"exp04_{mode}.json")
    return summary


if __name__ == "__main__":
    print(run("smoke", Path("results/raw")))
