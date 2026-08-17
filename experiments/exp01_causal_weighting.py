"""Experiment 1: direct intervention vs oracle likelihood-ratio weighting."""

from __future__ import annotations

from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from crome_identification.interventions.likelihood_ratio import (
    build_rho_path_for_policy,
    trajectory_lr,
)
from crome_identification.metrics import ess
from crome_identification.processes.marked_events import simulate_trajectory
from crome_identification.processes.simulator import observation_grid, sample_endpoint
from crome_identification.seeding import make_seed_bundle

from ._common import kernel_from_cfg, load_exp_config, mode_reps, save_json


def _monte_carlo_mean_summary(
    values: list[float] | np.ndarray,
    *,
    level: float = 0.95,
) -> dict[str, Any]:
    """Summarize a Monte Carlo mean and its simulation uncertainty."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between zero and one")

    mean = float(np.mean(values))
    mc_sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    mc_se = float(mc_sd / np.sqrt(values.size))
    critical_value = NormalDist().inv_cdf(0.5 + level / 2.0)
    half_width = critical_value * mc_se
    return {
        "mean": mean,
        "mc_sd": mc_sd,
        "mc_se": mc_se,
        "mc_ci": {
            "method": "normal",
            "level": level,
            "lower": mean - half_width,
            "upper": mean + half_width,
        },
    }


def _y_at_horizon(traj, t0: float, q: float, Delta: float, sigma_meas: float, rng) -> float:
    t = t0 + q
    # interpolate latent path (oracle continuous) + optional meas noise at single point
    y = float(np.interp(t, traj.times_grid, traj.Y_latent))
    if sigma_meas > 0:
        y += float(rng.normal(0.0, sigma_meas))
    return y


def run(mode: str = "smoke", outdir: Path | None = None) -> dict[str, Any]:
    cfg = load_exp_config("causal_weighting")
    cfg["mode"] = mode
    n_reps = mode_reps(cfg, mode)
    mc_confidence_level = float(cfg.get("mc_confidence_level", 0.95))
    # smoke uses smaller n for speed
    n = 30 if mode == "smoke" else int(cfg["n"])
    kernel = kernel_from_cfg(cfg)
    q_grid = np.linspace(0.0, float(cfg["Q"]), 6)
    t0 = float(cfg["t0"])
    window = tuple(cfg["rho_window"])

    records = []
    for rep in range(n_reps):
        bundle = make_seed_bundle(int(cfg["master_seed"]), "exp01", mode, rep)
        rng = bundle.rng

        # observational trajectories
        L_list, Yw, Y0, Yg = [], [], [], []
        for i in range(n):
            child = np.random.default_rng(rng.integers(0, 2**63 - 1))
            traj0 = simulate_trajectory(
                T=float(cfg["T"]),
                delta=float(cfg["delta"]),
                rng=child,
                kernel=kernel,
                lambda0=np.asarray(cfg["lambda0"]),
                alpha_x=np.asarray(cfg["alpha_x"]),
                alpha_v=np.asarray(cfg["alpha_v"]),
                intervene=False,
            )
            rho_path = build_rho_path_for_policy(
                traj0.times_grid,
                int(cfg["C"]),
                int(cfg["rho_type"]),
                float(cfg["rho"]),
                window,
            )
            # recompute observational intensity (rho=1) for LR
            from crome_identification.processes.intensity import type_intensity

            n_steps = traj0.times_grid.size - 1
            intensity_obs = np.zeros((n_steps, int(cfg["C"])))
            for r in range(n_steps):
                intensity_obs[r] = type_intensity(
                    float(traj0.X[r]),
                    float(traj0.V[r]),
                    np.asarray(cfg["lambda0"]),
                    np.asarray(cfg["alpha_x"]),
                    np.asarray(cfg["alpha_v"]),
                )
            L = trajectory_lr(
                traj0,
                t0,
                t0 + float(cfg["Q"]),
                observational_intensity=intensity_obs,
                rho_path=rho_path,
            )
            y = _y_at_horizon(traj0, t0, float(cfg["Q"]), float(cfg["Delta"]), 0.0, child)
            L_list.append(L)
            Yw.append(L * y)
            Y0.append(y)

            # independent direct-g draw
            child_g = np.random.default_rng(rng.integers(0, 2**63 - 1))
            traj_g = simulate_trajectory(
                T=float(cfg["T"]),
                delta=float(cfg["delta"]),
                rng=child_g,
                kernel=kernel,
                lambda0=np.asarray(cfg["lambda0"]),
                alpha_x=np.asarray(cfg["alpha_x"]),
                alpha_v=np.asarray(cfg["alpha_v"]),
                rho_type=int(cfg["rho_type"]),
                rho=float(cfg["rho"]),
                rho_window=window,
                intervene=True,
            )
            Yg.append(_y_at_horizon(traj_g, t0, float(cfg["Q"]), float(cfg["Delta"]), 0.0, child_g))

        L_arr = np.asarray(L_list)
        ht = float(np.mean(Yw))
        hajek = float(np.sum(Yw) / np.sum(L_arr)) if L_arr.sum() > 0 else np.nan
        direct = float(np.mean(Yg))
        unweighted = float(np.mean(Y0))
        records.append(
            {
                "rep": rep,
                "mean_L": float(L_arr.mean()),
                "ess_over_n": ess(L_arr) / n,
                "max_L": float(L_arr.max()),
                "ht": ht,
                "hajek": hajek,
                "direct": direct,
                "unweighted": unweighted,
                "ht_minus_direct": ht - direct,
                "hajek_minus_direct": hajek - direct,
                "unweighted_minus_direct": unweighted - direct,
            }
        )

    monte_carlo = {
        metric: _monte_carlo_mean_summary(
            [r[record_key] for r in records],
            level=mc_confidence_level,
        )
        for metric, record_key in {
            "mean_L": "mean_L",
            "ess_over_n": "ess_over_n",
            "ht_minus_direct": "ht_minus_direct",
            "hajek_minus_direct": "hajek_minus_direct",
            "unweighted_minus_direct": "unweighted_minus_direct",
        }.items()
    }
    summary = {
        "experiment": "exp01_causal_weighting",
        "mode": mode,
        "n": n,
        "n_reps": n_reps,
        "mean_L": monte_carlo["mean_L"]["mean"],
        "mean_ess_over_n": monte_carlo["ess_over_n"]["mean"],
        "mean_ht_minus_direct": monte_carlo["ht_minus_direct"]["mean"],
        "mean_hajek_minus_direct": monte_carlo["hajek_minus_direct"]["mean"],
        "mean_unweighted_minus_direct": monte_carlo["unweighted_minus_direct"]["mean"],
        "mc_confidence_level": mc_confidence_level,
        "monte_carlo": monte_carlo,
        "records_head": records[: min(5, len(records))],
    }
    if outdir is not None:
        save_json(summary, Path(outdir) / f"exp01_{mode}.json")
    return summary


if __name__ == "__main__":
    print(run("smoke", Path("results/raw")))
