"""Experiment 7: baseline misspecification and timestamp error."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from crome_identification.inverse.design_matrix import build_design_matrix
from crome_identification.inverse.estimators import estimate_theta
from crome_identification.observation.timestamp_error import apply_timestamp_error
from crome_identification.processes.marked_events import simulate_trajectory
from crome_identification.processes.simulator import observation_grid, sample_endpoint
from crome_identification.seeding import make_seed_bundle

from ._common import kernel_from_cfg, load_exp_config, mode_reps, save_raw_and_summary


def _run_one(cfg: dict[str, Any], mode: str, rep: int) -> dict[str, Any]:
    n = {"smoke": 15, "dev": 40, "main": 80}[mode]
    kernel = kernel_from_cfg(cfg)
    bundle = make_seed_bundle(int(cfg["master_seed"]), "exp07", mode, rep)
    rng = bundle.rng
    Delta = float(cfg["Delta"])
    T = float(cfg["T"])

    # simulate once, then apply different residualizations / timestamp modes
    trajs = []
    for i in range(n):
        child = np.random.default_rng(rng.integers(0, 2**63 - 1))
        trajs.append(
            simulate_trajectory(
                T=T,
                delta=float(cfg["delta"]),
                rng=child,
                kernel=kernel,
                lambda0=np.asarray(cfg["lambda0"]),
                alpha_x=np.asarray(cfg["alpha_x"]),
                alpha_v=np.asarray(cfg["alpha_v"]),
            )
        )

    obs_t = observation_grid(T, Delta)
    observed_outcomes = []
    for i, traj in enumerate(trajs):
        measurement_rng = make_seed_bundle(
            int(cfg["master_seed"]), "exp07_measurement", f"{mode}:rep={rep}", i
        ).rng
        _, Y = sample_endpoint(traj, obs_t, float(cfg["sigma_meas"]), measurement_rng)
        observed_outcomes.append(Y)

    L_basis = 2
    coeffs = np.column_stack([kernel.J, kernel.a1, kernel.a2, kernel.a3])
    theta_true = coeffs[:, : L_basis + 1].reshape(-1)
    true_compound_jump = float(np.sum(kernel.J))

    def fit_for(baseline_mode: str, ts_mode: str, d_tau: float) -> dict[str, float]:
        obs_list, ev_t, ev_m = [], [], []
        Y_store, B_store = [], []
        for i, traj in enumerate(trajs):
            Y = observed_outcomes[i]
            if baseline_mode == "oracle":
                B = np.interp(obs_t, traj.times_grid, traj.B)
            elif baseline_mode == "correct_mean":
                B = cfg["beta_X"] * np.interp(obs_t, traj.times_grid, traj.X) + cfg[
                    "beta_V"
                ] * np.interp(obs_t, traj.times_grid, traj.V)
            elif baseline_mode == "omit_V":
                B = cfg["beta_X"] * np.interp(obs_t, traj.times_grid, traj.X)
            else:  # wrong_trend
                B = 0.1 * obs_t
            timestamp_rng = make_seed_bundle(
                int(cfg["master_seed"]),
                "exp07_timestamp",
                f"{mode}:rep={rep}:{ts_mode}:{d_tau}",
                i,
            ).rng
            taus = apply_timestamp_error(
                traj.event_times,
                ts_mode if ts_mode != "jitter" else "jitter",
                Delta=Delta,
                d_tau=d_tau,
                T=T,
                rng=timestamp_rng,
            )
            if ts_mode == "rounded":
                taus = apply_timestamp_error(traj.event_times, "rounded", Delta=Delta)
            order = np.argsort(taus, kind="stable")
            obs_list.append(obs_t)
            ev_t.append(taus[order])
            ev_m.append(traj.event_marks[order])
            Y_store.append(Y)
            B_store.append(B)

        A, _, index_map, column_scales = build_design_matrix(
            obs_list,
            ev_t,
            ev_m,
            C=int(cfg["C"]),
            L_basis=L_basis,
            Qresp=float(cfg.get("Qresp", 5.0)),
            return_column_scales=True,
        )
        if A.shape[0] == 0:
            return {"pred_mse": np.nan, "n_row": 0.0}
        Z = np.asarray([Y_store[i][j] - B_store[i][j] for i, j in index_map])
        theta_normalized = estimate_theta(A, Z)
        theta = column_scales * theta_normalized
        compound_jump = float(np.sum(theta[:: L_basis + 1]))
        return {
            "pred_mse": float(np.mean((A @ theta_normalized - Z) ** 2)),
            "n_row": float(A.shape[0]),
            "theta_norm": float(np.linalg.norm(theta)),
            "theta_error": float(np.linalg.norm(theta - theta_true)),
            "compound_jump_error": compound_jump - true_compound_jump,
        }

    grid = {}
    for bmode in cfg.get("baseline_modes", ["oracle", "omit_V"]):
        for tmode in ["exact", "rounded"]:
            grid[f"{bmode}|{tmode}"] = fit_for(bmode, tmode, 0.0)
        for frac in cfg.get("d_tau_frac", [0.05, 0.1]):
            d_tau = float(frac) * Delta
            grid[f"{bmode}|jitter_{frac}"] = fit_for(bmode, "jitter", d_tau)

    summary = {
        "experiment": "exp07_perturbations",
        "mode": mode,
        "rep": rep,
        "n_traj": n,
        "grid": grid,
    }
    return summary


def _mc_summary(values: list[float]) -> dict[str, float | None]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": None, "mc_se": None, "q05": None, "median": None, "q95": None}
    return {
        "mean": float(np.mean(x)),
        "mc_se": float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0,
        "q05": float(np.quantile(x, 0.05)),
        "median": float(np.median(x)),
        "q95": float(np.quantile(x, 0.95)),
    }


def run(mode: str = "smoke", outdir: Path | None = None) -> dict[str, Any]:
    cfg = load_exp_config("perturbation")
    cfg["mode"] = mode
    n_reps = mode_reps(cfg, mode)
    records = [_run_one(cfg, mode, rep) for rep in range(n_reps)]
    grid: dict[str, Any] = {}
    for key in records[0]["grid"]:
        metric_names = records[0]["grid"][key].keys()
        metric_summaries = {
            metric: _mc_summary(
                [float(record["grid"][key][metric]) for record in records]
            )
            for metric in metric_names
        }
        grid[key] = {
            metric: summary["mean"] for metric, summary in metric_summaries.items()
        }
        grid[key]["monte_carlo"] = metric_summaries

    summary = {
        "experiment": "exp07_perturbations",
        "mode": mode,
        "master_seed": int(cfg["master_seed"]),
        "n_traj": records[0]["n_traj"],
        "n_reps": n_reps,
        "config": cfg,
        "grid": grid,
        "replication_records": records,
        "interpretation": (
            "Repeated-design sensitivity only; no joint baseline-response or timestamp-set theorem."
        ),
    }
    if outdir is not None:
        save_raw_and_summary(summary, Path(outdir) / f"exp07_{mode}.json")
    return summary


if __name__ == "__main__":
    print(run("smoke", Path("results/raw")))
