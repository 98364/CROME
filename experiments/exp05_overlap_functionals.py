"""Experiment 5: mark-aware overlap design spectrum and recovery."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from crome_identification.inverse.design_matrix import build_design_matrix
from crome_identification.inverse.diagnostics import gram_spectrum, matrix_diagnostics
from crome_identification.inverse.estimators import estimate_theta
from crome_identification.inverse.functionals import identifiable_functional
from crome_identification.processes.marked_events import simulate_trajectory
from crome_identification.processes.simulator import observation_grid, sample_endpoint
from crome_identification.seeding import make_seed_bundle

from ._common import kernel_from_cfg, load_exp_config, mode_reps, save_raw_and_summary


def _run_one(cfg: dict[str, Any], mode: str, rep: int) -> dict[str, Any]:
    n = {"smoke": 20, "dev": 50, "main": 100}[mode]
    kernel = kernel_from_cfg(cfg)
    bundle = make_seed_bundle(int(cfg["master_seed"]), "exp05", mode, rep)
    rng = bundle.rng

    obs_times_list, event_times_list, event_marks_list = [], [], []
    Z_parts = []
    B_parts = []
    Y_parts = []

    for i in range(n):
        child = np.random.default_rng(rng.integers(0, 2**63 - 1))
        traj = simulate_trajectory(
            T=float(cfg["T"]),
            delta=float(cfg["delta"]),
            rng=child,
            kernel=kernel,
            lambda0=np.asarray(cfg["lambda0"]),
            alpha_x=np.asarray(cfg["alpha_x"]),
            alpha_v=np.asarray(cfg["alpha_v"]),
            intervene=False,
        )
        obs_t = observation_grid(float(cfg["T"]), float(cfg["Delta"]))
        _, Y = sample_endpoint(traj, obs_t, float(cfg["sigma_meas"]), child)
        B = np.interp(obs_t, traj.times_grid, traj.B)
        obs_times_list.append(obs_t)
        event_times_list.append(traj.event_times)
        event_marks_list.append(traj.event_marks)
        Y_parts.append(Y)
        B_parts.append(B)

    A, _, index_map, column_scales = build_design_matrix(
        obs_times_list,
        event_times_list,
        event_marks_list,
        C=int(cfg["C"]),
        L_basis=int(cfg.get("L_basis", 2)),
        Qresp=float(cfg.get("Qresp", 5.0)),
        normalize=True,
        return_column_scales=True,
    )
    Z = np.asarray([Y_parts[i][j] - B_parts[i][j] for i, j in index_map], dtype=float)
    diag = matrix_diagnostics(A, tol=float(cfg.get("svd_tol", 1e-10)))
    theta_normalized = (
        estimate_theta(A, Z, tol=float(cfg.get("svd_tol", 1e-10)))
        if A.shape[0]
        else np.array([])
    )
    theta_hat = column_scales * theta_normalized if theta_normalized.size else theta_normalized

    # compound functional: sum of jump columns (one-hot jump part)
    L_basis = int(cfg.get("L_basis", 2))
    C = int(cfg["C"])
    c_vec = np.zeros(C * (L_basis + 1))
    for a in range(C):
        c_vec[a * (L_basis + 1) + 0] = 1.0  # jump coeffs sum
    c_vec_normalized = c_vec * column_scales
    vartheta, ok = (
        identifiable_functional(A, Z, c_vec_normalized, tol=float(cfg.get("svd_tol", 1e-10)))
        if A.shape[0]
        else (np.array([np.nan]), False)
    )
    coeffs = np.column_stack([kernel.J, kernel.a1, kernel.a2, kernel.a3])
    theta_true = coeffs[:, : L_basis + 1].reshape(-1)
    true_compound_jump = float(np.sum(kernel.J))
    basis_names = ["jump", "basis_1", "basis_2", "basis_3"][: L_basis + 1]
    parameter_labels = [
        f"mark_{mark + 1}:{basis}"
        for mark in range(C)
        for basis in basis_names
    ]

    summary = {
        "experiment": "exp05_overlap_functionals",
        "mode": mode,
        "rep": rep,
        "n_traj": n,
        "diagnostics": diag,
        "singular_values": np.linalg.svd(A, compute_uv=False),
        "gram_eigenvalues": gram_spectrum(A),
        "parameter_labels": parameter_labels,
        "theta_true": theta_true,
        "theta_hat": theta_hat,
        "theta_hat_norm": float(np.linalg.norm(theta_hat)) if theta_hat.size else None,
        "full_parameter_error": (
            float(np.linalg.norm(theta_hat - theta_true)) if theta_hat.size else None
        ),
        "compound_jump_sum": float(np.asarray(vartheta).ravel()[0]),
        "compound_jump_error": float(np.asarray(vartheta).ravel()[0] - true_compound_jump),
        "compound_identifiable": bool(ok),
        "pred_mse": (
            float(np.mean((A @ theta_normalized - Z) ** 2))
            if theta_normalized.size
            else None
        ),
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
    cfg = load_exp_config("overlap")
    cfg["mode"] = mode
    n_reps = mode_reps(cfg, mode)
    records = [_run_one(cfg, mode, rep) for rep in range(n_reps)]
    first = records[0]

    theta_hats = np.asarray([record["theta_hat"] for record in records], dtype=float)
    singular_values = np.asarray(
        [record["singular_values"] for record in records], dtype=float
    )
    gram_eigenvalues = np.asarray(
        [record["gram_eigenvalues"] for record in records], dtype=float
    )
    diagnostic_keys = first["diagnostics"].keys()
    diagnostics = {
        key: float(np.mean([record["diagnostics"][key] for record in records]))
        for key in diagnostic_keys
    }
    summary = {
        "experiment": "exp05_overlap_functionals",
        "mode": mode,
        "master_seed": int(cfg["master_seed"]),
        "n_traj": first["n_traj"],
        "n_reps": n_reps,
        "config": cfg,
        "diagnostics": diagnostics,
        "singular_values": np.mean(singular_values, axis=0),
        "gram_eigenvalues": np.mean(gram_eigenvalues, axis=0),
        "parameter_labels": first["parameter_labels"],
        "theta_true": first["theta_true"],
        "theta_hat": np.mean(theta_hats, axis=0),
        "theta_hat_norm": float(np.mean([record["theta_hat_norm"] for record in records])),
        "full_parameter_error": float(
            np.mean([record["full_parameter_error"] for record in records])
        ),
        "compound_jump_sum": float(
            np.mean([record["compound_jump_sum"] for record in records])
        ),
        "compound_jump_error": float(
            np.mean([record["compound_jump_error"] for record in records])
        ),
        "compound_identifiable": bool(
            all(record["compound_identifiable"] for record in records)
        ),
        "pred_mse": float(np.mean([record["pred_mse"] for record in records])),
        "monte_carlo": {
            "full_parameter_error": _mc_summary(
                [record["full_parameter_error"] for record in records]
            ),
            "compound_jump_error": _mc_summary(
                [record["compound_jump_error"] for record in records]
            ),
            "absolute_compound_jump_error": _mc_summary(
                [abs(record["compound_jump_error"]) for record in records]
            ),
            "prediction_mse": _mc_summary([record["pred_mse"] for record in records]),
            "normalized_gram_condition": _mc_summary(
                [record["diagnostics"]["cond"] for record in records]
            ),
        },
        "replication_records": records,
    }
    if outdir is not None:
        save_raw_and_summary(summary, Path(outdir) / f"exp05_{mode}.json")
    return summary


if __name__ == "__main__":
    print(run("smoke", Path("results/raw")))
