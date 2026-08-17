"""Experiment 5b: incremental information from finite categorical marks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from crome_identification.inverse.design_matrix import build_design_matrix
from crome_identification.inverse.functionals import (
    identification_residual,
    is_identifiable,
    target_noise_amplification,
)
from crome_identification.seeding import make_seed_bundle

from ._common import load_exp_config, mode_reps, save_raw_and_summary


def default_config() -> dict[str, Any]:
    return {
        "Delta": 1.0,
        "T": 8.0,
        "Qresp": 8.0,
        "L_basis": 2,
        "sigma_meas": 0.05,
        "master_seed": 20260806,
        "svd_tol": 1e-10,
        "theta_true": [1.0, 0.6, 0.2, 0.5, 0.3, 0.1],
        "scenarios_smoke": ["strong_excitation", "simultaneous"],
        "scenarios_dev": [
            "weak_excitation",
            "strong_excitation",
            "near_collinearity",
            "simultaneous",
        ],
        "scenarios_main": [
            "weak_excitation",
            "strong_excitation",
            "near_collinearity",
            "simultaneous",
        ],
        "n_traj_smoke": 20,
        "n_traj_dev": 50,
        "n_traj_main": 100,
        "n_reps_smoke": 20,
        "n_reps_dev": 100,
        "n_reps_main": 300,
    }


def _config(mode: str) -> dict[str, Any]:
    cfg = default_config()
    cfg.update(load_exp_config("mark_information"))
    cfg["mode"] = mode
    cfg["n_traj"] = int(cfg[f"n_traj_{mode}"])
    cfg["scenarios"] = list(cfg[f"scenarios_{mode}"])
    return cfg


def _trajectory_events(
    scenario: str,
    trajectory: int,
    Delta: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    first = float(rng.uniform(0.05 * Delta, 0.30 * Delta))
    if scenario == "simultaneous":
        return np.asarray([first, first]), np.asarray([0, 1], dtype=int)

    if scenario == "strong_excitation":
        pattern = trajectory % 4
        if pattern == 0:
            return np.asarray([first]), np.asarray([0], dtype=int)
        if pattern == 1:
            return np.asarray([first]), np.asarray([1], dtype=int)
        second = float(rng.uniform(0.45 * Delta, 0.85 * Delta))
        marks = np.asarray([0, 1] if pattern == 2 else [1, 0], dtype=int)
        return np.asarray([first, second]), marks

    if scenario == "weak_excitation":
        if trajectory % 20 == 0:
            return np.asarray([first]), np.asarray([0], dtype=int)
        if trajectory % 20 == 1:
            return np.asarray([first]), np.asarray([1], dtype=int)
        second = first + float(rng.uniform(0.02 * Delta, 0.08 * Delta))
        return np.asarray([first, second]), np.asarray([0, 1], dtype=int)

    if scenario == "near_collinearity":
        if trajectory == 0:
            return np.asarray([first]), np.asarray([0], dtype=int)
        if trajectory == 1:
            return np.asarray([first]), np.asarray([1], dtype=int)
        second = first + 0.005 * Delta
        return np.asarray([first, second]), np.asarray([0, 1], dtype=int)

    raise ValueError(scenario)


def _normalize(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=float)
    scales = np.ones(A.shape[1], dtype=float)
    target = np.sqrt(max(A.shape[0], 1))
    normalized = A.copy()
    for column in range(A.shape[1]):
        norm = float(np.linalg.norm(A[:, column]))
        if norm > 0.0:
            scales[column] = target / norm
            normalized[:, column] *= scales[column]
    return normalized, scales


def _block_diagonal(blocks: list[np.ndarray]) -> np.ndarray:
    n_row = sum(block.shape[0] for block in blocks)
    n_col = sum(block.shape[1] for block in blocks)
    result = np.zeros((n_row, n_col), dtype=float)
    row_offset = 0
    col_offset = 0
    for block in blocks:
        result[
            row_offset : row_offset + block.shape[0],
            col_offset : col_offset + block.shape[1],
        ] = block
        row_offset += block.shape[0]
        col_offset += block.shape[1]
    return result


def _build_variants(cfg: dict[str, Any], scenario: str) -> tuple[dict[str, dict], np.ndarray]:
    Delta = float(cfg["Delta"])
    T = float(cfg["T"])
    n_traj = int(cfg["n_traj"])
    rng = make_seed_bundle(
        int(cfg["master_seed"]), "exp05b_design", scenario, 0
    ).rng
    obs_times: list[np.ndarray] = []
    event_times: list[np.ndarray] = []
    observed_marks: list[np.ndarray] = []
    for trajectory in range(n_traj):
        events, marks = _trajectory_events(scenario, trajectory, Delta, rng)
        obs_times.append(np.arange(1.0, T / Delta + 1.0) * Delta)
        event_times.append(events)
        observed_marks.append(marks)

    common_kwargs = {
        "L_basis": int(cfg["L_basis"]),
        "Qresp": float(cfg["Qresp"]),
        "normalize": False,
    }
    observed_raw, _, _ = build_design_matrix(
        obs_times, event_times, observed_marks, C=2, **common_kwargs
    )

    flat_marks = np.concatenate(observed_marks).copy()
    rng.shuffle(flat_marks)
    shuffled_marks = []
    mark_offset = 0
    for marks in observed_marks:
        next_offset = mark_offset + marks.size
        shuffled_marks.append(flat_marks[mark_offset:next_offset])
        mark_offset = next_offset
    shuffled_raw, _, _ = build_design_matrix(
        obs_times, event_times, shuffled_marks, C=2, **common_kwargs
    )

    collapsed_marks = [np.zeros(marks.size, dtype=int) for marks in observed_marks]
    collapsed_raw, _, _ = build_design_matrix(
        obs_times, event_times, collapsed_marks, C=1, **common_kwargs
    )

    free_blocks = []
    for obs_i, events_i, marks_i in zip(
        obs_times, event_times, observed_marks, strict=True
    ):
        block, _, _ = build_design_matrix(
            [obs_i], [events_i], [marks_i], C=2, **common_kwargs
        )
        free_blocks.append(block)
    free_raw = _block_diagonal(free_blocks)

    theta_true = np.asarray(cfg["theta_true"], dtype=float)
    mean_signal = observed_raw @ theta_true
    variants = {
        "collapsed_mark": {
            "A_raw": collapsed_raw,
            "targets": {
                "mark_target": np.asarray([1.0, 0.0, 0.0]),
                "compound_target": np.asarray([2.0, 0.0, 0.0]),
            },
        },
        "observed_shared_mark": {
            "A_raw": observed_raw,
            "targets": {
                "mark_target": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                "compound_target": np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            },
        },
        "shuffled_mark": {
            "A_raw": shuffled_raw,
            "targets": {
                "mark_target": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                "compound_target": np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            },
        },
        "free_trajectory": {
            "A_raw": free_raw,
            "targets": {
                "mark_target": np.tile(
                    np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) / n_traj,
                    n_traj,
                ),
                "compound_target": np.tile(
                    np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]) / n_traj,
                    n_traj,
                ),
            },
        },
    }
    return variants, mean_signal


def _partial_information(A: np.ndarray, tol: float) -> dict[str, Any]:
    block = A.shape[1] // 2
    common = A[:, :block] + A[:, block:]
    deviation = A[:, block:]
    residualized = deviation - common @ np.linalg.pinv(common, rcond=tol) @ deviation
    n_row = max(A.shape[0], 1)
    spectrum = np.linalg.eigvalsh((residualized.T @ residualized) / n_row)
    spectrum = np.sort(np.maximum(spectrum, 0.0))[::-1]
    reference_spectrum = np.linalg.eigvalsh((A.T @ A) / n_row)
    reference_scale = (
        float(np.max(reference_spectrum)) if reference_spectrum.size else 0.0
    )
    threshold = tol * reference_scale if reference_scale > 0.0 else tol
    return {
        "incremental_rank": int(np.sum(spectrum > threshold)),
        "conditional_gram_eigenvalues": spectrum,
    }


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
    cfg = _config(mode)
    tol = float(cfg["svd_tol"])
    sigma = float(cfg["sigma_meas"])
    theta_true = np.asarray(cfg["theta_true"], dtype=float)
    truth = {"mark_target": float(theta_true[0]), "compound_target": float(theta_true[0] + theta_true[3])}
    n_reps = mode_reps(cfg, mode)
    scenario_summaries: dict[str, Any] = {}
    replication_records: list[dict[str, Any]] = []

    for scenario in cfg["scenarios"]:
        variants, mean_signal = _build_variants(cfg, scenario)
        prepared: dict[str, Any] = {}
        for name, variant in variants.items():
            A, scales = _normalize(variant["A_raw"])
            target_info = {}
            for target_name, c_physical in variant["targets"].items():
                c_normalized = c_physical * scales
                target_info[target_name] = {
                    "physical": c_physical,
                    "identifiable": is_identifiable(A, c_normalized, tol=tol),
                    "identification_residual": identification_residual(
                        A, c_normalized, tol=tol
                    ),
                    "amplification": target_noise_amplification(A, c_normalized, tol=tol),
                    "truth": truth[target_name],
                }
            prepared[name] = {"A": A, "scales": scales, "targets": target_info}

        records_by_variant: dict[str, list[dict[str, Any]]] = {name: [] for name in prepared}
        for rep in range(n_reps):
            rng = make_seed_bundle(
                int(cfg["master_seed"]), "exp05b_noise", scenario, rep
            ).rng
            outcome = mean_signal + rng.normal(0.0, sigma, size=mean_signal.size)
            for name, variant in prepared.items():
                A = variant["A"]
                theta_normalized_hat = np.linalg.lstsq(A, outcome, rcond=tol)[0]
                theta_hat = variant["scales"] * theta_normalized_hat
                target_records = {}
                for target_name, item in variant["targets"].items():
                    estimate = float(item["physical"] @ theta_hat)
                    error = estimate - item["truth"]
                    target_records[target_name] = {
                        "estimate": estimate if item["identifiable"] else None,
                        "error": error if item["identifiable"] else None,
                        "pseudoinverse_error_diagnostic": error,
                    }
                record = {
                    "scenario": scenario,
                    "variant": name,
                    "rep": rep,
                    "prediction_mse": float(np.mean(np.square(A @ theta_normalized_hat - outcome))),
                    "targets": target_records,
                }
                records_by_variant[name].append(record)
                replication_records.append(record)

        variant_summaries = {}
        for name, variant in prepared.items():
            records = records_by_variant[name]
            target_summaries = {}
            for target_name, item in variant["targets"].items():
                errors = [
                    record["targets"][target_name]["error"]
                    for record in records
                    if record["targets"][target_name]["error"] is not None
                ]
                pseudo_errors = [
                    record["targets"][target_name]["pseudoinverse_error_diagnostic"]
                    for record in records
                ]
                target_summaries[target_name] = {
                    "truth": item["truth"],
                    "identifiable": item["identifiable"],
                    "identification_residual": item["identification_residual"],
                    "amplification": item["amplification"],
                    "rmse": (
                        float(np.sqrt(np.mean(np.square(errors)))) if errors else None
                    ),
                    "error": _mc_summary(errors),
                    "absolute_error": _mc_summary([abs(value) for value in errors]),
                    "pseudoinverse_rmse_diagnostic": float(
                        np.sqrt(np.mean(np.square(pseudo_errors)))
                    ),
                }
            variant_summaries[name] = {
                "n_parameter": int(variant["A"].shape[1]),
                "rank": int(np.linalg.matrix_rank(variant["A"], tol=tol)),
                "targets": target_summaries,
                "prediction_mse": _mc_summary(
                    [record["prediction_mse"] for record in records]
                ),
            }

        observed_info = _partial_information(
            variants["observed_shared_mark"]["A_raw"], tol
        )
        shuffled_info = _partial_information(variants["shuffled_mark"]["A_raw"], tol)
        scenario_summaries[scenario] = {
            "n_traj": int(cfg["n_traj"]),
            "n_reps": n_reps,
            "variants": variant_summaries,
            "mark_information": {
                "incremental_rank": observed_info["incremental_rank"],
                "conditional_gram_eigenvalues": observed_info[
                    "conditional_gram_eigenvalues"
                ],
                "shuffled_incremental_rank": shuffled_info["incremental_rank"],
                "shuffled_conditional_gram_eigenvalues": shuffled_info[
                    "conditional_gram_eigenvalues"
                ],
            },
        }

    summary = {
        "experiment": "exp05b_mark_information",
        "mode": mode,
        "master_seed": int(cfg["master_seed"]),
        "n_reps": n_reps,
        "config": cfg,
        "scenarios": scenario_summaries,
        "replication_records": replication_records,
        "interpretation": (
            "Identification residuals are model-specific. RMSE against the common scientific truth "
            "also reveals misspecification in collapsed and shuffled controls."
        ),
    }
    if outdir is not None:
        save_raw_and_summary(summary, Path(outdir) / f"exp05b_{mode}.json")
    return summary


if __name__ == "__main__":
    print(run("smoke", Path("results/raw")))
