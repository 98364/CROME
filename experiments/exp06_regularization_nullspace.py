"""Experiment 6: null space, compound functionals, regularization boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from crome_identification.inverse.estimators import estimate_theta, ridge_theta
from crome_identification.inverse.functionals import (
    identifiable_functional,
    is_identifiable,
    null_space_basis,
)
from crome_identification.seeding import make_seed_bundle

from ._common import load_exp_config, save_json


def run(mode: str = "smoke", outdir: Path | None = None) -> dict[str, Any]:
    cfg = load_exp_config("overlap")
    bundle = make_seed_bundle(int(cfg["master_seed"]), "exp06", mode, 0)
    rng = bundle.rng

    # Deterministic singular design: second column is exact copy of first
    n, p = 40, 4
    A0 = rng.normal(size=(n, p - 1))
    A = np.column_stack([A0[:, 0], A0[:, 0], A0[:, 1], A0[:, 2]])  # col0 == col1
    theta_true = np.array([1.0, 0.0, -0.5, 0.2])
    # truth only identified up to null: e1 - e2
    Z = A @ theta_true + rng.normal(0, 0.05, size=n)

    N = null_space_basis(A, tol=1e-10)
    # unidentifiable contrast: c = (1, -1, 0, 0)
    c_bad = np.array([1.0, -1.0, 0.0, 0.0])
    c_good = np.array([1.0, 1.0, 0.0, 0.0])  # col0+col1 direction in row space-ish

    ols = estimate_theta(A, Z)
    ridge = ridge_theta(A, Z, lam=1e-8)
    ridge2 = ridge_theta(A, Z, lam=10.0)
    # pure null-space direction allocation: different particular solutions
    theta_part = ols
    theta_shifted = ols + N[:, 0] if N.shape[1] else ols

    v_bad, ok_bad = identifiable_functional(A, Z, c_bad)
    v_good, ok_good = identifiable_functional(A, Z, c_good)

    contrast_spread = float(
        np.max(
            np.abs(
                [
                    c_bad @ ridge,
                    c_bad @ ridge2,
                    c_bad @ theta_part,
                    c_bad @ theta_shifted,
                ]
            )
        )
        - np.min(
            np.abs(
                [
                    c_bad @ ridge,
                    c_bad @ ridge2,
                    c_bad @ theta_part,
                    c_bad @ theta_shifted,
                ]
            )
        )
    )
    # Unidentifiable contrast changes when we move along Null(A) or change ridge.
    moves = abs(float(c_bad @ theta_part) - float(c_bad @ theta_shifted)) > 1e-8
    moves = moves or abs(float(c_bad @ ridge) - float(c_bad @ ridge2)) > 1e-6

    shift_grid = np.linspace(-2.0, 2.0, 41)
    shifted_solutions = [theta_part + shift * N[:, 0] for shift in shift_grid]
    null_shift_path = {
        "shift": shift_grid,
        "bad_contrast": [float(c_bad @ theta) for theta in shifted_solutions],
        "good_contrast": [float(c_good @ theta) for theta in shifted_solutions],
        "pred_mse": [float(np.mean((A @ theta - Z) ** 2)) for theta in shifted_solutions],
    }

    ridge_lambdas = np.logspace(-8.0, 2.0, 31)
    ridge_solutions = [ridge_theta(A, Z, lam=float(lam)) for lam in ridge_lambdas]
    ridge_path = {
        "lambda": ridge_lambdas,
        "bad_contrast": [float(c_bad @ theta) for theta in ridge_solutions],
        "good_contrast": [float(c_good @ theta) for theta in ridge_solutions],
        "pred_mse": [float(np.mean((A @ theta - Z) ** 2)) for theta in ridge_solutions],
    }

    summary = {
        "experiment": "exp06_regularization_nullspace",
        "mode": mode,
        "null_dim": int(N.shape[1]),
        "bad_contrast_identifiable": bool(is_identifiable(A, c_bad)),
        "good_contrast_identifiable": bool(is_identifiable(A, c_good)),
        "ols_contrast_bad": float(c_bad @ ols),
        "ridge_small_contrast_bad": float(c_bad @ ridge),
        "ridge_large_contrast_bad": float(c_bad @ ridge2),
        "null_shift_contrast_bad": float(c_bad @ theta_shifted),
        "regularization_moves_unidentifiable_contrast": bool(moves),
        "contrast_spread": contrast_spread,
        "null_shift_path": null_shift_path,
        "ridge_path": ridge_path,
        "good_functional_ols": float(np.asarray(v_good).ravel()[0]),
        "good_functional_stable_under_null": bool(
            abs(float(c_good @ theta_part) - float(c_good @ theta_shifted)) < 1e-6
        ),
        "good_functional_ok": bool(ok_good),
        "bad_functional_ok": bool(ok_bad),
        "pred_mse_ols": float(np.mean((A @ ols - Z) ** 2)),
        "note": "Prediction can be good while event allocation is arbitrary under Null(A).",
    }
    if outdir is not None:
        save_json(summary, Path(outdir) / f"exp06_{mode}.json")
    return summary


if __name__ == "__main__":
    print(run("smoke", Path("results/raw")))
