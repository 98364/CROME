"""Shared numerical diagnostics."""

from __future__ import annotations

import numpy as np


def ess(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    w = w[np.isfinite(w) & (w >= 0)]
    if w.size == 0:
        return 0.0
    s = w.sum()
    if s <= 0:
        return 0.0
    w = w / s
    return float(1.0 / np.sum(w**2))


def rmse(err: np.ndarray) -> float:
    e = np.asarray(err, dtype=float)
    return float(np.sqrt(np.mean(e**2)))


def bias_variance_rmse(estimates: np.ndarray, truth: float) -> dict[str, float]:
    x = np.asarray(estimates, dtype=float)
    m = float(np.mean(x))
    var = float(np.var(x, ddof=1)) if x.size > 1 else 0.0
    b = m - float(truth)
    return {
        "mean": m,
        "bias": b,
        "variance": var,
        "rmse": float(np.sqrt(b**2 + var)),
        "se": float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0,
    }


def monte_carlo_se_binomial(p_hat: float, n: int) -> float:
    p = float(np.clip(p_hat, 0.0, 1.0))
    if n <= 0:
        return np.nan
    return float(np.sqrt(p * (1.0 - p) / n))


def lag_diagnostics(lags: np.ndarray, eta: float) -> dict[str, float]:
    q = np.asarray(lags, dtype=float)
    q = q[q > 0]
    if q.size == 0:
        return {
            "q_min": np.inf,
            "n_near_zero": 0.0,
            "n_lags": 0.0,
            "support_gap": np.inf,
        }
    near = q[q <= eta]
    return {
        "q_min": float(q.min()),
        "n_near_zero": float(near.size),
        "n_lags": float(q.size),
        "support_gap": float(q.min()),
    }
