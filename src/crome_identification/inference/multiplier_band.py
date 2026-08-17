"""Trajectory-cluster multiplier bootstrap simultaneous band on a fixed lag grid."""

from __future__ import annotations

import numpy as np


def multiplier_simultaneous_band(
    Y: np.ndarray,
    *,
    level: float = 0.95,
    n_boot: int = 999,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray | float]:
    """
    Y: array (n_traj, n_grid) of estimates per trajectory at fixed lags.
    Studentized sup-t multiplier bootstrap for the mean curve.
    """
    if rng is None:
        rng = np.random.default_rng()
    Y = np.asarray(Y, dtype=float)
    n, m = Y.shape
    mu = Y.mean(axis=0)
    se = Y.std(axis=0, ddof=1) / np.sqrt(n)
    se = np.maximum(se, 1e-12)
    # centered residuals
    R = Y - mu
    stats = np.empty(n_boot)
    for b in range(n_boot):
        xi = rng.normal(size=n)
        boot_mean = (xi[:, None] * R).mean(axis=0)
        tvals = np.abs(boot_mean) / se
        stats[b] = tvals.max()
    crit = float(np.quantile(stats, level))
    half = crit * se
    return {
        "mean": mu,
        "ell": mu - half,
        "u": mu + half,
        "critical_value": crit,
        "se": se,
    }
