"""Type-specific intensities (history-dependent assignment)."""

from __future__ import annotations

import numpy as np


def type_intensity(
    x: float,
    v: float,
    lambda0: np.ndarray,
    alpha_x: np.ndarray,
    alpha_v: np.ndarray,
    *,
    delta_y_past: float = 0.0,
    n_recent: float = 0.0,
    alpha_y: np.ndarray | None = None,
    alpha_n: np.ndarray | None = None,
    cap: float = 50.0,
) -> np.ndarray:
    """
    lambda_c = lambda0_c * exp{ alpha_xc X + alpha_vc V + alpha_yc dY + alpha_nc N_recent }.

    Main DGP uses alpha_y = alpha_n = 0 (non-propagating assignment).
    """
    lam0 = np.asarray(lambda0, dtype=float)
    ax = np.asarray(alpha_x, dtype=float)
    av = np.asarray(alpha_v, dtype=float)
    ay = np.zeros_like(lam0) if alpha_y is None else np.asarray(alpha_y, dtype=float)
    an = np.zeros_like(lam0) if alpha_n is None else np.asarray(alpha_n, dtype=float)
    lin = ax * x + av * v + ay * delta_y_past + an * n_recent
    lin = np.clip(lin, -cap, cap)
    return lam0 * np.exp(lin)


def apply_tilt(
    lam: np.ndarray,
    rho: np.ndarray | float,
) -> np.ndarray:
    """Stochastic intensity intervention: lambda^g = rho * lambda."""
    return np.asarray(lam, dtype=float) * np.asarray(rho, dtype=float)
