"""Event–observation lag support utilities."""

from __future__ import annotations

import numpy as np


def observed_positive_lags(
    observation_times: np.ndarray,
    event_times: np.ndarray,
) -> np.ndarray:
    """
    All Q = t_obs - tau for t_obs > tau.
    Never includes q = 0.
    """
    obs = np.asarray(observation_times, dtype=float)
    ev = np.asarray(event_times, dtype=float)
    lags = []
    for tau in ev:
        for t in obs:
            q = t - tau
            if q > 0:
                lags.append(q)
    return np.asarray(lags, dtype=float)


def first_forward_recurrence(tau: float, Delta: float) -> float:
    """
    R = (floor(tau/Delta)+1)*Delta - tau ∈ (0, Delta].
    If tau falls exactly on a grid point, R = Delta (not 0).
    """
    if Delta <= 0:
        raise ValueError("Delta must be positive")
    # Move the floating quotient by one ULP toward +inf so decimal grid hits
    # such as 0.3 / 0.1 are treated as the intended integer boundary.
    quotient = np.nextafter(float(tau) / float(Delta), np.inf)
    k = np.floor(quotient) + 1.0
    R = k * Delta - tau
    if R <= 0:
        # numerical edge: treat as exact grid hit → next bin
        R = Delta
    return float(R)


def phase_mod(tau: float | np.ndarray, Delta: float) -> np.ndarray:
    return np.mod(np.asarray(tau, dtype=float), Delta)


def empirical_support_gap(lags: np.ndarray) -> float:
    lags = np.asarray(lags, dtype=float)
    pos = lags[lags > 0]
    if pos.size == 0:
        return np.inf
    return float(pos.min())


def zero_is_accumulation_point(lags: np.ndarray, tol: float = 1e-12) -> bool:
    """Finite-sample proxy: whether min positive lag is arbitrarily small (for tests)."""
    return empirical_support_gap(lags) <= tol
