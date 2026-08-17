"""Observational-equivalence constructions for support-gap non-identification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EquivalentPair:
    """Two responses equal on observed positive lags but different jumps."""

    J_a: float
    J_b: float
    q_star: float
    r_a: np.ndarray  # values on evaluation grid
    r_b: np.ndarray
    q_grid: np.ndarray


def response_with_jump(q: float | np.ndarray, J: float, decay: float = 2.0) -> np.ndarray:
    """Smooth response with jump J: lim_{q↓0} r(q) = J, continuous for q > 0."""
    q = np.asarray(q, dtype=float)
    # D(q) → 0 as q → 0+
    return J + 0.4 * J * (1.0 - np.exp(-decay * q))


def response_without_jump_matching(
    q: float | np.ndarray,
    J_a: float,
    q_star: float,
    decay: float = 2.0,
) -> np.ndarray:
    """
    Build r_B with J_B = 0 that matches r_A on [q_star, ∞) by fast near-zero transition.

    For q >= q_star: r_B(q) = r_A(q).
    For 0 < q < q_star: continuous path from 0 to r_A(q_star).
    """
    q = np.asarray(q, dtype=float)
    r_a = response_with_jump(q, J_a, decay=decay)
    r_star = float(response_with_jump(q_star, J_a, decay=decay))
    out = np.empty_like(q, dtype=float)
    for i, qi in enumerate(np.atleast_1d(q).astype(float)):
        if qi >= q_star:
            out.flat[i] = r_a.flat[i]
        elif qi <= 0:
            out.flat[i] = 0.0  # pre-event convention for plotting only
        else:
            # smoothstep from 0 to r_star on (0, q_star)
            t = qi / q_star
            s = t * t * (3.0 - 2.0 * t)
            out.flat[i] = s * r_star
    return out.reshape(q.shape)


def observational_equivalent_pair(
    q_star: float,
    J_a: float = 1.0,
    q_grid: np.ndarray | None = None,
    decay: float = 2.0,
) -> EquivalentPair:
    if q_grid is None:
        q_grid = np.linspace(0.0, 5.0 * q_star, 501)
    r_a = response_with_jump(q_grid, J_a, decay=decay)
    r_b = response_without_jump_matching(q_grid, J_a, q_star, decay=decay)
    return EquivalentPair(
        J_a=J_a,
        J_b=0.0,
        q_star=q_star,
        r_a=r_a,
        r_b=r_b,
        q_grid=q_grid,
    )


def sample_on_lags(response_fn, lags: np.ndarray) -> np.ndarray:
    lags = np.asarray(lags, dtype=float)
    if np.any(lags <= 0):
        raise ValueError("observed lags must be strictly positive (no q=0 post-event)")
    return np.asarray(response_fn(lags), dtype=float)
