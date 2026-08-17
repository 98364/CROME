"""Anchored Hölder sharp identified sets for the jump (Theorem 2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float

    @property
    def width(self) -> float:
        return float(self.upper - self.lower)

    def contains(self, x: float, tol: float = 1e-12) -> bool:
        return (self.lower - tol) <= x <= (self.upper + tol)

    def is_empty(self, tol: float = 1e-12) -> bool:
        return self.upper < self.lower - tol


def interval_intersection(intervals: list[Interval]) -> Interval:
    if not intervals:
        return Interval(-np.inf, np.inf)
    lo = max(iv.lower for iv in intervals)
    hi = min(iv.upper for iv in intervals)
    return Interval(lo, hi)


def anchored_holder_constraint(r_q: float, q: float, L: float, alpha: float) -> Interval:
    """
    |r(q) - J| <= L q^alpha  ⇒  J ∈ [r(q) - L q^α, r(q) + L q^α].
    """
    if q <= 0:
        raise ValueError("q must be positive")
    half = L * (q**alpha)
    return Interval(r_q - half, r_q + half)


def multi_lag_intersection(
    r_at_lags: np.ndarray,
    lags: np.ndarray,
    L: float,
    alpha: float,
    eta: float | None = None,
) -> Interval:
    """
    I_J(L, α) = ∩_{q ∈ Q_η} [r(q) - L q^α, r(q) + L q^α]
    """
    r_at_lags = np.asarray(r_at_lags, dtype=float)
    lags = np.asarray(lags, dtype=float)
    if r_at_lags.shape != lags.shape:
        raise ValueError("r_at_lags and lags shape mismatch")
    mask = lags > 0
    if eta is not None:
        mask &= lags <= eta
    intervals = [
        anchored_holder_constraint(float(r), float(q), L, alpha)
        for r, q in zip(r_at_lags[mask], lags[mask], strict=True)
    ]
    return interval_intersection(intervals)


def anchored_holder_set(
    r_at_lags: np.ndarray,
    lags: np.ndarray,
    L: float,
    alpha: float,
    eta: float | None = None,
) -> Interval:
    return multi_lag_intersection(r_at_lags, lags, L, alpha, eta)


def single_lag_set(S_delta: float, Delta: float, L: float, alpha: float) -> Interval:
    """Corollary 3."""
    return anchored_holder_constraint(S_delta, Delta, L, alpha)


def attainability_construction(
    J: float,
    lags: np.ndarray,
    r_at_lags: np.ndarray,
    L: float,
    alpha: float,
    q_eval: np.ndarray,
) -> np.ndarray:
    """
    Build a continuous r_J in the anchored class matching finite observations
    (piecewise-linear z(q) construction from the plan).
    """
    lags = np.asarray(lags, dtype=float)
    r_at_lags = np.asarray(r_at_lags, dtype=float)
    q_eval = np.asarray(q_eval, dtype=float)
    if L == 0:
        return np.full_like(q_eval, J, dtype=float)

    order = np.argsort(lags)
    qj = lags[order]
    rj = r_at_lags[order]
    zj = (rj - J) / (L * np.power(qj, alpha))
    zj = np.clip(zj, -1.0, 1.0)

    # piecewise linear z on [q1, qM], constant extension outside
    z_eval = np.interp(q_eval, qj, zj, left=zj[0], right=zj[-1])
    z_eval = np.clip(z_eval, -1.0, 1.0)
    return J + L * np.power(np.maximum(q_eval, 0.0), alpha) * z_eval
