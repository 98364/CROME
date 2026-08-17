"""Confidence regions that expand identified sets with simultaneous bands."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .holder_sets import Interval, multi_lag_intersection


@dataclass(frozen=True)
class ConfidenceRegion:
    interval: Interval
    level: float
    L: float
    alpha: float


def confidence_region_for_jump(
    ell: np.ndarray,
    u: np.ndarray,
    lags: np.ndarray,
    L: float,
    alpha: float,
    *,
    level: float = 0.95,
    eta: float | None = None,
) -> ConfidenceRegion:
    """
    C_J^{1-γ} = ∩_q [ell(q) - L q^α, u(q) + L q^α]
    when [ell, u] is a simultaneous band for r(q) on the lag grid.
    """
    ell = np.asarray(ell, dtype=float)
    u = np.asarray(u, dtype=float)
    lags = np.asarray(lags, dtype=float)
    mask = lags > 0
    if eta is not None:
        mask &= lags <= eta
    intervals = []
    for lo, hi, q in zip(ell[mask], u[mask], lags[mask], strict=True):
        half = L * (float(q) ** alpha)
        intervals.append(Interval(float(lo) - half, float(hi) + half))
    # reuse intersection logic
    if not intervals:
        iv = Interval(-np.inf, np.inf)
    else:
        iv = Interval(
            max(x.lower for x in intervals),
            min(x.upper for x in intervals),
        )
    return ConfidenceRegion(interval=iv, level=level, L=L, alpha=alpha)


def oracle_set_from_true_r(
    r_true: np.ndarray,
    lags: np.ndarray,
    L: float,
    alpha: float,
    eta: float | None = None,
) -> Interval:
    return multi_lag_intersection(r_true, lags, L, alpha, eta)
