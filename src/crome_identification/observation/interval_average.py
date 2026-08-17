"""Interval-average observation operator (appendix path; independent of endpoint)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def interval_average_response(
    response: Callable[[np.ndarray], np.ndarray],
    tau: float,
    a: float,
    b: float,
    n_quad: int = 64,
) -> float:
    """
    (1/(b-a)) ∫_a^b r(t-tau) 1{t>tau} dt
    """
    if b <= a:
        raise ValueError("require b > a")
    t = np.linspace(a, b, n_quad)
    q = t - tau
    vals = np.zeros_like(t)
    mask = q > 0
    if np.any(mask):
        vals[mask] = response(q[mask])
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(vals, t) / (b - a))
