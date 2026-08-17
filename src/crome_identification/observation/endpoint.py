"""Endpoint observation operator on response functions."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def sample_on_lags(
    response: Callable[[np.ndarray], np.ndarray] | np.ndarray,
    lags: np.ndarray,
    q_grid: np.ndarray | None = None,
) -> np.ndarray:
    """
    Endpoint operator O_Q(r) = {r(q): q in Q}.
    Requires strictly positive lags.
    """
    lags = np.asarray(lags, dtype=float)
    if np.any(lags <= 0):
        raise ValueError("endpoint analysis forbids non-positive lags (no q=0)")
    if callable(response):
        return np.asarray(response(lags), dtype=float)
    if q_grid is None:
        raise ValueError("q_grid required when response is an array")
    return np.interp(lags, q_grid, np.asarray(response, dtype=float))
