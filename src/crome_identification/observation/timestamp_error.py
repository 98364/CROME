"""Timestamp error / coarsening operators."""

from __future__ import annotations

import numpy as np

from ..processes.simulator import coarsen_event_times, jitter_event_times


def apply_timestamp_error(
    event_times: np.ndarray,
    mode: str,
    *,
    Delta: float | None = None,
    d_tau: float = 0.0,
    T: float = 20.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    mode = mode.lower()
    if mode in {"exact", "async", "none"}:
        return np.asarray(event_times, dtype=float).copy()
    if mode in {"aligned", "coarsened", "rounded"}:
        if Delta is None:
            raise ValueError("Delta required for coarsened timestamps")
        return coarsen_event_times(event_times, Delta)
    if mode in {"jitter", "error"}:
        if rng is None:
            raise ValueError("rng required for jitter")
        return jitter_event_times(event_times, d_tau, rng, T)
    raise ValueError(f"unknown timestamp mode: {mode}")
