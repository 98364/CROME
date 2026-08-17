"""Observation schedules and endpoint sampling."""

from __future__ import annotations

import numpy as np

from ..responses.kernels import response_kernel
from .marked_events import Trajectory


def observation_grid(T: float, Delta: float) -> np.ndarray:
    """Regular outcome grid j * Delta in [0, T]."""
    m = int(np.floor(T / Delta))
    return np.arange(0, m + 1, dtype=float) * Delta


def sample_endpoint(
    traj: Trajectory,
    obs_times: np.ndarray,
    sigma_meas: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Endpoint observation of latent path + measurement noise.
    Returns (obs_times, Y_obs).
    """
    obs_times = np.asarray(obs_times, dtype=float)
    kernel = traj.kernel or traj.meta.get("kernel")
    process_noise = traj.process_noise
    if process_noise is None:
        process_noise = traj.meta.get("process_noise")

    if kernel is None:
        Y = np.interp(obs_times, traj.times_grid, traj.Y_latent)
    else:
        # Interpolate only continuous components. Reconstruct event responses at
        # the requested times so interpolation never crosses a jump.
        Y = np.interp(obs_times, traj.times_grid, traj.B)
        if process_noise is not None:
            Y = Y + np.interp(obs_times, traj.times_grid, process_noise)
        for tau, mark in zip(traj.event_times, traj.event_marks, strict=True):
            mask = obs_times >= tau
            if np.any(mask):
                Y[mask] += response_kernel(kernel, int(mark), obs_times[mask] - tau)
    if sigma_meas > 0:
        Y = Y + rng.normal(0.0, sigma_meas, size=Y.shape)
    return obs_times, Y


def coarsen_event_times(event_times: np.ndarray, Delta: float) -> np.ndarray:
    """Align events to observation bins (Regime A)."""
    if Delta <= 0:
        raise ValueError("Delta must be positive")
    if event_times.size == 0:
        return event_times.copy()
    quotient = np.nextafter(np.asarray(event_times, dtype=float) / Delta, np.inf)
    return np.floor(quotient) * Delta


def jitter_event_times(
    event_times: np.ndarray,
    d_tau: float,
    rng: np.random.Generator,
    T: float,
) -> np.ndarray:
    """
    Timestamp error: tilde tau = tau + U, U ~ Unif[-d, d], rejection resample if out of [0, T].
    """
    if event_times.size == 0 or d_tau <= 0:
        return event_times.copy()
    out = np.empty_like(event_times)
    for i, tau in enumerate(event_times):
        for _ in range(10_000):
            u = rng.uniform(-d_tau, d_tau)
            t = tau + u
            if 0.0 <= t <= T:
                out[i] = t
                break
        else:
            out[i] = float(np.clip(tau, 0.0, T))
    # Preserve positional event identity. Callers that require chronological
    # order must apply the same permutation to timestamps and marks together.
    return out


def build_outcome_path(traj: Trajectory) -> np.ndarray:
    return traj.Y_latent
