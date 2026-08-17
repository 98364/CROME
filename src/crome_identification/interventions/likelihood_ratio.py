"""Girsanov-style likelihood ratio for absolute continuous intensity tilts."""

from __future__ import annotations

import numpy as np

from ..processes.marked_events import Trajectory


def likelihood_ratio_path(
    event_times: np.ndarray,
    event_marks: np.ndarray,
    times_grid: np.ndarray,
    intensity_obs: np.ndarray,
    rho_path: np.ndarray,
    t0: float,
    t1: float,
) -> float:
    """
    L^g_{t0,t1} = exp{
      sum_c ∫ log(rho_c) dN_c - sum_c ∫ (rho_c - 1) lambda_c ds
    }
    on [t0, t1], using the same piecewise-constant intensity path as the simulator.

    intensity_obs: (n_cells, C) factual intensity BEFORE tilt (or with rho already
    folded — we recover lambda and rho separately via rho_path).
    Here intensity_obs is the intensity used to generate events under the
    observational regime (rho≡1), and rho_path is the intervention tilt.
    """
    delta = float(times_grid[1] - times_grid[0])
    n_steps = times_grid.size - 1
    C = intensity_obs.shape[1]

    # compensator integral: sum_c ∫ (rho-1) lambda ds
    comp = 0.0
    for r in range(n_steps):
        tl = times_grid[r]
        tr = times_grid[r + 1]
        # intersection length with [t0, t1]
        left = max(tl, t0)
        right = min(tr, t1)
        if right <= left:
            continue
        dt = right - left
        for c in range(C):
            rho = float(rho_path[r, c])
            lam = float(intensity_obs[r, c])
            comp += (rho - 1.0) * lam * dt

    # jump part: sum log rho at events in (t0, t1]
    jump = 0.0
    for tau, m in zip(event_times, event_marks, strict=True):
        if t0 < tau <= t1:
            # rho at cell containing tau
            r = min(int(np.floor(tau / delta)), n_steps - 1)
            r = max(r, 0)
            rho = float(rho_path[r, int(m)])
            if rho <= 0:
                return 0.0
            jump += np.log(rho)

    return float(np.exp(jump - comp))


def trajectory_lr(
    traj: Trajectory,
    t0: float,
    t1: float,
    *,
    observational_intensity: np.ndarray | None = None,
    rho_path: np.ndarray | None = None,
) -> float:
    """
    For observational trajectories, intensity_path is lambda (rho=1).
    Pass the intervention rho_path separately.
    """
    intensity = observational_intensity if observational_intensity is not None else traj.intensity_path
    rho = rho_path if rho_path is not None else traj.rho_path
    return likelihood_ratio_path(
        traj.event_times,
        traj.event_marks,
        traj.times_grid,
        intensity,
        rho,
        t0,
        t1,
    )


def build_rho_path_for_policy(
    times_grid: np.ndarray,
    C: int,
    rho_type: int,
    rho: float,
    window: tuple[float, float],
) -> np.ndarray:
    n_steps = times_grid.size - 1
    rho_path = np.ones((n_steps, C), dtype=float)
    t0, t1 = window
    idx = rho_type - 1
    for r in range(n_steps):
        tl = times_grid[r]
        if t0 <= tl < t1 and 0 <= idx < C:
            rho_path[r, idx] = rho
    return rho_path
