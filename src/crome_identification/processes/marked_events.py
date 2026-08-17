"""Piecewise-constant competing Poisson marked-event simulator."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..responses.kernels import SharedKernelParams, response_kernel
from .baseline import baseline_path, simulate_u
from .covariates import simulate_ou
from .intensity import apply_tilt, type_intensity


@dataclass
class Trajectory:
    """One independent trajectory on [0, T]."""

    times_grid: np.ndarray
    X: np.ndarray
    V: np.ndarray
    U: np.ndarray
    B: np.ndarray
    Y_latent: np.ndarray
    event_times: np.ndarray
    event_marks: np.ndarray
    intensity_path: np.ndarray  # (n_cells, C) piecewise constant on cells
    rho_path: np.ndarray  # (n_cells, C) applied tilt
    kernel: SharedKernelParams | None = None
    process_noise: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


def _rho_for_cell(
    t_left: float,
    C: int,
    rho_type: int | None,
    rho: float,
    rho_window: tuple[float, float] | None,
) -> np.ndarray:
    r = np.ones(C, dtype=float)
    if rho_type is None or rho_window is None:
        return r
    t0, t1 = rho_window
    if t0 <= t_left < t1:
        # rho_type is 1-based in config (type 1); convert to 0-based
        idx = int(rho_type) - 1
        if 0 <= idx < C:
            r[idx] = float(rho)
    return r


def simulate_events_piecewise_poisson(
    times_grid: np.ndarray,
    X: np.ndarray,
    V: np.ndarray,
    lambda0: np.ndarray,
    alpha_x: np.ndarray,
    alpha_v: np.ndarray,
    rng: np.random.Generator,
    *,
    rho_type: int | None = None,
    rho: float = 1.0,
    rho_window: tuple[float, float] | None = None,
    alpha_y: np.ndarray | None = None,
    alpha_n: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    On each cell [rδ, (r+1)δ), intensity is constant from left state.
    K ~ Poisson(Λ δ); event times Uniform in cell; marks ~ λ_c / Λ.

    Returns event_times, event_marks, intensity_path (n_cells, C), rho_path.
    """
    delta = float(times_grid[1] - times_grid[0])
    n_steps = times_grid.size - 1
    C = int(np.asarray(lambda0).size)
    intensity_path = np.zeros((n_steps, C), dtype=float)
    rho_path = np.ones((n_steps, C), dtype=float)
    times: list[float] = []
    marks: list[int] = []

    for r in range(n_steps):
        t_left = float(times_grid[r])
        lam = type_intensity(
            float(X[r]),
            float(V[r]),
            lambda0,
            alpha_x,
            alpha_v,
            alpha_y=alpha_y,
            alpha_n=alpha_n,
        )
        rho_vec = _rho_for_cell(t_left, C, rho_type, rho, rho_window)
        lam_g = apply_tilt(lam, rho_vec)
        intensity_path[r] = lam_g
        rho_path[r] = rho_vec
        Lam = float(lam_g.sum())
        if Lam <= 0:
            continue
        K = rng.poisson(Lam * delta)
        if K <= 0:
            continue
        # continuous times inside cell
        u = rng.uniform(0.0, delta, size=K)
        et = t_left + u
        probs = lam_g / Lam
        mk = rng.choice(C, size=K, p=probs)
        # keep chronological order within cell
        order = np.argsort(et)
        for j in order:
            times.append(float(et[j]))
            marks.append(int(mk[j]))

    return (
        np.asarray(times, dtype=float),
        np.asarray(marks, dtype=int),
        intensity_path,
        rho_path,
    )


def build_latent_outcome(
    times_grid: np.ndarray,
    B: np.ndarray,
    event_times: np.ndarray,
    event_marks: np.ndarray,
    kernel: SharedKernelParams,
    process_noise: np.ndarray | None = None,
) -> np.ndarray:
    """
    Cadlag outcome on fine grid:
      Y(t) = B(t) + sum_{tau_k <= t} r(m_k, t - tau_k) + process_noise
    with r(m, 0) = J(m).
    """
    Y = B.copy()
    if process_noise is not None:
        Y = Y + process_noise
    if event_times.size == 0:
        return Y
    for t_idx, t in enumerate(times_grid):
        contrib = 0.0
        for tau, m in zip(event_times, event_marks, strict=True):
            if tau <= t + 1e-15:
                q = t - tau
                contrib += float(response_kernel(kernel, int(m), max(q, 0.0)))
        Y[t_idx] = Y[t_idx] + contrib
    return Y


def simulate_trajectory(
    T: float,
    delta: float,
    rng: np.random.Generator,
    kernel: SharedKernelParams,
    *,
    lambda0: np.ndarray,
    alpha_x: np.ndarray,
    alpha_v: np.ndarray,
    kappa_X: float = 0.5,
    mu_X: float = 0.0,
    sigma_X: float = 0.4,
    kappa_V: float = 0.8,
    mu_V: float = 0.0,
    sigma_V: float = 0.25,
    beta_X: float = 0.5,
    beta_V: float = 0.3,
    kappa_U: float = 1.0,
    sigma_U: float = 0.2,
    process_noise_sigma: float = 0.0,
    rho_type: int | None = None,
    rho: float = 1.0,
    rho_window: tuple[float, float] | None = None,
    intervene: bool = False,
) -> Trajectory:
    if T <= 0 or delta <= 0:
        raise ValueError("T and delta must be positive")
    n_steps = int(round(T / delta))
    if not np.isclose(n_steps * delta, T, rtol=1e-12, atol=1e-12 * max(1.0, abs(T))):
        raise ValueError("T must be an integer multiple of delta")
    times_grid = np.linspace(0.0, n_steps * delta, n_steps + 1)
    X = simulate_ou(n_steps, delta, kappa_X, mu_X, sigma_X, rng)
    V = simulate_ou(n_steps, delta, kappa_V, mu_V, sigma_V, rng)
    U = simulate_u(n_steps, delta, kappa_U, sigma_U, rng)
    B = baseline_path(X, V, U, beta_X, beta_V)

    use_rho_type = rho_type if intervene else None
    use_window = rho_window if intervene else None

    event_times, event_marks, intensity_path, rho_path = simulate_events_piecewise_poisson(
        times_grid,
        X,
        V,
        lambda0,
        alpha_x,
        alpha_v,
        rng,
        rho_type=use_rho_type,
        rho=rho,
        rho_window=use_window,
    )

    process_noise = None
    if process_noise_sigma > 0:
        process_noise = simulate_ou(n_steps, delta, 1.2, 0.0, process_noise_sigma, rng)

    Y = build_latent_outcome(times_grid, B, event_times, event_marks, kernel, process_noise)

    return Trajectory(
        times_grid=times_grid,
        X=X,
        V=V,
        U=U,
        B=B,
        Y_latent=Y,
        event_times=event_times,
        event_marks=event_marks,
        intensity_path=intensity_path,
        rho_path=rho_path,
        kernel=kernel,
        process_noise=process_noise,
        meta={"T": T, "delta": delta, "intervene": intervene},
    )


def simulate_trajectories(n: int, **kwargs) -> list[Trajectory]:
    rng: np.random.Generator = kwargs.pop("rng")
    out = []
    for i in range(n):
        # independent child streams
        child = np.random.default_rng(rng.integers(0, 2**63 - 1))
        out.append(simulate_trajectory(rng=child, **kwargs))
    return out
