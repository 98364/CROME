"""Baseline / no-response paths."""

from __future__ import annotations

import numpy as np

from .covariates import simulate_ou


def simulate_u(
    n_steps: int,
    delta: float,
    kappa_u: float,
    sigma_u: float,
    rng: np.random.Generator,
) -> np.ndarray:
    return simulate_ou(n_steps, delta, kappa_u, 0.0, sigma_u, rng)


def baseline_path(
    x: np.ndarray,
    v: np.ndarray,
    u: np.ndarray,
    beta_x: float,
    beta_v: float,
) -> np.ndarray:
    """B(t) = beta_X X(t) + beta_V V(t) + U(t)."""
    return beta_x * x + beta_v * v + u
