"""Ornstein–Uhlenbeck state processes on a fine grid."""

from __future__ import annotations

import numpy as np


def stationary_ou_init(
    mu: float,
    sigma: float,
    kappa: float,
    rng: np.random.Generator,
    size: int = 1,
) -> np.ndarray:
    """Draw from N(mu, sigma^2 / (2 kappa))."""
    if kappa <= 0:
        raise ValueError("kappa must be positive for stationary initialization")
    var = (sigma**2) / (2.0 * kappa)
    return rng.normal(mu, np.sqrt(var), size=size)


def simulate_ou(
    n_steps: int,
    delta: float,
    kappa: float,
    mu: float,
    sigma: float,
    rng: np.random.Generator,
    x0: float | None = None,
) -> np.ndarray:
    """
    Discrete OU on grid r = 0..n_steps:
      X_{r+1} = X_r + kappa (mu - X_r) delta + sigma sqrt(delta) eps
    Returns array of length n_steps + 1.
    """
    x = np.empty(n_steps + 1, dtype=float)
    if x0 is None:
        x[0] = float(stationary_ou_init(mu, sigma, kappa, rng, size=1)[0])
    else:
        x[0] = float(x0)
    sqrt_d = np.sqrt(delta)
    for r in range(n_steps):
        eps = rng.normal()
        x[r + 1] = x[r] + kappa * (mu - x[r]) * delta + sigma * sqrt_d * eps
    return x
