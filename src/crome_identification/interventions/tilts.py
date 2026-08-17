"""Predictable intensity tilt policies."""

from __future__ import annotations

import numpy as np


def constant_type_tilt(
    t: float,
    C: int,
    type_index_1based: int,
    rho: float,
    window: tuple[float, float],
) -> np.ndarray:
    """Type-c intensity multiplied by rho on [t0, t1)."""
    out = np.ones(C, dtype=float)
    t0, t1 = window
    if t0 <= t < t1:
        idx = type_index_1based - 1
        if 0 <= idx < C:
            out[idx] = rho
    return out


def local_kernel_tilt(
    t: float,
    C: int,
    type_index_1based: int,
    epsilon: float,
    s: float,
    w: float,
    kernel: str = "uniform",
) -> np.ndarray:
    """
    rho = 1 + eps K_w(t-s) 1{m=c} on support [s, s+w].
    """
    out = np.ones(C, dtype=float)
    if not (s <= t < s + w):
        return out
    u = (t - s) / w
    if kernel == "uniform":
        K = 1.0 / w
    elif kernel == "epanechnikov":
        # on [0,1] mapped support
        K = (0.75 * (1.0 - (2.0 * u - 1.0) ** 2) / w) if 0 <= u <= 1 else 0.0
    else:
        raise ValueError(kernel)
    idx = type_index_1based - 1
    out[idx] = 1.0 + epsilon * K
    if out[idx] <= 0:
        raise ValueError("tilt not strictly positive; reduce epsilon")
    return out
