"""Shared mark/state response kernels (main DGP family)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SharedKernelParams:
    """
    r(c, q) = J_c + a1_c (e^{-beta1 q} - 1) + a2_c (1 - e^{-beta2 q}) + a3_c q e^{-beta3 q}

    History interaction can be added by scaling coefficients with phi(h).
    Convention: r(c, 0) means post-event right limit J_c (cadlag).
    Analysis data only uses strict positive lags.
    """

    J: np.ndarray
    a1: np.ndarray
    a2: np.ndarray
    a3: np.ndarray
    beta: np.ndarray  # (beta1, beta2, beta3)

    @classmethod
    def from_lists(
        cls,
        J: list[float] | np.ndarray,
        a1: list[float] | np.ndarray,
        a2: list[float] | np.ndarray,
        a3: list[float] | np.ndarray,
        beta: list[float] | np.ndarray,
    ) -> SharedKernelParams:
        return cls(
            J=np.asarray(J, dtype=float),
            a1=np.asarray(a1, dtype=float),
            a2=np.asarray(a2, dtype=float),
            a3=np.asarray(a3, dtype=float),
            beta=np.asarray(beta, dtype=float),
        )


def response_kernel(params: SharedKernelParams, mark: int, q: float | np.ndarray) -> np.ndarray:
    """Evaluate r(mark, q) for scalar or array q >= 0."""
    q_arr = np.asarray(q, dtype=float)
    c = int(mark)
    b1, b2, b3 = params.beta
    cont = (
        params.a1[c] * (np.exp(-b1 * q_arr) - 1.0)
        + params.a2[c] * (1.0 - np.exp(-b2 * q_arr))
        + params.a3[c] * q_arr * np.exp(-b3 * q_arr)
    )
    return params.J[c] + cont


def response_matrix(params: SharedKernelParams, q: np.ndarray) -> np.ndarray:
    """Return shape (C, len(q))."""
    q = np.asarray(q, dtype=float)
    C = params.J.size
    out = np.empty((C, q.size), dtype=float)
    for c in range(C):
        out[c] = response_kernel(params, c, q)
    return out


def holder_boundary_response(J: float, L: float, alpha: float, sign: float, q: np.ndarray) -> np.ndarray:
    """D(q) = sign * L * q^alpha for boundary DGP (Theorem 2 slope checks)."""
    q = np.asarray(q, dtype=float)
    return J + sign * L * np.power(np.maximum(q, 0.0), alpha)
