"""Jump / post-event continuous component split."""

from __future__ import annotations

import numpy as np

from .kernels import SharedKernelParams, response_kernel


def jump_of(params: SharedKernelParams, mark: int) -> float:
    return float(params.J[int(mark)])


def split_jump_continuous(
    params: SharedKernelParams,
    mark: int,
    q: float | np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    r(q) = J + D(q) with lim_{q↓0} D(q) = 0.
    """
    q_arr = np.asarray(q, dtype=float)
    J = jump_of(params, mark)
    r = response_kernel(params, mark, q_arr)
    D = r - J
    return J, D
