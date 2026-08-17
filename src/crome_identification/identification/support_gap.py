"""Lag-support gap / accumulation identification boundary (Theorem 1 family)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..responses.equivalence import (
    observational_equivalent_pair,
    response_with_jump,
    response_without_jump_matching,
    sample_on_lags,
)


@dataclass(frozen=True)
class SupportGapResult:
    identifiable_jump: bool
    q_star: float | None
    message: str


def support_gap_nonidentification(
    lags: np.ndarray,
    *,
    allow_free_near_zero: bool = True,
) -> SupportGapResult:
    """
    If exists q* > 0 with Q ∩ (0, q*) = ∅ and near-zero response unrestricted,
    jump is not point-identified.
    """
    lags = np.asarray(lags, dtype=float)
    pos = lags[lags > 0]
    if pos.size == 0:
        return SupportGapResult(False, None, "no positive lags")
    q_min = float(pos.min())
    if q_min > 0 and allow_free_near_zero:
        return SupportGapResult(
            False,
            q_min,
            f"support gap on (0, {q_min}]; jump not point-identified",
        )
    return SupportGapResult(True, 0.0, "zero is accumulation / no gap under assumptions")


def two_point_lower_bound(J_a: float, J_b: float) -> float:
    """Minimax two-point lower bound |Ja - Jb| / 2."""
    return abs(float(J_a) - float(J_b)) / 2.0


def assert_observational_equivalence(
    q_star: float,
    observed_lags: np.ndarray,
    J_a: float = 1.0,
) -> dict[str, float | bool]:
    """Gate 1 deterministic check."""
    lags = np.asarray(observed_lags, dtype=float)
    assert np.all(lags > 0), "lags must be strictly positive"
    r_a = lambda q: response_with_jump(q, J_a)
    r_b = lambda q: response_without_jump_matching(q, J_a, q_star)
    ya = sample_on_lags(r_a, lags)
    yb = sample_on_lags(r_b, lags)
    pair = observational_equivalent_pair(q_star, J_a=J_a)
    return {
        "allclose_on_lags": bool(np.allclose(ya, yb)),
        "jump_a": pair.J_a,
        "jump_b": pair.J_b,
        "jumps_differ": pair.J_a != pair.J_b,
        "lower_bound": two_point_lower_bound(pair.J_a, pair.J_b),
    }
