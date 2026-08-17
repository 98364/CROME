"""Same-target baselines for CROME comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from ..inference.boundary_regression import one_sided_local_linear


@dataclass(frozen=True)
class BaselineTargetResult:
    method: str
    target_estimate: float | None
    success: bool
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    failure_reason: str = ""


@dataclass(frozen=True)
class InverseHyperparameters:
    ridge_lambda: float
    tsvd_rank: int
    ridge_validation_mse: float
    tsvd_validation_mse: float


def _validated_inverse_inputs(
    A: np.ndarray, y: np.ndarray, C: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design = np.asarray(A, dtype=float)
    outcome = np.asarray(y, dtype=float)
    target = np.asarray(C, dtype=float)
    if design.ndim != 2:
        raise ValueError("A must be two-dimensional")
    if outcome.ndim != 1 or outcome.shape[0] != design.shape[0]:
        raise ValueError("y must have one entry per design row")
    if target.ndim == 2 and target.shape[0] == 1:
        target = target.reshape(-1)
    if target.ndim != 1 or target.shape[0] != design.shape[1]:
        raise ValueError("C must be a scalar target row matching A")
    if not (
        np.all(np.isfinite(design))
        and np.all(np.isfinite(outcome))
        and np.all(np.isfinite(target))
    ):
        raise ValueError("A, y, and C must contain finite values")
    return design, outcome, target


def fit_naive_boundary(
    lags: np.ndarray,
    responses: np.ndarray,
    *,
    bandwidth: float,
) -> BaselineTargetResult:
    started = perf_counter()
    fit = one_sided_local_linear(lags, responses, float(bandwidth))
    estimate = float(fit["estimate"])
    elapsed = perf_counter() - started
    if not math.isfinite(estimate):
        return BaselineTargetResult(
            method="naive_boundary",
            target_estimate=None,
            success=False,
            hyperparameters={"bandwidth": float(bandwidth)},
            runtime_seconds=elapsed,
            failure_reason="no positive lag received nonzero boundary-kernel weight",
        )
    return BaselineTargetResult(
        method="naive_boundary",
        target_estimate=estimate,
        success=True,
        hyperparameters={"bandwidth": float(bandwidth)},
        runtime_seconds=elapsed,
    )


def _ridge_theta(design: np.ndarray, outcome: np.ndarray, lam: float) -> np.ndarray:
    if lam == 0.0:
        return np.linalg.lstsq(design, outcome, rcond=1e-12)[0]
    gram = design.T @ design
    return np.linalg.solve(gram + lam * np.eye(design.shape[1]), design.T @ outcome)


def fit_ridge_target(
    A: np.ndarray,
    y: np.ndarray,
    C: np.ndarray,
    *,
    lam: float,
) -> BaselineTargetResult:
    design, outcome, target = _validated_inverse_inputs(A, y, C)
    lam = float(lam)
    if not math.isfinite(lam) or lam < 0:
        raise ValueError("lam must be finite and non-negative")
    started = perf_counter()
    try:
        estimate = float(target @ _ridge_theta(design, outcome, lam))
    except np.linalg.LinAlgError as exc:
        return BaselineTargetResult(
            method="ridge",
            target_estimate=None,
            success=False,
            hyperparameters={"lambda": lam},
            runtime_seconds=perf_counter() - started,
            failure_reason=str(exc),
        )
    return BaselineTargetResult(
        method="ridge",
        target_estimate=estimate,
        success=math.isfinite(estimate),
        hyperparameters={"lambda": lam},
        runtime_seconds=perf_counter() - started,
        failure_reason="" if math.isfinite(estimate) else "non-finite target estimate",
    )


def _tsvd_theta(design: np.ndarray, outcome: np.ndarray, rank: int) -> np.ndarray:
    U, singular_values, Vt = np.linalg.svd(design, full_matrices=False)
    retained = singular_values[:rank]
    tolerance = 1e-12 * max(1.0, float(singular_values[0]) if singular_values.size else 1.0)
    inverse = np.zeros_like(retained)
    np.divide(1.0, retained, out=inverse, where=retained > tolerance)
    return Vt[:rank].T @ (inverse * (U[:, :rank].T @ outcome))


def fit_tsvd_target(
    A: np.ndarray,
    y: np.ndarray,
    C: np.ndarray,
    *,
    rank: int,
) -> BaselineTargetResult:
    design, outcome, target = _validated_inverse_inputs(A, y, C)
    maximum_rank = min(design.shape)
    if not isinstance(rank, (int, np.integer)) or not 1 <= int(rank) <= maximum_rank:
        raise ValueError(f"rank must lie between 1 and {maximum_rank}")
    rank = int(rank)
    started = perf_counter()
    estimate = float(target @ _tsvd_theta(design, outcome, rank))
    return BaselineTargetResult(
        method="tsvd",
        target_estimate=estimate if math.isfinite(estimate) else None,
        success=math.isfinite(estimate),
        hyperparameters={"rank": rank},
        runtime_seconds=perf_counter() - started,
        failure_reason="" if math.isfinite(estimate) else "non-finite target estimate",
    )


def select_inverse_hyperparameters(
    A: np.ndarray,
    y: np.ndarray,
    ridge_grid: list[float],
    rank_grid: list[int],
    *,
    n_splits: int = 5,
) -> InverseHyperparameters:
    """Select ridge lambda and tSVD rank using only the supplied rows."""

    design = np.asarray(A, dtype=float)
    outcome = np.asarray(y, dtype=float)
    if design.ndim != 2 or outcome.ndim != 1 or outcome.shape[0] != design.shape[0]:
        raise ValueError("A/y shape mismatch")
    if design.shape[0] < 2:
        raise ValueError("at least two supplied rows are required")
    if not ridge_grid or not rank_grid:
        raise ValueError("ridge and rank grids must be non-empty")
    ridge_candidates = sorted({float(value) for value in ridge_grid}, reverse=True)
    if any(not math.isfinite(value) or value < 0 for value in ridge_candidates):
        raise ValueError("ridge candidates must be finite and non-negative")
    max_rank = min(design.shape)
    rank_candidates = sorted({int(value) for value in rank_grid})
    if any(value < 1 or value > max_rank for value in rank_candidates):
        raise ValueError("rank candidates exceed the supplied design dimensions")

    n_splits = max(2, min(int(n_splits), design.shape[0]))
    fold_ids = np.arange(design.shape[0]) % n_splits

    def validation_mse(kind: str, value: float | int) -> float:
        errors: list[float] = []
        for fold in range(n_splits):
            train = fold_ids != fold
            validation = ~train
            if kind == "ridge":
                theta = _ridge_theta(design[train], outcome[train], float(value))
            else:
                feasible_rank = min(int(value), min(design[train].shape))
                theta = _tsvd_theta(design[train], outcome[train], feasible_rank)
            errors.extend(np.square(design[validation] @ theta - outcome[validation]).tolist())
        return float(np.mean(errors))

    ridge_scores = [(validation_mse("ridge", value), -value, value) for value in ridge_candidates]
    rank_scores = [(validation_mse("tsvd", value), value, value) for value in rank_candidates]
    best_ridge_mse, _, best_ridge = min(ridge_scores)
    best_rank_mse, _, best_rank = min(rank_scores)
    return InverseHyperparameters(
        ridge_lambda=float(best_ridge),
        tsvd_rank=int(best_rank),
        ridge_validation_mse=float(best_ridge_mse),
        tsvd_validation_mse=float(best_rank_mse),
    )
