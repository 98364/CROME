"""Synthetic geometry whose temporal support and overlap operator share one cause.

The target coefficient enters an observation row only for target-mark events whose
lag lies in the declared near-zero window.  Consequently, deleting near-zero
target support also deletes the target column from the population operator; this
prevents the semantic mismatch in which overlap recovers a target that the
temporal module calls nonrecoverable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


_PERTURBATIONS = {
    "clean",
    "timestamp_jitter",
    "rounding",
    "basis_error",
    "mark_noise",
    "baseline_error",
}


@dataclass(frozen=True)
class PerturbationSpec:
    kind: str = "clean"
    level: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in _PERTURBATIONS:
            raise ValueError(f"unknown perturbation kind: {self.kind}")
        if not math.isfinite(self.level) or self.level < 0:
            raise ValueError("perturbation level must be finite and non-negative")
        if self.kind == "mark_noise" and self.level > 1:
            raise ValueError("mark-noise level must not exceed one")


@dataclass(frozen=True)
class AlignedGeometry:
    trajectory_ids: np.ndarray
    lags: np.ndarray
    observed_lags: np.ndarray
    marks: np.ndarray
    observed_marks: np.ndarray
    true_design: np.ndarray
    observed_design: np.ndarray
    replicate_design: np.ndarray
    outcomes: np.ndarray
    trace_responses: np.ndarray
    theta: np.ndarray
    target: np.ndarray
    true_target: float
    truth_status: str
    target_support_mass: float
    support_bandwidth: float
    perturbation: PerturbationSpec


def _validate(
    n_units: int,
    parameter_dim: int,
    target_support_mass: float,
    support_bandwidth: float,
) -> tuple[int, int, float, float]:
    if not isinstance(n_units, (int, np.integer)) or int(n_units) < 2:
        raise ValueError("n_units must be an integer of at least two")
    if not isinstance(parameter_dim, (int, np.integer)) or int(parameter_dim) < 2:
        raise ValueError("parameter_dim must be an integer of at least two")
    mass = float(target_support_mass)
    bandwidth = float(support_bandwidth)
    if not math.isfinite(mass) or not 0 <= mass <= 1:
        raise ValueError("target_support_mass must lie in [0, 1]")
    if not math.isfinite(bandwidth) or not 0 < bandwidth < 0.5:
        raise ValueError("support_bandwidth must lie in (0, 0.5)")
    return int(n_units), int(parameter_dim), mass, bandwidth


def _balanced_marks(n_units: int, parameter_dim: int, rng: np.random.Generator) -> np.ndarray:
    marks = np.arange(n_units, dtype=int) % parameter_dim
    rng.shuffle(marks)
    return marks


def _operator(
    lags: np.ndarray,
    marks: np.ndarray,
    parameter_dim: int,
    support_bandwidth: float,
) -> np.ndarray:
    design = np.zeros((lags.size, parameter_dim), dtype=float)
    basis = np.exp(-lags)
    nuisance = marks != 0
    design[np.flatnonzero(nuisance), marks[nuisance]] = basis[nuisance]
    target_rows = (marks == 0) & (lags <= support_bandwidth)
    design[target_rows, 0] = basis[target_rows]
    return design


def _perturb_operator(
    true_lags: np.ndarray,
    true_marks: np.ndarray,
    parameter_dim: int,
    support_bandwidth: float,
    spec: PerturbationSpec,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed_lags = true_lags.copy()
    observed_marks = true_marks.copy()
    if spec.kind == "timestamp_jitter" and spec.level > 0:
        observed_lags = np.maximum(
            np.finfo(float).eps,
            true_lags + rng.uniform(-spec.level, spec.level, size=true_lags.size),
        )
    elif spec.kind == "rounding" and spec.level > 0:
        observed_lags = np.maximum(
            np.finfo(float).eps,
            np.round(true_lags / spec.level) * spec.level,
        )
    elif spec.kind == "mark_noise" and spec.level > 0:
        corrupted = rng.random(true_marks.size) < spec.level
        offsets = rng.integers(1, parameter_dim, size=true_marks.size)
        observed_marks[corrupted] = (true_marks[corrupted] + offsets[corrupted]) % parameter_dim

    design = _operator(observed_lags, observed_marks, parameter_dim, support_bandwidth)
    if spec.kind == "basis_error" and spec.level > 0:
        active = design != 0
        design = design + active * rng.normal(0.0, spec.level, size=design.shape)
    return observed_lags, observed_marks, design


def generate_aligned_geometry(
    *,
    n_units: int,
    parameter_dim: int,
    target_support_mass: float,
    perturbation: PerturbationSpec | None = None,
    seed: int = 0,
    support_bandwidth: float = 0.1,
    support_gap: float = 0.35,
    outcome_noise_sigma: float = 0.03,
    trace_noise_sigma: float = 0.03,
    temporal_slope: float = 0.05,
    theta: np.ndarray | None = None,
) -> AlignedGeometry:
    """Generate one aligned synthetic benchmark sample.

    ``target_support_mass`` is the population probability that a target-mark row
    lies in the near-zero interval.  The same draw determines both temporal
    evidence and whether column zero appears in the true response operator.
    """

    n_units, parameter_dim, mass, bandwidth = _validate(
        n_units, parameter_dim, target_support_mass, support_bandwidth
    )
    if not math.isfinite(support_gap) or not bandwidth < support_gap < 1:
        raise ValueError("support_gap must lie between support_bandwidth and one")
    spec = perturbation or PerturbationSpec()
    rng = np.random.default_rng(seed)
    marks = _balanced_marks(n_units, parameter_dim, rng)
    lags = rng.uniform(support_gap, 1.0, size=n_units)
    target_rows = np.flatnonzero(marks == 0)
    near_zero = rng.random(target_rows.size) < mass
    lags[target_rows[near_zero]] = rng.uniform(
        np.finfo(float).eps, bandwidth, size=int(np.sum(near_zero))
    )
    nuisance_rows = np.flatnonzero(marks != 0)
    lags[nuisance_rows] = rng.uniform(
        np.finfo(float).eps, 1.0, size=nuisance_rows.size
    )

    true_design = _operator(lags, marks, parameter_dim, bandwidth)
    observed_lags, observed_marks, observed_design = _perturb_operator(
        lags, marks, parameter_dim, bandwidth, spec, rng
    )
    _, _, replicate_design = _perturb_operator(
        lags, marks, parameter_dim, bandwidth, spec, rng
    )

    if theta is None:
        theta_values = np.linspace(1.0, -0.4, parameter_dim, dtype=float)
    else:
        theta_values = np.asarray(theta, dtype=float)
        if theta_values.shape != (parameter_dim,) or not np.all(np.isfinite(theta_values)):
            raise ValueError("theta must be a finite vector matching parameter_dim")
    target = np.zeros(parameter_dim, dtype=float)
    target[0] = 1.0
    outcomes = true_design @ theta_values + rng.normal(
        0.0, outcome_noise_sigma, size=n_units
    )
    if spec.kind == "baseline_error" and spec.level > 0:
        outcomes = outcomes + spec.level * np.sin(2.0 * np.pi * lags)
    trace_responses = theta_values[marks] + temporal_slope * lags + rng.normal(
        0.0, trace_noise_sigma, size=n_units
    )
    return AlignedGeometry(
        trajectory_ids=np.arange(n_units, dtype=int),
        lags=lags,
        observed_lags=observed_lags,
        marks=marks,
        observed_marks=observed_marks,
        true_design=true_design,
        observed_design=observed_design,
        replicate_design=replicate_design,
        outcomes=outcomes,
        trace_responses=trace_responses,
        theta=theta_values,
        target=target,
        true_target=float(target @ theta_values),
        truth_status="POINT_ESTIMABLE" if mass > 0 else "NONRECOVERABLE",
        target_support_mass=mass,
        support_bandwidth=bandwidth,
        perturbation=spec,
    )
