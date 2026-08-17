"""Calibration budgets for split-sample CROME certification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
from scipy.stats import beta as beta_distribution
from scipy.stats import chi2


@dataclass(frozen=True)
class LowerMassCalibration:
    valid: bool
    lower_mass_c: float
    probability_lower: float
    n_near_zero: int
    n_calibration: int
    bandwidth: float
    beta: float
    delta: float
    assumptions: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["assumptions"] = list(self.assumptions)
        return out


@dataclass(frozen=True)
class DesignErrorCalibration:
    sigma_hat: float
    sigma_upper: float
    operator_error_bound: float
    n_calibration_entries: int
    n_test: int
    parameter_dim: int
    delta: float
    assumptions: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["assumptions"] = list(self.assumptions)
        return out


@dataclass(frozen=True)
class OutcomeNoiseCalibration:
    sigma_hat: float
    sigma_upper: float
    vector_norm_bound: float
    simultaneous_coordinate_bound: float
    n_calibration: int
    n_test: int
    delta: float
    assumptions: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["assumptions"] = list(self.assumptions)
        return out


@dataclass(frozen=True)
class BoundedDesignErrorCalibration:
    row_error_quantile: float
    operator_error_bound: float
    n_calibration: int
    n_test: int
    parameter_dim: int
    row_error_bound: float
    delta: float
    assumptions: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["assumptions"] = list(self.assumptions)
        return out


def _probability(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie strictly between zero and one")
    return value


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def calibrate_lower_mass(
    calibration_lags: np.ndarray,
    *,
    bandwidth: float,
    beta: float,
    delta: float,
) -> LowerMassCalibration:
    """One-sided lower confidence bound for ``P(0<Q<=h) / h**beta``."""

    lags = np.asarray(calibration_lags, dtype=float)
    if lags.ndim != 1 or lags.size == 0 or not np.all(np.isfinite(lags)):
        raise ValueError("calibration_lags must be a non-empty finite vector")
    if np.any(lags <= 0):
        raise ValueError("calibration lags must be strictly positive")
    bandwidth = _positive("bandwidth", bandwidth)
    beta = _positive("beta", beta)
    delta = _probability("delta", delta)
    successes = int(np.sum(lags <= bandwidth))
    probability_lower = (
        float(beta_distribution.ppf(delta, successes, lags.size - successes + 1))
        if successes > 0
        else 0.0
    )
    lower_mass_c = probability_lower / bandwidth**beta
    return LowerMassCalibration(
        valid=successes > 0 and lower_mass_c > 0.0,
        lower_mass_c=float(lower_mass_c),
        probability_lower=probability_lower,
        n_near_zero=successes,
        n_calibration=int(lags.size),
        bandwidth=bandwidth,
        beta=beta,
        delta=delta,
        assumptions=(
            "exchangeable calibration trajectories",
            f"P(0<Q<=h) >= c h^{beta:g}",
            "bandwidth fixed before test evaluation",
        ),
    )


def calibrate_gaussian_design_error(
    first_design: np.ndarray,
    second_design: np.ndarray,
    *,
    n_test: int,
    delta: float,
) -> DesignErrorCalibration:
    """Calibrate a Gaussian operator-error bound from paired design replicates."""

    first = np.asarray(first_design, dtype=float)
    second = np.asarray(second_design, dtype=float)
    if (
        first.ndim != 2
        or first.shape != second.shape
        or first.size == 0
        or not np.all(np.isfinite(first))
        or not np.all(np.isfinite(second))
    ):
        raise ValueError("paired designs must be finite non-empty matrices of equal shape")
    if not isinstance(n_test, (int, np.integer)) or int(n_test) <= 0:
        raise ValueError("n_test must be a positive integer")
    n_test = int(n_test)
    delta = _probability("delta", delta)
    difference = first - second
    degrees = int(difference.size)
    sum_squares = float(np.sum(np.square(difference)))
    if sum_squares <= 0.0:
        raise ValueError("paired designs contain no measurable perturbation")
    sigma_hat = math.sqrt(sum_squares / (2.0 * degrees))
    scale_delta = delta / 2.0
    spectral_delta = delta / 2.0
    sigma_upper = math.sqrt(sum_squares / (2.0 * chi2.ppf(scale_delta, degrees)))
    parameter_dim = int(first.shape[1])
    operator_error_bound = sigma_upper * (
        math.sqrt(n_test)
        + math.sqrt(parameter_dim)
        + math.sqrt(2.0 * math.log(1.0 / spectral_delta))
    )
    return DesignErrorCalibration(
        sigma_hat=float(sigma_hat),
        sigma_upper=float(sigma_upper),
        operator_error_bound=float(operator_error_bound),
        n_calibration_entries=degrees,
        n_test=n_test,
        parameter_dim=parameter_dim,
        delta=delta,
        assumptions=(
            "paired independent design measurements",
            "independent centered homoskedastic Gaussian entry errors",
            "test design perturbations share the calibrated scale",
        ),
    )


def calibrate_bounded_design_error(
    reference_design: np.ndarray,
    observed_design: np.ndarray,
    *,
    n_test: int,
    row_error_bound: float,
    delta: float,
) -> BoundedDesignErrorCalibration:
    """Calibrate an assumption-conditional operator budget from row errors.

    The split-conformal order statistic controls a new row marginally under
    exchangeability.  Turning it into ``sqrt(n_test) * q`` is explicitly
    conditional on the test operator's row errors satisfying the same bound;
    this routine therefore must not be cited as a population-exact certificate.
    """

    reference = np.asarray(reference_design, dtype=float)
    observed = np.asarray(observed_design, dtype=float)
    if (
        reference.ndim != 2
        or reference.shape != observed.shape
        or reference.shape[0] == 0
        or not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(observed))
    ):
        raise ValueError("reference and observed designs must be equal finite matrices")
    if not isinstance(n_test, (int, np.integer)) or int(n_test) <= 0:
        raise ValueError("n_test must be a positive integer")
    bound = _positive("row_error_bound", row_error_bound)
    delta = _probability("delta", delta)
    errors = np.linalg.norm(reference - observed, axis=1)
    if np.any(errors > bound + 1e-12 * max(1.0, bound)):
        raise ValueError("observed calibration error exceeds row_error_bound")
    n_calibration = int(errors.size)
    rank = int(math.ceil((n_calibration + 1) * (1.0 - delta)))
    quantile = bound if rank > n_calibration else float(np.partition(errors, rank - 1)[rank - 1])
    return BoundedDesignErrorCalibration(
        row_error_quantile=quantile,
        operator_error_bound=float(math.sqrt(int(n_test)) * quantile),
        n_calibration=n_calibration,
        n_test=int(n_test),
        parameter_dim=int(reference.shape[1]),
        row_error_bound=bound,
        delta=delta,
        assumptions=(
            "exchangeable calibration and test design rows",
            "bounded independent row errors",
            "test row errors obey the calibrated simultaneous row bound",
        ),
    )


def calibrate_gaussian_outcome_noise(
    calibration_residuals: np.ndarray,
    *,
    n_test: int,
    delta: float,
) -> OutcomeNoiseCalibration:
    """Calibrate vector and simultaneous-coordinate Gaussian noise bounds."""

    residuals = np.asarray(calibration_residuals, dtype=float)
    if residuals.ndim != 1 or residuals.size == 0 or not np.all(np.isfinite(residuals)):
        raise ValueError("calibration_residuals must be a non-empty finite vector")
    if not isinstance(n_test, (int, np.integer)) or int(n_test) <= 0:
        raise ValueError("n_test must be a positive integer")
    n_test = int(n_test)
    delta = _probability("delta", delta)
    degrees = int(residuals.size)
    sum_squares = float(np.sum(np.square(residuals)))
    if sum_squares <= 0.0:
        raise ValueError("calibration residuals contain no measurable variation")
    sigma_hat = math.sqrt(sum_squares / degrees)
    scale_delta = delta / 3.0
    norm_delta = delta / 3.0
    coordinate_delta = delta / 3.0
    sigma_upper = math.sqrt(sum_squares / chi2.ppf(scale_delta, degrees))
    vector_norm_bound = sigma_upper * math.sqrt(chi2.ppf(1.0 - norm_delta, n_test))
    simultaneous_coordinate_bound = sigma_upper * math.sqrt(
        2.0 * math.log(2.0 * n_test / coordinate_delta)
    )
    return OutcomeNoiseCalibration(
        sigma_hat=float(sigma_hat),
        sigma_upper=float(sigma_upper),
        vector_norm_bound=float(vector_norm_bound),
        simultaneous_coordinate_bound=float(simultaneous_coordinate_bound),
        n_calibration=degrees,
        n_test=n_test,
        delta=delta,
        assumptions=(
            "exchangeable calibration and test trajectories",
            "independent centered homoskedastic Gaussian residuals",
            "calibration residual model fixed before test evaluation",
        ),
    )
