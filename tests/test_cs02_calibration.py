from dataclasses import replace

import numpy as np
import pytest

from crome_identification.certification import (
    AnchoredEnvelopeProof,
    FailureSemantics,
    calibrate_gaussian_design_error,
    calibrate_gaussian_outcome_noise,
    calibrate_lower_mass,
    certify_temporal_gap_conditional,
    verify_proof,
)


def test_lower_mass_calibration_returns_one_sided_positive_bound():
    budget = calibrate_lower_mass(
        np.linspace(0.001, 1.0, 500),
        bandwidth=0.4,
        beta=1.0,
        delta=0.05,
    )

    assert budget.valid
    assert 0 < budget.probability_lower < 0.4
    assert 0 < budget.lower_mass_c <= 1.0
    assert budget.n_calibration == 500


def test_lower_mass_calibration_marks_empty_near_zero_region_invalid():
    budget = calibrate_lower_mass(
        np.linspace(0.5, 1.0, 100),
        bandwidth=0.4,
        beta=1.0,
        delta=0.05,
    )

    assert not budget.valid
    assert budget.lower_mass_c == 0.0


def test_paired_design_replicates_produce_positive_operator_budget():
    rng = np.random.default_rng(4)
    truth = rng.normal(size=(150, 3))
    A1 = truth + rng.normal(0.0, 0.01, size=truth.shape)
    A2 = truth + rng.normal(0.0, 0.01, size=truth.shape)

    budget = calibrate_gaussian_design_error(A1, A2, n_test=200, delta=0.05)

    assert budget.sigma_upper > 0.01
    assert budget.operator_error_bound > 0
    assert budget.parameter_dim == 3
    assert budget.n_test == 200


def test_outcome_residuals_produce_vector_and_coordinate_budgets():
    rng = np.random.default_rng(8)
    residuals = rng.normal(0.0, 0.02, size=200)

    budget = calibrate_gaussian_outcome_noise(residuals, n_test=100, delta=0.05)

    assert budget.sigma_upper > 0
    assert budget.vector_norm_bound > budget.sigma_upper
    assert budget.simultaneous_coordinate_bound > budget.sigma_upper


def test_gap_certificate_contains_truth_after_noise_expansion():
    lags = np.array([0.5, 1.0])
    responses = 1.0 + 0.1 * lags

    cert = certify_temporal_gap_conditional(
        lags,
        responses,
        holder_L=0.2,
        holder_alpha=1.0,
        response_error_bound=0.05,
        deterministic_uncertainty=True,
    )

    assert cert.valid
    assert cert.feasible_set is not None
    assert cert.feasible_set.lower <= 1.0 <= cert.feasible_set.upper
    assert cert.scope.value == "assumption_conditional"
    assert cert.failure_semantics is FailureSemantics.DETERMINISTIC_BOUND
    assert isinstance(cert.proof, AnchoredEnvelopeProof)
    assert cert.proof.recompute_artifact_hash() == cert.proof.artifact_hash
    tampered = replace(cert.proof, responses=("99",) + cert.proof.responses[1:])
    assert not verify_proof(tampered).valid


def test_gap_certificate_rejects_empty_noise_expanded_intersection():
    cert = certify_temporal_gap_conditional(
        np.array([0.5, 0.5]),
        np.array([0.0, 2.0]),
        holder_L=0.1,
        holder_alpha=1.0,
        response_error_bound=0.01,
        deterministic_uncertainty=True,
    )

    assert not cert.valid
    assert "empty" in cert.reason


def test_probabilistic_gap_certificate_binds_its_actual_delta():
    cert = certify_temporal_gap_conditional(
        np.array([0.5, 1.0]),
        np.array([1.05, 1.1]),
        holder_L=0.2,
        holder_alpha=1.0,
        response_error_bound=0.05,
        budget_components=("response_noise",),
        component_delta=0.02,
    )

    assert cert.failure_semantics is FailureSemantics.PROBABILITY_LEDGER
    assert cert.failure_allocations == (("response_noise", 0.02),)
