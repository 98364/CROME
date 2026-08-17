from dataclasses import replace

import numpy as np
import pytest

from crome_identification.certification.decision import decide_target
from crome_identification.certification.temporal import (
    certify_temporal_conditional,
    certify_temporal_design_known,
    certify_temporal_empirical,
    certify_temporal_gap_conditional,
)
from crome_identification.certification.types import (
    CertificateClaimType,
    CertificateScope,
    CertificationMode,
    DecisionStatus,
    TargetInterval,
    SupportGapProof,
)


def test_design_known_accumulation_returns_point_constraint():
    certificate = certify_temporal_design_known(
        True,
        lags=np.array([0.01]),
        responses=np.array([1.0]),
        bandwidth=0.1,
        holder_L=0.0,
        holder_alpha=1.0,
        response_error_bound=0.02,
        provenance="unit-test schedule and trace bound",
    )

    assert certificate.valid
    assert certificate.mode is CertificationMode.DESIGN_KNOWN
    assert certificate.scope is CertificateScope.POPULATION_EXACT
    assert certificate.feasible_set == TargetInterval(0.98, 1.02)
    assert certificate.null_witness is None


def test_design_known_gap_with_anchored_set_returns_bounded_constraint():
    certificate = certify_temporal_gap_conditional(
        np.array([0.5, 1.0]),
        np.array([1.0, 1.0]),
        holder_L=1.0,
        holder_alpha=1.0,
        response_error_bound=0.0,
        deterministic_uncertainty=True,
        provenance="unit-test support gap and envelope",
    )

    assert certificate.valid
    assert certificate.feasible_set == TargetInterval(0.5, 1.5)
    assert certificate.point_estimate is None

    decision = decide_target([certificate], scientific_tolerance=0.05)
    assert decision.status is DecisionStatus.SET_ESTIMABLE
    assert decision.point_estimate is None


def test_boolean_gap_label_without_proof_is_inconclusive():
    certificate = certify_temporal_design_known(False)

    assert not certificate.valid
    assert certificate.feasible_set is None
    assert certificate.null_witness is None
    assert certificate.claim_type is CertificateClaimType.NUMERICAL_DIAGNOSTIC

    decision = decide_target([certificate], scientific_tolerance=0.05)
    assert decision.status is DecisionStatus.INCONCLUSIVE


def test_verified_support_gap_can_certify_nonidentification():
    proof = SupportGapProof(
        observed_support=("0.5", 1, 2),
        gap=(0, "0.25"),
        observed_signature_left=(0, 0, 0),
        observed_signature_right=(0, 0, 0),
        target_value_left=0,
        target_value_right=1,
        response_class_premise="closed under supplied local bump",
        support_specification="frozen deterministic schedule",
        units="hours",
        provenance="unit-test schedule",
        baseline_level=0,
        bump_amplitude=1,
        bump_power=2,
        schema_version="crome.support-gap.v1",
        producer_version="crome-identification/0.1.0",
        artifact_hash="sha256:" + "0" * 64,
        target_id="scalar_target",
    )
    proof = replace(proof, artifact_hash=proof.recompute_artifact_hash())

    certificate = certify_temporal_design_known(
        False,
        support_gap_proof=proof,
        provenance="unit-test schedule",
    )

    assert certificate.valid
    assert certificate.proof is proof
    assert certificate.null_witness is not None
    assert decide_target([certificate], scientific_tolerance=0.05).status is (
        DecisionStatus.NONRECOVERABLE
    )


def test_conditional_near_zero_mass_returns_auditable_error_radius():
    lags = np.linspace(0.001, 0.099, 100)
    responses = 1.0 + 0.2 * lags

    certificate = certify_temporal_conditional(
        lags,
        responses,
        bandwidth=0.1,
        n_units=1000,
        lower_mass_c=1.0,
        lower_mass_beta=1.0,
        holder_L=1.0,
        holder_alpha=1.0,
        noise_scale=0.05,
        delta=0.05,
        baseline_error=0.01,
        provenance="unit-test lower-mass fixture",
    )

    expected_radius = 0.1 + 0.05 * np.sqrt(2.0 * np.log(80.0) / 100.0) + 0.01
    assert certificate.valid
    assert certificate.mode is CertificationMode.ASSUMPTION_CONDITIONAL
    assert certificate.scope is CertificateScope.ASSUMPTION_CONDITIONAL
    assert certificate.error_radius == pytest.approx(expected_radius)
    assert certificate.diagnostics["n_near_zero"] == 100
    assert certificate.diagnostics["required_near_zero_count"] == pytest.approx(50.0)
    assert certificate.diagnostics["failure_probability_bound"] < 0.05


def test_insufficient_conditional_near_zero_mass_is_inconclusive():
    certificate = certify_temporal_conditional(
        np.array([0.9, 1.0]),
        np.array([1.0, 1.0]),
        bandwidth=0.1,
        n_units=20,
        lower_mass_c=1.0,
        lower_mass_beta=1.0,
        holder_L=1.0,
        holder_alpha=1.0,
        noise_scale=0.1,
        delta=0.05,
        provenance="unit-test insufficient-mass fixture",
    )

    assert not certificate.valid
    assert certificate.feasible_set is None
    assert "near-zero" in certificate.reason

    decision = decide_target([certificate], scientific_tolerance=0.2)
    assert decision.status is DecisionStatus.INCONCLUSIVE
    assert decision.point_estimate is None


def test_conditional_certificate_rejects_unjustified_independent_unit_assumption():
    certificate = certify_temporal_conditional(
        np.array([0.01, 0.02]),
        np.array([1.0, 1.0]),
        bandwidth=0.1,
        n_units=100,
        lower_mass_c=1.0,
        lower_mass_beta=1.0,
        holder_L=1.0,
        holder_alpha=1.0,
        noise_scale=0.1,
        delta=0.05,
        independent_units=False,
        provenance="unit-test dependent fixture",
    )

    assert not certificate.valid
    assert "independent" in certificate.reason


def test_conditional_certificate_rejects_more_rows_than_independent_units():
    with pytest.raises(ValueError, match="cannot exceed n_units"):
        certify_temporal_conditional(
            np.array([0.01, 0.02, 0.03]),
            np.array([1.0, 1.0, 1.0]),
            bandwidth=0.1,
            n_units=2,
            lower_mass_c=1.0,
            lower_mass_beta=1.0,
            holder_L=1.0,
            holder_alpha=1.0,
            noise_scale=0.1,
            delta=0.05,
            provenance="unit-test duplicated-unit fixture",
        )


def test_empirical_lag_diagnostics_cannot_issue_point_status():
    certificate = certify_temporal_empirical(np.array([0.001, 0.1, 0.5]))

    assert certificate.valid
    assert certificate.mode is CertificationMode.EMPIRICAL_ONLY
    assert certificate.scope is CertificateScope.FINITE_SAMPLE_ONLY
    assert certificate.diagnostics["minimum_positive_lag"] == pytest.approx(0.001)

    decision = decide_target([certificate], scientific_tolerance=0.01)
    assert decision.status is DecisionStatus.INCONCLUSIVE
    assert decision.point_estimate is None


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"bandwidth": 0.0}, "bandwidth"),
        ({"lower_mass_c": 0.0}, "lower_mass_c"),
        ({"lower_mass_beta": 0.0}, "lower_mass_beta"),
        ({"holder_L": -1.0}, "holder_L"),
        ({"holder_alpha": 0.0}, "holder_alpha"),
        ({"noise_scale": -1.0}, "noise_scale"),
        ({"delta": 1.0}, "delta"),
    ],
)
def test_conditional_certificate_validates_assumption_parameters(kwargs, match):
    inputs = {
        "bandwidth": 0.1,
        "n_units": 100,
        "lower_mass_c": 1.0,
        "lower_mass_beta": 1.0,
        "holder_L": 1.0,
        "holder_alpha": 1.0,
        "noise_scale": 0.1,
        "delta": 0.05,
    }
    inputs.update(kwargs)

    with pytest.raises(ValueError, match=match):
        certify_temporal_conditional(
            np.array([0.01, 0.02]),
            np.array([1.0, 1.0]),
            provenance="unit-test invalid assumption fixture",
            **inputs,
        )
