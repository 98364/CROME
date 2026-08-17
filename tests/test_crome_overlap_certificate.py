import numpy as np
import pytest

from crome_identification.certification.decision import decide_target
from crome_identification.certification.overlap import (
    certify_overlap_target,
    target_direct_weights,
)
from crome_identification.certification.types import (
    CertificateClaimType,
    CertificateScope,
    CertificationMode,
    DecisionStatus,
    ExactNullProof,
    FailureSemantics,
)


def test_target_direct_weights_satisfy_dual_equation_for_identified_target():
    A = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    C = np.array([1.0, -1.0])

    weights, residual = target_direct_weights(A, C)

    assert weights.shape == (A.shape[0],)
    assert weights @ A == pytest.approx(C)
    assert residual == pytest.approx(0.0, abs=1e-12)


def test_direct_target_estimator_matches_identified_projection():
    certificate = certify_overlap_target(
        np.eye(2),
        np.array([1.0, 2.0]),
        np.array([1.0, 0.0]),
        noise_radius=0.0,
        design_error_bound=0.0,
        theta_radius=3.0,
        exact_design=True,
        provenance="unit-test exact identity design",
    )

    assert certificate.valid
    assert certificate.mode is CertificationMode.DESIGN_KNOWN
    assert certificate.scope is CertificateScope.POPULATION_EXACT
    assert certificate.point_estimate == pytest.approx(1.0)
    assert certificate.error_radius == pytest.approx(0.0)
    assert certificate.feasible_set.lower == pytest.approx(1.0)
    assert certificate.feasible_set.upper == pytest.approx(1.0)
    assert certificate.diagnostics["target_residual"] == pytest.approx(0.0)


def test_float_null_search_is_only_a_numerical_diagnostic():
    A = np.array([[1.0, 1.0]])
    C = np.array([1.0, -1.0])

    certificate = certify_overlap_target(
        A,
        np.array([0.5]),
        C,
        noise_radius=0.0,
        design_error_bound=0.0,
        theta_radius=3.0,
        exact_design=True,
        provenance="unit-test floating design",
    )

    assert certificate.valid
    assert certificate.feasible_set is None
    assert certificate.point_estimate is None
    assert certificate.null_witness is None
    assert certificate.claim_type is CertificateClaimType.NUMERICAL_DIAGNOSTIC
    assert "candidate_null_witness" in certificate.diagnostics

    decision = decide_target([certificate], scientific_tolerance=0.05)
    assert decision.status is DecisionStatus.INCONCLUSIVE
    assert decision.point_estimate is None


def test_exact_proof_payload_can_certify_target_nonidentification():
    proof = ExactNullProof(
        operator=((1, 1),),
        target=(1, -1),
        witness=(1, -1),
        parameter_space="unrestricted",
        operator_id="sha256:unit-test",
        target_id="unit-test-target",
        units="response units",
        provenance="unit-test exact operator",
    )

    certificate = certify_overlap_target(
        np.array([[1.0, 1.0]]),
        np.array([0.5]),
        np.array([1.0, -1.0]),
        noise_radius=0.0,
        design_error_bound=0.0,
        theta_radius=3.0,
        exact_design=True,
        exact_null_proof=proof,
        target_id="unit-test-target",
        provenance="unit-test exact operator",
    )

    assert certificate.null_witness == (1.0, -1.0)
    assert certificate.proof is proof
    assert decide_target([certificate], scientific_tolerance=0.05).status is (
        DecisionStatus.NONRECOVERABLE
    )


def test_design_error_budget_can_change_point_to_set():
    common = {
        "Ahat": np.eye(2),
        "y": np.array([1.0, 2.0]),
        "C": np.array([1.0, 0.0]),
        "noise_radius": 0.01,
        "theta_radius": 3.0,
        "exact_design": False,
        "deterministic_uncertainty": True,
        "provenance": "unit-test bounded-error inputs",
    }
    tight = certify_overlap_target(**common, design_error_bound=0.0)
    wide = certify_overlap_target(**common, design_error_bound=0.2)

    tight_decision = decide_target([tight], scientific_tolerance=0.05)
    wide_decision = decide_target([wide], scientific_tolerance=0.05)

    assert tight.error_radius == pytest.approx(0.01)
    assert wide.error_radius == pytest.approx(0.61)
    assert tight_decision.status is DecisionStatus.POINT_ESTIMABLE
    assert wide_decision.status is DecisionStatus.SET_ESTIMABLE
    assert wide_decision.point_estimate is None


def test_baseline_and_approximation_budgets_enter_target_radius_additively():
    certificate = certify_overlap_target(
        np.eye(2),
        np.array([1.0, 2.0]),
        np.array([1.0, 0.0]),
        noise_radius=0.1,
        design_error_bound=0.0,
        theta_radius=3.0,
        baseline_error=0.2,
        approximation_error=0.3,
        exact_design=False,
        deterministic_uncertainty=True,
        provenance="unit-test bounded-error inputs",
    )

    assert certificate.error_radius == pytest.approx(0.6)
    assert certificate.diagnostics["baseline_error"] == pytest.approx(0.2)
    assert certificate.diagnostics["approximation_error"] == pytest.approx(0.3)


def test_probabilistic_overlap_requires_explicit_delta_binding():
    with pytest.raises(ValueError, match="component_delta"):
        certify_overlap_target(
            np.eye(2),
            np.array([1.0, 2.0]),
            np.array([1.0, 0.0]),
            noise_radius=0.01,
            design_error_bound=0.0,
            theta_radius=3.0,
            exact_design=False,
            provenance="unit-test probabilistic uncertainty",
        )

    deterministic = certify_overlap_target(
        np.eye(2),
        np.array([1.0, 2.0]),
        np.array([1.0, 0.0]),
        noise_radius=0.01,
        design_error_bound=0.0,
        theta_radius=3.0,
        exact_design=False,
        deterministic_uncertainty=True,
        provenance="unit-test deterministic norm bound",
    )
    assert deterministic.failure_semantics is FailureSemantics.DETERMINISTIC_BOUND


@pytest.mark.parametrize(
    "field,value",
    [
        ("noise_radius", -0.1),
        ("design_error_bound", -0.1),
        ("theta_radius", -0.1),
        ("baseline_error", -0.1),
        ("approximation_error", -0.1),
    ],
)
def test_negative_error_budgets_are_rejected(field, value):
    kwargs = {
        "noise_radius": 0.0,
        "design_error_bound": 0.0,
        "theta_radius": 1.0,
        "baseline_error": 0.0,
        "approximation_error": 0.0,
        "exact_design": False,
        "provenance": "unit-test invalid budget inputs",
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        certify_overlap_target(
            np.eye(2),
            np.array([1.0, 2.0]),
            np.array([1.0, 0.0]),
            **kwargs,
        )


def test_overlap_certificate_rejects_vector_valued_target_in_minimal_prototype():
    with pytest.raises(ValueError, match="scalar target"):
        certify_overlap_target(
            np.eye(2),
            np.array([1.0, 2.0]),
            np.eye(2),
            noise_radius=0.0,
            design_error_bound=0.0,
            theta_radius=1.0,
            exact_design=True,
            provenance="unit-test vector target rejection",
        )
