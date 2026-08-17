from dataclasses import replace
import inspect

import numpy as np

import pytest

from crome_identification.certification.decision import (
    ModelInfeasibleError,
    decide_target,
)
from crome_identification.certification.overlap import certify_overlap_target
from crome_identification.certification.types import (
    CertificateClaimType,
    CertificateScope,
    CertificationMode,
    DecisionStatus,
    ExactNullProof,
    FailureBudget,
    FailureBudgetLedger,
    AnchoredEnvelopeProof,
    OperationalStatus,
    StructuralStatus,
    TargetCertificate,
    TargetInterval,
)


def _bound_proof(
    interval: TargetInterval,
    *,
    provenance: str,
    target_id: str = "unit-test-target",
) -> AnchoredEnvelopeProof:
    proof = AnchoredEnvelopeProof(
        lags=("1.0",),
        responses=(repr(interval.midpoint),),
        holder_L="0.0",
        holder_alpha="1.0",
        response_error_bound=repr(interval.width / 2.0),
        claimed_lower=repr(interval.lower),
        claimed_upper=repr(interval.upper),
        schema_version="crome.anchored-envelope.v1",
        producer_version="crome-identification/0.1.0",
        artifact_hash="sha256:" + "0" * 64,
        units="response units",
        premise="unit-test data-bound anchored interval",
        provenance=provenance,
        failure_allocations=(),
        target_id=target_id,
    )
    return replace(proof, artifact_hash=proof.recompute_artifact_hash())


def _certificate(
    source: str,
    *,
    feasible_set: TargetInterval | None = None,
    null_witness: tuple[float, ...] | None = None,
    valid: bool = True,
    mode: CertificationMode = CertificationMode.DESIGN_KNOWN,
) -> TargetCertificate:
    proof = None
    claim_type = CertificateClaimType.NUMERICAL_DIAGNOSTIC
    provenance = "unit-test proof payload"
    if feasible_set is not None:
        proof = _bound_proof(feasible_set, provenance=provenance)
        claim_type = CertificateClaimType.STRUCTURAL_EVIDENCE
    if null_witness is not None:
        if len(null_witness) != 2:
            raise ValueError("test helper only supports a two-coordinate null witness")
        proof = ExactNullProof(
            operator=((1, 1),),
            target=(1, -1),
            witness=(1, -1),
            parameter_space="unrestricted",
            operator_id="sha256:unit-test",
            target_id="unit-test-target",
            units="response units",
            provenance=provenance,
        )
        claim_type = CertificateClaimType.EXACT_WITNESS
    return TargetCertificate(
        source=source,
        mode=mode,
        scope=(
            CertificateScope.FINITE_SAMPLE_ONLY
            if mode is CertificationMode.EMPIRICAL_ONLY
            else CertificateScope.POPULATION_EXACT
        ),
        valid=valid,
        feasible_set=feasible_set,
        null_witness=null_witness,
        claim_type=claim_type,
        provenance=provenance,
        proof=proof,
        target_id="unit-test-target",
    )


def test_target_interval_intersection_is_commutative():
    left = TargetInterval(0.0, 2.0)
    right = TargetInterval(1.0, 3.0)

    assert left.intersect(right) == TargetInterval(1.0, 2.0)
    assert right.intersect(left) == TargetInterval(1.0, 2.0)


def test_single_public_certificate_requires_a_bound_target_identifier():
    certificate = certify_overlap_target(
        np.eye(1),
        np.array([1.0]),
        np.array([1.0]),
        noise_radius=0.0,
        design_error_bound=0.0,
        theta_radius=1.0,
        exact_design=True,
        provenance="unit-test exact linear target",
    )

    decision = decide_target(
        [replace(certificate, target_id="")], scientific_tolerance=0.0
    )

    assert decision.status is DecisionStatus.INCONCLUSIVE
    assert "target identifier" in " ".join(decision.reasons)


def test_different_linear_target_functionals_cannot_share_one_composition():
    certificates = [
        certify_overlap_target(
            np.eye(2),
            np.array([1.0, 1.0]),
            target,
            noise_radius=0.0,
            design_error_bound=0.0,
            theta_radius=1.0,
            exact_design=True,
            provenance="unit-test exact linear target",
        )
        for target in (np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    ]

    decision = decide_target(certificates, scientific_tolerance=0.0)

    assert decision.status is DecisionStatus.INCONCLUSIVE
    assert "target" in " ".join(decision.reasons)


def test_linear_proof_delta_cannot_be_relabelled_by_the_outer_certificate():
    certificate = certify_overlap_target(
        np.eye(1),
        np.array([1.0]),
        np.array([1.0]),
        noise_radius=0.1,
        design_error_bound=0.0,
        theta_radius=1.0,
        exact_design=True,
        budget_components=("outcome_noise",),
        component_delta=0.1,
        provenance="unit-test calibrated noise",
    )
    forged = replace(
        certificate,
        failure_allocations=(("outcome_noise", 0.001),),
    )
    ledger = FailureBudgetLedger(
        total_delta=0.01,
        allocations=(
            FailureBudget("outcome_noise", 0.001, "forged smaller allocation"),
        ),
    )

    decision = decide_target(
        [forged], scientific_tolerance=0.2, failure_ledger=ledger
    )

    assert decision.status is DecisionStatus.INCONCLUSIVE
    assert "proof and certificate delta" in " ".join(decision.reasons)


def test_point_decision_is_the_only_status_with_point_output():
    temporal = _certificate("temporal", feasible_set=TargetInterval(0.99, 1.01))
    overlap = _certificate("overlap", feasible_set=TargetInterval(0.98, 1.02))

    decision = decide_target(
        [temporal, overlap],
        scientific_tolerance=0.01,
    )

    assert decision.status is DecisionStatus.POINT_ESTIMABLE
    assert decision.feasible_set == TargetInterval(0.99, 1.01)
    assert decision.point_estimate == pytest.approx(1.0)
    assert decision.uncertainty == decision.feasible_set


def test_bounded_feasible_set_returns_set_without_point_leakage():
    decision = decide_target(
        [_certificate("temporal", feasible_set=TargetInterval(0.5, 1.5))],
        scientific_tolerance=0.01,
    )

    assert decision.status is DecisionStatus.SET_ESTIMABLE
    assert decision.feasible_set == TargetInterval(0.5, 1.5)
    assert decision.point_estimate is None
    assert decision.uncertainty == decision.feasible_set


def test_unbounded_exact_null_witness_returns_nonrecoverable_without_point():
    decision = decide_target(
        [_certificate("overlap", null_witness=(1.0, -1.0))],
        scientific_tolerance=0.01,
    )

    assert decision.status is DecisionStatus.NONRECOVERABLE
    assert not decision.feasible_set.is_bounded
    assert decision.point_estimate is None
    assert decision.uncertainty is None


def test_empirical_only_evidence_is_inconclusive_even_when_interval_is_narrow():
    empirical = _certificate(
        "temporal",
        feasible_set=TargetInterval(0.999, 1.001),
        mode=CertificationMode.EMPIRICAL_ONLY,
    )

    decision = decide_target([empirical], scientific_tolerance=0.01)

    assert decision.status is DecisionStatus.INCONCLUSIVE
    assert decision.certificate_scope is CertificateScope.FINITE_SAMPLE_ONLY
    assert decision.point_estimate is None
    assert decision.uncertainty is None


def test_invalid_evidence_is_inconclusive():
    decision = decide_target(
        [_certificate("temporal", valid=False)],
        scientific_tolerance=0.01,
    )

    assert decision.status is DecisionStatus.INCONCLUSIVE
    assert decision.point_estimate is None


def test_parameter_bounds_do_not_erase_structural_nonidentification():
    decision = decide_target(
        [_certificate("overlap", null_witness=(1.0, -1.0))],
        parameter_space=TargetInterval(-2.0, 2.0),
        scientific_tolerance=0.01,
    )

    assert decision.status is DecisionStatus.NONRECOVERABLE
    assert decision.structural_status is StructuralStatus.NONIDENTIFIED
    assert decision.operational_status is OperationalStatus.SET
    assert decision.feasible_set == TargetInterval(-2.0, 2.0)
    assert decision.point_estimate is None


def test_conflicting_constraints_raise_model_infeasible_error():
    with pytest.raises(ModelInfeasibleError, match="empty target feasible set"):
        decide_target(
            [
                _certificate("temporal", feasible_set=TargetInterval(0.0, 1.0)),
                _certificate("overlap", feasible_set=TargetInterval(2.0, 3.0)),
            ],
            scientific_tolerance=0.01,
        )


def test_negative_scientific_tolerance_is_rejected():
    with pytest.raises(ValueError, match="scientific_tolerance"):
        decide_target(
            [_certificate("temporal", feasible_set=TargetInterval(0.0, 1.0))],
            scientific_tolerance=-1.0,
        )


def test_unbounded_interval_serializes_infinite_endpoints_as_explicit_null():
    payload = TargetInterval.unbounded().as_dict()

    assert payload["lower"] is None
    assert payload["upper"] is None
    assert payload["width"] is None
    assert payload["bounded"] is False


def test_public_router_has_no_switches_that_disable_safety_checks():
    parameters = inspect.signature(decide_target).parameters

    assert "enforce_global_ledger" not in parameters
    assert "enforce_typed_contract" not in parameters


def test_structural_nonidentification_and_point_at_tau_can_coexist():
    exact_null = TargetCertificate(
        source="exact-null",
        mode=CertificationMode.DESIGN_KNOWN,
        scope=CertificateScope.POPULATION_EXACT,
        valid=True,
        null_witness=(1.0, -1.0),
        claim_type=CertificateClaimType.EXACT_WITNESS,
        provenance="frozen fixture",
        proof=ExactNullProof(
            operator=((1, 1),),
            target=(1, -1),
            witness=(1, -1),
            parameter_space="box",
            operator_id="sha256:fixture",
            target_id="unit-test-target",
            units="response units",
            provenance="frozen fixture",
            base_parameter=(0, 0),
            step="0.005",
            parameter_lower=("-0.01", "-0.01"),
            parameter_upper=("0.01", "0.01"),
        ),
        target_id="unit-test-target",
    )
    tolerance_bound = TargetCertificate(
        source="bounded-prior",
        mode=CertificationMode.DESIGN_KNOWN,
        scope=CertificateScope.POPULATION_EXACT,
        valid=True,
        feasible_set=TargetInterval(-0.01, 0.01),
        claim_type=CertificateClaimType.STRUCTURAL_EVIDENCE,
        provenance="declared constrained target domain",
        proof=_bound_proof(
            TargetInterval(-0.01, 0.01),
            provenance="declared constrained target domain",
        ),
        target_id="unit-test-target",
    )

    decision = decide_target(
        [exact_null, tolerance_bound],
        scientific_tolerance=0.02,
    )

    assert decision.structural_status is StructuralStatus.NONIDENTIFIED
    assert decision.operational_status is OperationalStatus.POINT_AT_TAU
    assert decision.status is DecisionStatus.NONRECOVERABLE
    assert decision.point_estimate is None


def test_unverified_bounded_claim_is_inconclusive_even_when_label_says_valid():
    unverified = TargetCertificate(
        source="caller",
        mode=CertificationMode.DESIGN_KNOWN,
        scope=CertificateScope.POPULATION_EXACT,
        valid=True,
        feasible_set=TargetInterval(0.99, 1.01),
        claim_type=CertificateClaimType.STRUCTURAL_EVIDENCE,
        target_id="unit-test-target",
        provenance="caller label",
    )

    decision = decide_target([unverified], scientific_tolerance=0.02)

    assert decision.status is DecisionStatus.INCONCLUSIVE
    assert decision.operational_status is OperationalStatus.INCONCLUSIVE
    assert "proof payload" in " ".join(decision.reasons)
