from dataclasses import replace

import numpy as np
import pytest

from crome_identification.certification import (
    CertificateClaimType,
    CertificateScope,
    CertificationMode,
    DecisionStatus,
    FailureBudget,
    FailureBudgetLedger,
    FailureSemantics,
    ExactNullProof,
    AnchoredEnvelopeProof,
    SetSubtype,
    TargetCertificate,
    TargetInterval,
    certificate_optimized_weights,
    certificate_radius,
    certify_overlap_target,
    certify_temporal_conditional,
    certify_temporal_design_known,
    decide_target,
)


def _bound_proof(
    lower: float,
    upper: float,
    provenance: str,
    *,
    target_id: str = "unit-test-target",
    units: str = "response units",
    failure_allocations: tuple[tuple[str, float], ...] = (),
) -> AnchoredEnvelopeProof:
    midpoint = (lower + upper) / 2.0
    radius = (upper - lower) / 2.0
    proof = AnchoredEnvelopeProof(
        lags=("1.0",),
        responses=(repr(midpoint),),
        holder_L="0.0",
        holder_alpha="1.0",
        response_error_bound=repr(radius),
        claimed_lower=repr(lower),
        claimed_upper=repr(upper),
        schema_version="crome.anchored-envelope.v1",
        producer_version="crome-identification/0.1.0",
        artifact_hash="sha256:" + "0" * 64,
        units=units,
        premise="unit-test data-bound anchored interval",
        provenance=provenance,
        failure_allocations=tuple(
            (name, repr(delta)) for name, delta in failure_allocations
        ),
        target_id=target_id,
    )
    return replace(proof, artifact_hash=proof.recompute_artifact_hash())


def _exact_null_proof() -> ExactNullProof:
    return ExactNullProof(
        operator=((1, 1),),
        target=(1, -1),
        witness=(1, -1),
        parameter_space="unrestricted",
        operator_id="sha256:unit-test",
        target_id="unit-test-target",
        units="response units",
        provenance="unit-test exact operator",
    )


def test_certificate_optimized_weights_do_not_exceed_least_squares_radius():
    design = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0e-3],
            [1.0, 2.0e-3],
        ]
    )
    target = np.array([0.0, 1.0])
    least_squares = np.linalg.lstsq(design.T, target, rcond=1e-12)[0]

    optimized, diagnostics = certificate_optimized_weights(
        design,
        target,
        theta_radius=2.0,
        design_error_bound=0.02,
        noise_radius=0.05,
        baseline_support_radius=0.01,
        approximation_support_radius=0.01,
    )
    optimized_radius = certificate_radius(
        design,
        target,
        optimized,
        theta_radius=2.0,
        design_error_bound=0.02,
        noise_radius=0.05,
        baseline_support_radius=0.01,
        approximation_support_radius=0.01,
    )
    current_radius = certificate_radius(
        design,
        target,
        least_squares,
        theta_radius=2.0,
        design_error_bound=0.02,
        noise_radius=0.05,
        baseline_support_radius=0.01,
        approximation_support_radius=0.01,
    )

    assert optimized_radius <= current_radius + 1e-9
    assert diagnostics["primal_radius"] == pytest.approx(optimized_radius)
    assert diagnostics["duality_gap"] >= -1e-9
    assert diagnostics["duality_gap"] <= 1e-5


def test_missing_global_ledger_blocks_conditional_point_upgrade():
    certificate = TargetCertificate(
        source="overlap",
        mode=CertificationMode.ASSUMPTION_CONDITIONAL,
        scope=CertificateScope.ASSUMPTION_CONDITIONAL,
        valid=True,
        feasible_set=TargetInterval(0.99, 1.01),
        claim_type=CertificateClaimType.ROBUST_TARGET_BOUND,
        budget_components=("outcome_noise", "design_error"),
        failure_semantics=FailureSemantics.PROBABILITY_LEDGER,
        target_id="unit-test-target",
        provenance="unit-test bound",
        proof=_bound_proof(0.99, 1.01, "unit-test bound"),
    )

    decision = decide_target(
        [certificate],
        scientific_tolerance=0.02,
    )

    assert decision.status is DecisionStatus.INCONCLUSIVE
    assert decision.point_estimate is None
    assert "global failure ledger" in " ".join(decision.reasons)


def test_complete_global_ledger_allows_conditional_point_with_joint_delta():
    certificate = TargetCertificate(
        source="overlap",
        mode=CertificationMode.ASSUMPTION_CONDITIONAL,
        scope=CertificateScope.ASSUMPTION_CONDITIONAL,
        valid=True,
        feasible_set=TargetInterval(0.99, 1.01),
        claim_type=CertificateClaimType.ROBUST_TARGET_BOUND,
        budget_components=("outcome_noise", "design_error"),
        failure_allocations=(("outcome_noise", 0.02), ("design_error", 0.02)),
        failure_semantics=FailureSemantics.PROBABILITY_LEDGER,
        target_id="unit-test-target",
        provenance="unit-test bound",
        proof=_bound_proof(
            0.99,
            1.01,
            "unit-test bound",
            failure_allocations=(("outcome_noise", 0.02), ("design_error", 0.02)),
        ),
    )
    ledger = FailureBudgetLedger(
        total_delta=0.05,
        allocations=(
            FailureBudget("outcome_noise", 0.02, "calibration split"),
            FailureBudget("design_error", 0.02, "calibration split"),
        ),
    )

    decision = decide_target(
        [certificate],
        scientific_tolerance=0.02,
        failure_ledger=ledger,
    )

    assert decision.status is DecisionStatus.POINT_ESTIMABLE
    assert decision.joint_failure_probability == pytest.approx(0.04)


def test_ledger_allocation_smaller_than_certificate_claim_blocks_public_output():
    certificate = TargetCertificate(
        source="overlap",
        mode=CertificationMode.ASSUMPTION_CONDITIONAL,
        scope=CertificateScope.ASSUMPTION_CONDITIONAL,
        valid=True,
        feasible_set=TargetInterval(0.99, 1.01),
        claim_type=CertificateClaimType.ROBUST_TARGET_BOUND,
        budget_components=("outcome_noise",),
        failure_allocations=(("outcome_noise", 0.02),),
        failure_semantics=FailureSemantics.PROBABILITY_LEDGER,
        target_id="unit-test-target",
        provenance="unit-test bound",
        proof=_bound_proof(
            0.99,
            1.01,
            "unit-test bound",
            failure_allocations=(("outcome_noise", 0.02),),
        ),
    )
    understated = FailureBudgetLedger(
        total_delta=0.05,
        allocations=(FailureBudget("outcome_noise", 0.001, "calibration split"),),
    )

    decision = decide_target(
        [certificate],
        scientific_tolerance=0.02,
        failure_ledger=understated,
    )

    assert decision.status is DecisionStatus.INCONCLUSIVE
    assert "delta" in " ".join(decision.reasons)


def test_multi_certificate_composition_rejects_mismatched_target_or_units():
    first = TargetCertificate(
        source="first",
        mode=CertificationMode.DESIGN_KNOWN,
        scope=CertificateScope.POPULATION_EXACT,
        valid=True,
        feasible_set=TargetInterval(0.9, 1.1),
        claim_type=CertificateClaimType.STRUCTURAL_EVIDENCE,
        target_id="target-A",
        proof=_bound_proof(0.9, 1.1, "first", target_id="target-A"),
    )
    second = TargetCertificate(
        source="second",
        mode=CertificationMode.DESIGN_KNOWN,
        scope=CertificateScope.POPULATION_EXACT,
        valid=True,
        feasible_set=TargetInterval(0.95, 1.05),
        claim_type=CertificateClaimType.STRUCTURAL_EVIDENCE,
        target_id="target-B",
        proof=_bound_proof(
            0.95, 1.05, "second", target_id="target-B", units="dollars"
        ),
    )

    decision = decide_target([first, second], scientific_tolerance=0.2)

    assert decision.status is DecisionStatus.INCONCLUSIVE
    assert "target" in " ".join(decision.reasons)


def test_multi_certificate_composition_requires_target_ids_and_common_units():
    without_ids = [
        TargetCertificate(
            source=source,
            mode=CertificationMode.DESIGN_KNOWN,
            scope=CertificateScope.POPULATION_EXACT,
            valid=True,
            feasible_set=TargetInterval(0.9, 1.1),
            claim_type=CertificateClaimType.STRUCTURAL_EVIDENCE,
            proof=_bound_proof(0.9, 1.1, source),
        )
        for source in ("first", "second")
    ]
    missing = decide_target(without_ids, scientific_tolerance=0.2)
    assert missing.status is DecisionStatus.INCONCLUSIVE
    assert "target identifier" in " ".join(missing.reasons)

    first = replace(without_ids[0], target_id="target-A")
    second = replace(
        without_ids[1],
        target_id="target-A",
        proof=_bound_proof(
            0.9, 1.1, "second", target_id="target-A", units="dollars"
        ),
    )
    incompatible = decide_target([first, second], scientific_tolerance=0.2)
    assert incompatible.status is DecisionStatus.INCONCLUSIVE
    assert "units" in " ".join(incompatible.reasons)


def test_prior_only_interval_is_inconclusive_under_positive_shrinkage_threshold():
    prior = TargetInterval(-2.0, 2.0)
    certificate = TargetCertificate(
        source="robust_overlap",
        mode=CertificationMode.ASSUMPTION_CONDITIONAL,
        scope=CertificateScope.ASSUMPTION_CONDITIONAL,
        valid=True,
        feasible_set=prior,
        claim_type=CertificateClaimType.ROBUST_TARGET_BOUND,
        target_id="unit-test-target",
        provenance="unit-test prior bound",
        proof=_bound_proof(-2.0, 2.0, "unit-test prior bound"),
    )

    decision = decide_target(
        [certificate],
        parameter_space=prior,
        scientific_tolerance=0.01,
        prior_only_shrinkage_threshold=0.01,
    )

    assert decision.status is DecisionStatus.INCONCLUSIVE
    assert decision.set_subtype is SetSubtype.PRIOR_ONLY
    assert decision.diagnostics["prior_to_certified_shrinkage"] == pytest.approx(0.0)


def test_prior_domain_can_be_a_diagnostic_without_bounding_structural_witness():
    witness = TargetCertificate(
        source="overlap",
        mode=CertificationMode.DESIGN_KNOWN,
        scope=CertificateScope.POPULATION_EXACT,
        valid=True,
        null_witness=(1.0, -1.0),
        claim_type=CertificateClaimType.EXACT_WITNESS,
        target_id="unit-test-target",
        provenance="unit-test exact operator",
        proof=_exact_null_proof(),
    )

    decision = decide_target(
        [witness],
        prior_target_domain=TargetInterval(-2.0, 2.0),
        scientific_tolerance=0.01,
    )

    assert decision.status is DecisionStatus.NONRECOVERABLE


def test_duplicate_budget_use_blocks_illegal_composition():
    certificates = [
        TargetCertificate(
            source=source,
            mode=CertificationMode.ASSUMPTION_CONDITIONAL,
            scope=CertificateScope.ASSUMPTION_CONDITIONAL,
            valid=True,
            feasible_set=TargetInterval(0.9, 1.1),
            claim_type=CertificateClaimType.ROBUST_TARGET_BOUND,
            budget_components=("shared_delta",),
            failure_allocations=(("shared_delta", 0.02),),
            failure_semantics=FailureSemantics.PROBABILITY_LEDGER,
            target_id="unit-test-target",
            provenance=f"unit-test {source} bound",
            proof=_bound_proof(
                0.9,
                1.1,
                f"unit-test {source} bound",
                failure_allocations=(("shared_delta", 0.02),),
            ),
        )
        for source in ("temporal", "overlap")
    ]
    ledger = FailureBudgetLedger(
        total_delta=0.05,
        allocations=(FailureBudget("shared_delta", 0.02, "calibration split"),),
    )

    decision = decide_target(
        certificates,
        scientific_tolerance=0.2,
        failure_ledger=ledger,
    )

    assert decision.status is DecisionStatus.INCONCLUSIVE
    assert "reused" in " ".join(decision.reasons)


def test_overlap_certifier_separates_exact_structure_from_conditional_precision():
    certificate = certify_overlap_target(
        np.eye(2),
        np.array([1.0, 2.0]),
        np.array([1.0, 0.0]),
        noise_radius=0.1,
        design_error_bound=0.0,
        theta_radius=2.0,
        exact_design=True,
        budget_components=("outcome_noise",),
        component_delta=0.02,
        provenance="held-out calibration split",
    )

    assert certificate.mode is CertificationMode.DESIGN_KNOWN
    assert certificate.scope is CertificateScope.ASSUMPTION_CONDITIONAL
    assert certificate.claim_type is CertificateClaimType.ROBUST_TARGET_BOUND
    assert certificate.budget_components == ("outcome_noise",)
    assert certificate.failure_allocations == (("outcome_noise", 0.02),)
    assert certificate.failure_semantics is FailureSemantics.PROBABILITY_LEDGER
    assert certificate.provenance == "held-out calibration split"


def test_typed_routing_rejects_numerical_witness_for_structural_nonrecoverability():
    numerical = TargetCertificate(
        source="overlap",
        mode=CertificationMode.DESIGN_KNOWN,
        scope=CertificateScope.POPULATION_EXACT,
        valid=True,
        null_witness=(1.0, -1.0),
        claim_type=CertificateClaimType.NUMERICAL_DIAGNOSTIC,
        target_id="unit-test-target",
    )

    decision = decide_target(
        [numerical],
        scientific_tolerance=0.01,
    )

    assert decision.status is DecisionStatus.INCONCLUSIVE
    assert "proof payload" in " ".join(decision.reasons)


def test_temporal_certifiers_emit_typed_structural_and_conditional_evidence():
    gap = certify_temporal_design_known(False)
    conditional = certify_temporal_conditional(
        np.full(100, 0.01),
        np.full(100, 1.0),
        bandwidth=0.1,
        n_units=1000,
        lower_mass_c=1.0,
        lower_mass_beta=1.0,
        holder_L=0.1,
        holder_alpha=1.0,
        noise_scale=0.01,
        delta=0.05,
        budget_components=("temporal_count", "trace_noise"),
        provenance="held-out trajectory split",
    )

    assert not gap.valid
    assert gap.claim_type is CertificateClaimType.NUMERICAL_DIAGNOSTIC
    assert conditional.claim_type is CertificateClaimType.ROBUST_TARGET_BOUND
    assert conditional.budget_components == ("temporal_count", "trace_noise")
    assert conditional.failure_semantics is FailureSemantics.PROBABILITY_LEDGER
