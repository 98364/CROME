from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from crome_identification.certification import (
    CertificateClaimType,
    CertificateScope,
    CertificationMode,
    ExactNullProof,
    LinearBoundProof,
    ScalarBoundProof,
    SupportGapProof,
    TemporalBoundProof,
    TemporalDesignBoundProof,
    TargetCertificate,
    TargetInterval,
    decide_target,
    certify_temporal_design_known,
    verify_proof,
)
from crome_identification.certification.types import DecisionStatus


def _exact_null_proof(**changes):
    values = {
        "operator": ((1, 1),),
        "target": (1, -1),
        "witness": (1, -1),
        "parameter_space": "unrestricted",
        "operator_id": "sha256:fixture-operator",
        "target_id": "unit-test-target",
        "units": "standardized response units",
        "provenance": "frozen structural fixture",
    }
    values.update(changes)
    return ExactNullProof(**values)


def test_exact_null_proof_is_checked_with_rational_arithmetic():
    result = verify_proof(_exact_null_proof())

    assert result.valid
    assert result.proves_nonidentification
    assert result.diagnostics["target_variation"] == Fraction(2, 1)


def test_wrong_or_infeasible_exact_witness_is_rejected():
    wrong = verify_proof(_exact_null_proof(witness=(1, 0)))
    infeasible = verify_proof(
        _exact_null_proof(
            parameter_space="box",
            base_parameter=(0, 0),
            step=1,
            parameter_lower=(Fraction(-1, 2), Fraction(-1, 2)),
            parameter_upper=(Fraction(1, 2), Fraction(1, 2)),
        )
    )

    assert not wrong.valid
    assert "residual" in wrong.reason
    assert not infeasible.valid
    assert "feasible" in infeasible.reason


def test_box_exact_null_proof_requires_two_distinct_parameter_points():
    result = verify_proof(
        _exact_null_proof(
            parameter_space="box",
            base_parameter=(0, 0),
            step=0,
            parameter_lower=(-1, -1),
            parameter_upper=(1, 1),
        )
    )

    assert not result.valid
    assert not result.proves_nonidentification
    assert "nonzero" in result.reason


def test_support_gap_proof_requires_gap_and_observational_equivalence():
    proof = SupportGapProof(
        observed_support=(Fraction(1, 2), 1, 2),
        gap=(0, Fraction(1, 3)),
        observed_signature_left=(0, 0, 0),
        observed_signature_right=(0, 0, 0),
        target_value_left=0,
        target_value_right=1,
        response_class_premise="closed under the supplied local bump",
        support_specification="auditable deterministic observation schedule",
        units="hours",
        provenance="schedule hash sha256:fixture",
        baseline_level=0,
        bump_amplitude=1,
        bump_power=2,
        schema_version="crome.support-gap.v1",
        producer_version="crome-identification/0.1.0",
        artifact_hash="sha256:" + "0" * 64,
        target_id="unit-test-target",
    )
    proof = replace(proof, artifact_hash=proof.recompute_artifact_hash())

    result = verify_proof(proof)
    bad_signature = verify_proof(
        replace(
            replace(proof, observed_signature_right=(0, 1, 0)),
            artifact_hash=replace(
                proof, observed_signature_right=(0, 1, 0)
            ).recompute_artifact_hash(),
        )
    )
    wrong_target_payload = replace(proof, bump_amplitude=2)
    wrong_target = verify_proof(
        replace(
            wrong_target_payload,
            artifact_hash=wrong_target_payload.recompute_artifact_hash(),
        )
    )

    assert result.valid
    assert result.proves_nonidentification
    assert not bad_signature.valid
    assert "observationally equivalent" in bad_signature.reason
    assert not wrong_target.valid
    assert "target" in wrong_target.reason


def test_support_gap_proof_rejects_an_internal_gap_unrelated_to_zero():
    proof = SupportGapProof(
        observed_support=(Fraction(1, 10), Fraction(5, 2)),
        gap=(1, 2),
        observed_signature_left=(0, 0),
        observed_signature_right=(0, 0),
        target_value_left=0,
        target_value_right=1,
        response_class_premise="closed under the supplied local bump",
        support_specification="auditable deterministic observation schedule",
        units="hours",
        provenance="schedule hash sha256:internal-gap-fixture",
        baseline_level=0,
        bump_amplitude=1,
        bump_power=2,
        schema_version="crome.support-gap.v1",
        producer_version="crome-identification/0.1.0",
        artifact_hash="sha256:" + "0" * 64,
        target_id="unit-test-target",
    )
    proof = replace(proof, artifact_hash=proof.recompute_artifact_hash())

    result = verify_proof(proof)

    assert not result.valid
    assert not result.proves_nonidentification
    assert "zero-anchored" in result.reason


def test_support_gap_proof_requires_strictly_positive_observed_lags():
    proof = SupportGapProof(
        observed_support=(-1, 1),
        gap=(0, Fraction(1, 2)),
        observed_signature_left=(0, 0),
        observed_signature_right=(0, 0),
        target_value_left=0,
        target_value_right=1,
        response_class_premise="closed under the supplied local bump",
        support_specification="auditable deterministic observation schedule",
        units="hours",
        provenance="schedule hash sha256:nonpositive-lag-fixture",
        baseline_level=0,
        bump_amplitude=1,
        bump_power=2,
        schema_version="crome.support-gap.v1",
        producer_version="crome-identification/0.1.0",
        artifact_hash="sha256:" + "0" * 64,
        target_id="unit-test-target",
    )
    proof = replace(proof, artifact_hash=proof.recompute_artifact_hash())

    result = verify_proof(proof)

    assert not result.valid
    assert "positive lags" in result.reason


def test_linear_bound_radius_is_recomputed_from_payload():
    proof = LinearBoundProof(
        operator=((1, 0), (0, 1)),
        outcome=(1, 2),
        target=(1, 0),
        weights=(1, 0),
        theta_radius=3,
        design_error_bound=0,
        noise_radius=Fraction(1, 10),
        baseline_support_radius=0,
        approximation_support_radius=0,
        additive_baseline_error=Fraction(1, 5),
        additive_approximation_error=0,
        center=1,
        claimed_radius=Fraction(3, 10),
        units="response units",
        uncertainty_definition="Euclidean rectangular uncertainty set",
        provenance="held-out calibration split",
        schema_version="crome.linear-bound.v1",
        producer_version="crome-identification/0.1.0",
        artifact_hash="sha256:" + "0" * 64,
        failure_allocations=(),
        target_id="unit-test-target",
    )
    proof = LinearBoundProof(
        **{**proof.__dict__, "artifact_hash": proof.recompute_artifact_hash()}
    )

    result = verify_proof(proof)
    tampered = verify_proof(
        LinearBoundProof(**{**proof.__dict__, "claimed_radius": Fraction(1, 10)})
    )

    assert result.valid
    assert result.recomputed_interval == TargetInterval(0.7, 1.3)
    assert not tampered.valid
    assert "artifact hash" in tampered.reason


def test_linear_bound_with_design_error_does_not_prove_structural_identification():
    proof = LinearBoundProof(
        operator=((1,),),
        outcome=(0,),
        target=(1,),
        weights=(1,),
        theta_radius=1,
        design_error_bound=Fraction(1, 10),
        noise_radius=0,
        baseline_support_radius=0,
        approximation_support_radius=0,
        additive_baseline_error=0,
        additive_approximation_error=0,
        center=0,
        claimed_radius=Fraction(1, 10),
        units="response units",
        uncertainty_definition="estimated design with an operator-norm error bound",
        provenance="unit-test estimated design",
        schema_version="crome.linear-bound.v1",
        producer_version="crome-identification/0.1.0",
        artifact_hash="sha256:" + "0" * 64,
        failure_allocations=(),
        target_id="unit-test-target",
    )
    proof = replace(proof, artifact_hash=proof.recompute_artifact_hash())

    result = verify_proof(proof)

    assert result.valid
    assert result.recomputed_interval == TargetInterval(-0.1, 0.1)
    assert not result.proves_identification


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("center", 999),
        ("outcome", (999, 2)),
        ("weights", (0, 1)),
    ],
)
def test_linear_bound_rejects_center_or_sufficient_statistic_mutation(field, replacement):
    proof = LinearBoundProof(
        operator=((1, 0), (0, 1)),
        outcome=(1, 2),
        target=(1, 0),
        weights=(1, 0),
        theta_radius=3,
        design_error_bound=0,
        noise_radius=Fraction(1, 10),
        baseline_support_radius=0,
        approximation_support_radius=0,
        additive_baseline_error=Fraction(1, 5),
        additive_approximation_error=0,
        center=1,
        claimed_radius=Fraction(3, 10),
        units="response units",
        uncertainty_definition="Euclidean rectangular uncertainty set",
        provenance="held-out calibration split",
        schema_version="crome.linear-bound.v1",
        producer_version="crome-identification/0.1.0",
        artifact_hash="sha256:" + "0" * 64,
        failure_allocations=(),
        target_id="unit-test-target",
    )
    proof = LinearBoundProof(
        **{**proof.__dict__, "artifact_hash": proof.recompute_artifact_hash()}
    )

    result = verify_proof(LinearBoundProof(**{**proof.__dict__, field: replacement}))

    assert not result.valid
    assert "artifact hash" in result.reason


def test_generic_scalar_bound_is_diagnostic_only_and_fails_closed():
    proof = ScalarBoundProof(
        center=1,
        radius_components=(
            ("bias", Fraction(1, 10)),
            ("noise", Fraction(1, 20)),
        ),
        claimed_radius=Fraction(3, 20),
        units="response units",
        premise="simultaneous boundary-error event",
        provenance="held-out trace calibration split",
    )

    result = verify_proof(proof)

    assert not result.valid
    assert result.recomputed_interval is None
    assert "data-bound typed proof" in result.reason


def _temporal_bound_proof(**changes):
    values = {
        "lags": ("0.05",) * 100 + ("0.20",),
        "responses": ("1.01",) * 100 + ("9.0",),
        "bandwidth": "0.1",
        "n_units": 1000,
        "lower_mass_c": "1.0",
        "lower_mass_beta": "1.0",
        "holder_L": "1.0",
        "holder_alpha": "1.0",
        "noise_scale": "0.05",
        "noise_failure_budget": "0.025",
        "baseline_error": "0.01",
        "declared_failure_budget": "0.05",
        "claimed_center": "1.01",
        "claimed_radius": "0.12480207187300799",
        "failure_allocations": (
            ("temporal_count", "0.025"),
            ("trace_noise", "0.025"),
        ),
        "schema_version": "crome.temporal-bound.v1",
        "producer_version": "crome-identification/0.1.0",
        "artifact_hash": "",
        "units": "response units",
        "premise": "trajectory-level lower-mass and boundary-noise event",
        "provenance": "held-out trajectory fixture",
        "target_id": "unit-test-target",
    }
    values.update(changes)
    proof = TemporalBoundProof(**values)
    if not proof.artifact_hash:
        proof = TemporalBoundProof(
            **{
                **proof.__dict__,
                "artifact_hash": proof.recompute_artifact_hash(),
            }
        )
    return proof


def test_temporal_bound_recomputes_count_center_radius_and_failure_budget():
    result = verify_proof(_temporal_bound_proof())

    assert result.valid
    assert result.recomputed_interval.lower == pytest.approx(0.885197928126992)
    assert result.recomputed_interval.upper == pytest.approx(1.134802071873008)
    assert result.diagnostics["n_near_zero"] == 100
    assert result.diagnostics["required_near_zero_count"] == pytest.approx(50.0)
    assert result.required_failure_allocations == pytest.approx(
        {"temporal_count": 0.025, "trace_noise": 0.025}
    )


def test_design_known_temporal_bound_recomputes_selected_mean_and_binds_delta():
    certificate = certify_temporal_design_known(
        True,
        lags=np.array([0.05, 0.2]),
        responses=np.array([1.0, 9.0]),
        bandwidth=0.1,
        holder_L=1.0,
        holder_alpha=1.0,
        response_error_bound=0.2,
        budget_components=("trace_noise",),
        component_delta=0.02,
        provenance="held-out selected-mean fixture",
    )
    proof = certificate.proof

    assert isinstance(proof, TemporalDesignBoundProof)
    result = verify_proof(proof)
    assert result.valid
    assert result.recomputed_interval == TargetInterval(0.7, 1.3)
    assert result.required_failure_allocations == pytest.approx(
        {"trace_noise": 0.02}
    )

    tampered = replace(proof, responses=("2.0", "9.0"))
    assert not verify_proof(tampered).valid
    assert "artifact hash" in verify_proof(tampered).reason


@pytest.mark.parametrize(
    "field,replacement,reason",
    [
        ("responses", ("1.0", "1.2", "9.0"), "artifact hash"),
        ("bandwidth", "0.02", "artifact hash"),
        ("claimed_center", "1.02", "artifact hash"),
        ("claimed_radius", "0.1", "artifact hash"),
        ("schema_version", "crome.temporal-bound.v0", "schema"),
        ("producer_version", "", "producer"),
        ("artifact_hash", "sha256:" + "0" * 64, "artifact hash"),
    ],
)
def test_temporal_bound_one_field_mutations_are_rejected(field, replacement, reason):
    proof = _temporal_bound_proof()
    mutated = TemporalBoundProof(**{**proof.__dict__, field: replacement})

    result = verify_proof(mutated)

    assert not result.valid
    assert reason in result.reason


def test_temporal_bound_rejects_a_self_consistent_payload_without_enough_near_zero_rows():
    proof = _temporal_bound_proof(
        lags=("0.2", "0.3"),
        responses=("1.0", "1.0"),
        claimed_center="1.0",
    )

    result = verify_proof(proof)

    assert not result.valid
    assert "near-zero" in result.reason


def test_temporal_bound_rejects_more_rows_than_declared_independent_units():
    proof = _temporal_bound_proof(n_units=100)

    result = verify_proof(proof)

    assert not result.valid
    assert "cannot exceed n_units" in result.reason


def test_forged_valid_label_without_proof_cannot_trigger_nonrecoverable():
    forged = TargetCertificate(
        source="forged",
        mode=CertificationMode.DESIGN_KNOWN,
        scope=CertificateScope.POPULATION_EXACT,
        valid=True,
        null_witness=(1.0, -1.0),
        claim_type=CertificateClaimType.EXACT_WITNESS,
        target_id="unit-test-target",
        provenance="caller asserted valid",
    )

    decision = decide_target([forged], scientific_tolerance=0.01)

    assert decision.status is DecisionStatus.INCONCLUSIVE
    assert decision.point_estimate is None
    assert "proof payload" in " ".join(decision.reasons)
