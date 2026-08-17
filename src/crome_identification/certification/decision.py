"""Unified target-feasible-set assembly and four-way decision logic."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
import math
from typing import Any

from .types import (
    CertificateClaimType,
    CertificateScope,
    CertificationMode,
    CromeDecision,
    DecisionStatus,
    FailureBudgetLedger,
    FailureSemantics,
    ExactNullProof,
    LinearBoundProof,
    OperationalStatus,
    SetSubtype,
    StructuralStatus,
    TargetCertificate,
    TargetInterval,
)
from .verification import verify_proof


class ModelInfeasibleError(ValueError):
    """Raised when valid declared constraints have an empty intersection."""


_SCOPE_STRENGTH = {
    CertificateScope.FINITE_SAMPLE_ONLY: 0,
    CertificateScope.ASSUMPTION_CONDITIONAL: 1,
    CertificateScope.POPULATION_EXACT: 2,
}


def _weakest_scope(certificates: Sequence[TargetCertificate]) -> CertificateScope:
    return min(certificates, key=lambda cert: _SCOPE_STRENGTH[cert.scope]).scope


def _deduplicate(items: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for item in items if item))


def _inconclusive(
    certificates: Sequence[TargetCertificate],
    feasible_set: TargetInterval,
    reasons: Sequence[str],
    *,
    diagnostics: dict[str, Any] | None = None,
    set_subtype: SetSubtype | None = None,
    joint_failure_probability: float | None = None,
) -> CromeDecision:
    return CromeDecision(
        status=DecisionStatus.INCONCLUSIVE,
        certificate_scope=_weakest_scope(certificates),
        feasible_set=feasible_set,
        point_estimate=None,
        uncertainty=None,
        assumptions=_deduplicate(
            [assumption for cert in certificates for assumption in cert.assumptions]
        ),
        reasons=_deduplicate(list(reasons)),
        diagnostics={
            "certificate_sources": [cert.source for cert in certificates],
            **(diagnostics or {}),
        },
        set_subtype=set_subtype,
        joint_failure_probability=joint_failure_probability,
        numerical_slack=max((cert.numerical_slack for cert in certificates), default=0.0),
        structural_status=StructuralStatus.UNKNOWN,
        operational_status=OperationalStatus.INCONCLUSIVE,
    )


def _ledger_failure_reason(
    certificates: Sequence[TargetCertificate],
    ledger: FailureBudgetLedger | None,
) -> str | None:
    conditional = [
        cert
        for cert in certificates
        if cert.scope is CertificateScope.ASSUMPTION_CONDITIONAL
        or cert.mode is CertificationMode.ASSUMPTION_CONDITIONAL
    ]
    if not conditional:
        return None
    stochastic = []
    for cert in conditional:
        if cert.feasible_set is None and cert.null_witness is None:
            continue
        required_by_proof = False
        if cert.proof is not None:
            required_by_proof = bool(verify_proof(cert.proof).required_failure_allocations)
        if (
            cert.failure_semantics is FailureSemantics.PROBABILITY_LEDGER
            or cert.budget_components
            or cert.failure_allocations
            or required_by_proof
        ):
            stochastic.append(cert)
    if not stochastic:
        return None
    if ledger is None:
        return "conditional evidence is missing a global failure ledger"
    for cert in stochastic:
        if not cert.budget_components or not cert.failure_allocations:
            return f"{cert.source} does not bind its failure components to delta allocations"
        names = tuple(name for name, _ in cert.failure_allocations)
        if names != cert.budget_components:
            return f"{cert.source} failure-allocation names do not match its budget components"
        if any(
            not name.strip() or not math.isfinite(delta) or not 0.0 < delta < 1.0
            for name, delta in cert.failure_allocations
        ):
            return f"{cert.source} carries an invalid failure delta allocation"
    claimed = [component for cert in stochastic for component in cert.budget_components]
    if len(claimed) != len(set(claimed)):
        return "a failure-budget component was reused across certificates"
    missing = sorted(set(claimed) - ledger.components)
    if missing:
        return f"global failure ledger is missing components: {', '.join(missing)}"
    ledger_deltas = {
        allocation.component: allocation.delta for allocation in ledger.allocations
    }
    for cert in stochastic:
        for component, delta in cert.failure_allocations:
            if not math.isclose(ledger_deltas[component], delta, rel_tol=1e-12, abs_tol=1e-15):
                return (
                    f"global failure ledger delta for {component} does not match "
                    f"the {cert.source} certificate allocation"
                )
        if cert.proof is not None:
            verification = verify_proof(cert.proof)
            required = dict(verification.required_failure_allocations)
            declared = dict(cert.failure_allocations)
            if required and (
                not required.keys() <= declared.keys()
                or any(
                    not math.isclose(required[name], declared[name], rel_tol=1e-12, abs_tol=1e-15)
                    for name in required
                )
            ):
                return f"{cert.source} proof and certificate delta allocations do not match"
    return None


def _composition_identity_failure_reason(
    certificates: Sequence[TargetCertificate],
) -> str | None:
    public = [
        cert
        for cert in certificates
        if cert.feasible_set is not None or cert.null_witness is not None
    ]
    if not public:
        return None
    if any(not cert.target_id.strip() for cert in public):
        return "every public certificate must bind a target identifier"
    target_ids = {cert.target_id for cert in public if cert.target_id.strip()}
    if len(target_ids) > 1:
        return "certificates refer to different target identifiers"
    vector_targets: list[tuple[Fraction, ...]] = []
    for cert in public:
        if isinstance(cert.proof, (LinearBoundProof, ExactNullProof)):
            try:
                vector_targets.append(tuple(Fraction(value) for value in cert.proof.target))
            except (TypeError, ValueError, ZeroDivisionError):
                continue
    if len(set(vector_targets)) > 1:
        return "certificates bind different linear target functionals"
    unit_values = [
        str(getattr(cert.proof, "units", "")).strip()
        for cert in public
        if cert.proof is not None
    ]
    if any(not units for units in unit_values):
        return "every public proof must bind scalar-target units"
    units = set(unit_values)
    if len(units) > 1:
        return "certificates use incompatible scalar-target units"
    return None


def _proof_contract_failure_reason(
    certificates: Sequence[TargetCertificate],
) -> tuple[str | None, dict[str, Any], StructuralStatus]:
    verification_diagnostics: dict[str, Any] = {}
    proves_identification = False
    proves_nonidentification = False
    for cert in certificates:
        carries_public_claim = (
            cert.feasible_set is not None
            or cert.null_witness is not None
            or cert.proof is not None
        )
        if not carries_public_claim:
            continue
        if cert.proof is None:
            return (
                f"{cert.source} carries a public claim without a verifiable proof payload",
                verification_diagnostics,
                StructuralStatus.UNKNOWN,
            )
        verification = verify_proof(cert.proof)
        verification_diagnostics[cert.source] = {
            "valid": verification.valid,
            "proof_type": verification.proof_type,
            "reason": verification.reason,
            "proves_identification": verification.proves_identification,
            "proves_nonidentification": verification.proves_nonidentification,
        }
        if not verification.valid:
            return (
                f"{cert.source} proof verification failed: {verification.reason}",
                verification_diagnostics,
                StructuralStatus.UNKNOWN,
            )
        if (
            cert.scope is CertificateScope.POPULATION_EXACT
            and cert.mode is not CertificationMode.DESIGN_KNOWN
        ):
            return (
                f"{cert.source} population-exact scope is incompatible with "
                f"{cert.mode.value} mode",
                verification_diagnostics,
                StructuralStatus.UNKNOWN,
            )
        if (
            cert.scope is CertificateScope.POPULATION_EXACT
            and isinstance(cert.proof, LinearBoundProof)
        ):
            uncertainty_budgets = (
                cert.proof.design_error_bound,
                cert.proof.noise_radius,
                cert.proof.baseline_support_radius,
                cert.proof.approximation_support_radius,
                cert.proof.additive_baseline_error,
                cert.proof.additive_approximation_error,
            )
            if any(Fraction(value) != 0 for value in uncertainty_budgets):
                return (
                    f"{cert.source} population-exact linear proof cannot carry "
                    "design error or other nonzero uncertainty budgets",
                    verification_diagnostics,
                    StructuralStatus.UNKNOWN,
                )
        proof_target_id = str(getattr(cert.proof, "target_id", "")).strip()
        if not proof_target_id or proof_target_id != cert.target_id:
            return (
                f"{cert.source} certificate target id does not match its proof payload",
                verification_diagnostics,
                StructuralStatus.UNKNOWN,
            )
        if cert.feasible_set is not None:
            if verification.recomputed_interval is None:
                return (
                    f"{cert.source} proof payload does not recompute its target interval",
                    verification_diagnostics,
                    StructuralStatus.UNKNOWN,
                )
            scale = max(
                1.0,
                abs(cert.feasible_set.lower),
                abs(cert.feasible_set.upper),
            )
            if (
                abs(cert.feasible_set.lower - verification.recomputed_interval.lower)
                > 1e-12 * scale
                or abs(cert.feasible_set.upper - verification.recomputed_interval.upper)
                > 1e-12 * scale
            ):
                return (
                    f"{cert.source} interval does not match the independently recomputed bound",
                    verification_diagnostics,
                    StructuralStatus.UNKNOWN,
                )
        if cert.null_witness is not None and not verification.proves_nonidentification:
            return (
                f"{cert.source} proof payload does not establish target nonidentification",
                verification_diagnostics,
                StructuralStatus.UNKNOWN,
            )
        proves_identification = proves_identification or (
            verification.proves_identification
            and cert.scope is CertificateScope.POPULATION_EXACT
        )
        proves_nonidentification = (
            proves_nonidentification or verification.proves_nonidentification
        )
        if cert.null_witness is not None and (
            cert.scope is not CertificateScope.POPULATION_EXACT
            or cert.claim_type
            not in {
                CertificateClaimType.EXACT_WITNESS,
                CertificateClaimType.STRUCTURAL_EVIDENCE,
            }
        ):
            return (
                f"typed certificate contract forbids {cert.source} from issuing "
                "structural nonrecoverability from numerical evidence",
                verification_diagnostics,
                StructuralStatus.UNKNOWN,
            )
        if cert.feasible_set is not None and cert.claim_type in {
            CertificateClaimType.FINITE_SAMPLE_DIAGNOSTIC,
            CertificateClaimType.NUMERICAL_DIAGNOSTIC,
        }:
            return (
                f"typed certificate contract forbids {cert.source} diagnostic "
                "from upgrading a public claim",
                verification_diagnostics,
                StructuralStatus.UNKNOWN,
            )
    structural_status = (
        StructuralStatus.NONIDENTIFIED
        if proves_nonidentification
        else StructuralStatus.IDENTIFIED
        if proves_identification
        else StructuralStatus.UNKNOWN
    )
    return None, verification_diagnostics, structural_status


def decide_target(
    certificates: Sequence[TargetCertificate],
    *,
    parameter_space: TargetInterval | None = None,
    scientific_tolerance: float,
    failure_ledger: FailureBudgetLedger | None = None,
    prior_only_shrinkage_threshold: float = 0.0,
    prior_target_domain: TargetInterval | None = None,
) -> CromeDecision:
    """Combine all target constraints before issuing one mutually exclusive status."""

    if not certificates:
        raise ValueError("at least one target certificate is required")
    if not math.isfinite(scientific_tolerance) or scientific_tolerance < 0:
        raise ValueError("scientific_tolerance must be finite and non-negative")
    if (
        not math.isfinite(prior_only_shrinkage_threshold)
        or not 0.0 <= prior_only_shrinkage_threshold <= 1.0
    ):
        raise ValueError("prior_only_shrinkage_threshold must lie in [0, 1]")

    feasible_set = parameter_space or TargetInterval.unbounded()
    invalid_reasons = [
        cert.reason or f"{cert.source} certificate is invalid"
        for cert in certificates
        if not cert.valid
    ]
    if invalid_reasons:
        return _inconclusive(certificates, feasible_set, invalid_reasons)

    empirical_sources = [
        cert.source for cert in certificates if cert.mode is CertificationMode.EMPIRICAL_ONLY
    ]
    if empirical_sources:
        return _inconclusive(
            certificates,
            feasible_set,
            [f"empirical-only evidence cannot issue a population certificate: {source}"
             for source in empirical_sources],
        )

    identity_reason = _composition_identity_failure_reason(certificates)
    if identity_reason:
        return _inconclusive(certificates, feasible_set, [identity_reason])

    typed_reason, proof_diagnostics, structural_status = _proof_contract_failure_reason(
        certificates
    )
    if typed_reason:
        return _inconclusive(
            certificates,
            feasible_set,
            [typed_reason],
            diagnostics={"proof_verification": proof_diagnostics},
        )

    joint_failure_probability = failure_ledger.spent_delta if failure_ledger else None
    ledger_reason = _ledger_failure_reason(certificates, failure_ledger)
    if ledger_reason:
        return _inconclusive(
            certificates,
            feasible_set,
            [ledger_reason],
            diagnostics={"proof_verification": proof_diagnostics},
            joint_failure_probability=joint_failure_probability,
        )

    constrained_sources: list[str] = []
    for cert in certificates:
        if cert.feasible_set is None:
            continue
        intersection = feasible_set.intersect(cert.feasible_set)
        if intersection is None:
            raise ModelInfeasibleError(
                f"empty target feasible set after applying {cert.source} constraint"
            )
        feasible_set = intersection
        constrained_sources.append(cert.source)

    scope = _weakest_scope(certificates)
    assumptions = _deduplicate(
        [assumption for cert in certificates for assumption in cert.assumptions]
    )
    reasons = _deduplicate([cert.reason for cert in certificates])
    diagnostics: dict[str, Any] = {
        "certificate_sources": [cert.source for cert in certificates],
        "constrained_sources": constrained_sources,
        "null_witness_sources": [
            cert.source for cert in certificates if cert.null_witness is not None
        ],
        "proof_verification": proof_diagnostics,
        "legacy_projection_rule": (
            "NONIDENTIFIED -> NONRECOVERABLE; otherwise POINT_AT_TAU -> POINT, "
            "SET -> SET, and INCONCLUSIVE -> INCONCLUSIVE"
        ),
    }

    prior_reference = prior_target_domain or parameter_space
    if prior_reference is not None and prior_reference.is_bounded and feasible_set.is_bounded:
        prior_width = prior_reference.width
        shrinkage = (
            1.0
            if prior_width == 0.0 and feasible_set.width == 0.0
            else 1.0 - feasible_set.width / prior_width
        )
        shrinkage = float(min(1.0, max(0.0, shrinkage)))
        diagnostics["prior_to_certified_shrinkage"] = shrinkage
        if shrinkage < prior_only_shrinkage_threshold:
            return _inconclusive(
                certificates,
                feasible_set,
                ["certified set does not improve on the prior target domain"],
                diagnostics=diagnostics,
                set_subtype=SetSubtype.PRIOR_ONLY,
                joint_failure_probability=joint_failure_probability,
            )

    operational_status = OperationalStatus.INCONCLUSIVE
    if feasible_set.is_bounded:
        scale = max(1.0, abs(feasible_set.lower), abs(feasible_set.upper))
        rounding_slack = 1e-12 * scale
        if feasible_set.width <= 2.0 * scientific_tolerance + rounding_slack:
            operational_status = OperationalStatus.POINT_AT_TAU
            if structural_status is StructuralStatus.NONIDENTIFIED:
                return CromeDecision(
                    status=DecisionStatus.NONRECOVERABLE,
                    certificate_scope=scope,
                    feasible_set=feasible_set,
                    point_estimate=None,
                    uncertainty=feasible_set,
                    assumptions=assumptions,
                    reasons=reasons,
                    diagnostics=diagnostics,
                    joint_failure_probability=joint_failure_probability,
                    numerical_slack=max(
                        rounding_slack,
                        max((cert.numerical_slack for cert in certificates), default=0.0),
                    ),
                    structural_status=structural_status,
                    operational_status=operational_status,
                )
            return CromeDecision(
                status=DecisionStatus.POINT_ESTIMABLE,
                certificate_scope=scope,
                feasible_set=feasible_set,
                point_estimate=feasible_set.midpoint,
                uncertainty=feasible_set,
                assumptions=assumptions,
                reasons=reasons,
                diagnostics=diagnostics,
                joint_failure_probability=joint_failure_probability,
                numerical_slack=max(
                    rounding_slack,
                    max((cert.numerical_slack for cert in certificates), default=0.0),
                ),
                structural_status=structural_status,
                operational_status=operational_status,
            )
        operational_status = OperationalStatus.SET
        declared_subtypes = [cert.set_subtype for cert in certificates if cert.set_subtype]
        set_subtype = declared_subtypes[-1] if declared_subtypes else (
            SetSubtype.STRUCTURAL_IDENTIFIED
            if scope is CertificateScope.POPULATION_EXACT
            else SetSubtype.ASSUMPTION_CONDITIONAL
        )
        legacy_status = (
            DecisionStatus.NONRECOVERABLE
            if structural_status is StructuralStatus.NONIDENTIFIED
            else DecisionStatus.SET_ESTIMABLE
        )
        return CromeDecision(
            status=legacy_status,
            certificate_scope=scope,
            feasible_set=feasible_set,
            point_estimate=None,
            uncertainty=feasible_set,
            assumptions=assumptions,
            reasons=reasons,
            diagnostics=diagnostics,
            set_subtype=set_subtype,
            joint_failure_probability=joint_failure_probability,
            numerical_slack=max(
                rounding_slack,
                max((cert.numerical_slack for cert in certificates), default=0.0),
            ),
            structural_status=structural_status,
            operational_status=operational_status,
        )

    if structural_status is StructuralStatus.NONIDENTIFIED:
        return CromeDecision(
            status=DecisionStatus.NONRECOVERABLE,
            certificate_scope=scope,
            feasible_set=feasible_set,
            point_estimate=None,
            uncertainty=None,
            assumptions=assumptions,
            reasons=reasons,
            diagnostics=diagnostics,
            joint_failure_probability=joint_failure_probability,
            numerical_slack=max(
                (cert.numerical_slack for cert in certificates), default=0.0
            ),
            structural_status=structural_status,
            operational_status=OperationalStatus.INCONCLUSIVE,
        )

    return _inconclusive(
        certificates,
        feasible_set,
        [*reasons, "no finite target bound or exact failure witness is available"],
        joint_failure_probability=joint_failure_probability,
    )
