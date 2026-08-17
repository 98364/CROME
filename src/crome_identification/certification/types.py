"""Typed outputs for identifiability-aware target certification."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from fractions import Fraction
from typing import Any, Mapping, TypeAlias


RationalInput: TypeAlias = int | str | Fraction


class DecisionStatus(str, Enum):
    POINT_ESTIMABLE = "POINT_ESTIMABLE"
    SET_ESTIMABLE = "SET_ESTIMABLE"
    NONRECOVERABLE = "NONRECOVERABLE"
    INCONCLUSIVE = "INCONCLUSIVE"


class StructuralStatus(str, Enum):
    """Population-level identification conclusion carried by verified evidence."""

    IDENTIFIED = "IDENTIFIED"
    NONIDENTIFIED = "NONIDENTIFIED"
    UNKNOWN = "UNKNOWN"


class OperationalStatus(str, Enum):
    """Tolerance-qualified action based on the composed target set."""

    POINT_AT_TAU = "POINT_AT_TAU"
    SET = "SET"
    INCONCLUSIVE = "INCONCLUSIVE"


class CertificateScope(str, Enum):
    POPULATION_EXACT = "population_exact"
    ASSUMPTION_CONDITIONAL = "assumption_conditional"
    FINITE_SAMPLE_ONLY = "finite_sample_only"


class CertificationMode(str, Enum):
    DESIGN_KNOWN = "design_known"
    ASSUMPTION_CONDITIONAL = "assumption_conditional"
    EMPIRICAL_ONLY = "empirical_only"


class FailureSemantics(str, Enum):
    """Whether a public certificate consumes a joint failure-probability budget."""

    DETERMINISTIC_BOUND = "deterministic_bound"
    PROBABILITY_LEDGER = "probability_ledger"


class CertificateClaimType(str, Enum):
    """Semantic class of evidence carried by a certificate."""

    STRUCTURAL_EVIDENCE = "structural_evidence"
    ROBUST_TARGET_BOUND = "robust_target_bound"
    EXACT_WITNESS = "exact_witness"
    FINITE_SAMPLE_DIAGNOSTIC = "finite_sample_diagnostic"
    NUMERICAL_DIAGNOSTIC = "numerical_diagnostic"


class SetSubtype(str, Enum):
    """Meaning of a bounded non-point output."""

    STRUCTURAL_IDENTIFIED = "structural_identified_set"
    CONFIDENCE_OUTER = "confidence_outer_set"
    ASSUMPTION_CONDITIONAL = "assumption_conditional_set"
    PRIOR_ONLY = "prior_only"


@dataclass(frozen=True)
class ExactNullProof:
    """Exact rational proof that a feasible direction changes the target in Null(A)."""

    operator: tuple[tuple[RationalInput, ...], ...]
    target: tuple[RationalInput, ...]
    witness: tuple[RationalInput, ...]
    parameter_space: str
    operator_id: str
    target_id: str
    units: str
    provenance: str
    base_parameter: tuple[RationalInput, ...] | None = None
    step: RationalInput | None = None
    parameter_lower: tuple[RationalInput, ...] | None = None
    parameter_upper: tuple[RationalInput, ...] | None = None


@dataclass(frozen=True)
class SupportGapProof:
    """Auditable support-gap equivalence construction for a scalar target."""

    observed_support: tuple[RationalInput, ...]
    gap: tuple[RationalInput, RationalInput]
    observed_signature_left: tuple[RationalInput, ...]
    observed_signature_right: tuple[RationalInput, ...]
    target_value_left: RationalInput
    target_value_right: RationalInput
    response_class_premise: str
    support_specification: str
    units: str
    provenance: str
    baseline_level: RationalInput
    bump_amplitude: RationalInput
    bump_power: int
    schema_version: str
    producer_version: str
    artifact_hash: str
    target_id: str

    def recompute_artifact_hash(self) -> str:
        """Return the digest binding the supported local-bump proof grammar."""

        from .verification import support_gap_artifact_hash

        return support_gap_artifact_hash(self)


@dataclass(frozen=True)
class LinearBoundProof:
    """Payload from which a fixed-weight linear-certificate radius is recomputed."""

    operator: tuple[tuple[RationalInput, ...], ...]
    outcome: tuple[RationalInput, ...]
    target: tuple[RationalInput, ...]
    weights: tuple[RationalInput, ...]
    theta_radius: RationalInput
    design_error_bound: RationalInput
    noise_radius: RationalInput
    baseline_support_radius: RationalInput
    approximation_support_radius: RationalInput
    additive_baseline_error: RationalInput
    additive_approximation_error: RationalInput
    center: RationalInput
    claimed_radius: RationalInput
    units: str
    uncertainty_definition: str
    provenance: str
    schema_version: str
    producer_version: str
    artifact_hash: str
    failure_allocations: tuple[tuple[str, RationalInput], ...]
    target_id: str

    def recompute_artifact_hash(self) -> str:
        """Return the canonical digest used to bind this proof's decisive fields."""

        from .verification import linear_bound_artifact_hash

        return linear_bound_artifact_hash(self)


@dataclass(frozen=True)
class ScalarBoundProof:
    """Payload for a scalar interval whose named radius components can be summed."""

    center: RationalInput
    radius_components: tuple[tuple[str, RationalInput], ...]
    claimed_radius: RationalInput
    units: str
    premise: str
    provenance: str


@dataclass(frozen=True)
class AnchoredEnvelopeProof:
    """Data-bound proof for an anchored Holder-envelope intersection."""

    lags: tuple[RationalInput, ...]
    responses: tuple[RationalInput, ...]
    holder_L: RationalInput
    holder_alpha: RationalInput
    response_error_bound: RationalInput
    claimed_lower: RationalInput
    claimed_upper: RationalInput
    schema_version: str
    producer_version: str
    artifact_hash: str
    units: str
    premise: str
    provenance: str
    failure_allocations: tuple[tuple[str, RationalInput], ...]
    target_id: str

    def recompute_artifact_hash(self) -> str:
        from .verification import anchored_envelope_artifact_hash

        return anchored_envelope_artifact_hash(self)


@dataclass(frozen=True)
class TemporalBoundProof:
    """Self-contained payload for recomputing a T-CS1 temporal target interval."""

    lags: tuple[RationalInput, ...]
    responses: tuple[RationalInput, ...]
    bandwidth: RationalInput
    n_units: int
    lower_mass_c: RationalInput
    lower_mass_beta: RationalInput
    holder_L: RationalInput
    holder_alpha: RationalInput
    noise_scale: RationalInput
    noise_failure_budget: RationalInput
    baseline_error: RationalInput
    declared_failure_budget: RationalInput
    claimed_center: RationalInput
    claimed_radius: RationalInput
    failure_allocations: tuple[tuple[str, RationalInput], ...]
    schema_version: str
    producer_version: str
    artifact_hash: str
    units: str
    premise: str
    provenance: str
    target_id: str

    def recompute_artifact_hash(self) -> str:
        """Return the canonical digest used to bind this proof's decisive fields."""

        from .verification import temporal_bound_artifact_hash

        return temporal_bound_artifact_hash(self)


@dataclass(frozen=True)
class TemporalDesignBoundProof:
    """Data-bound selected-mean proof under a frozen temporal design."""

    lags: tuple[RationalInput, ...]
    responses: tuple[RationalInput, ...]
    bandwidth: RationalInput
    holder_L: RationalInput
    holder_alpha: RationalInput
    response_error_bound: RationalInput
    claimed_center: RationalInput
    claimed_radius: RationalInput
    failure_allocations: tuple[tuple[str, RationalInput], ...]
    schema_version: str
    producer_version: str
    artifact_hash: str
    units: str
    premise: str
    provenance: str
    target_id: str

    def recompute_artifact_hash(self) -> str:
        """Return the digest binding the selected rows, formula, and failure split."""

        from .verification import temporal_design_bound_artifact_hash

        return temporal_design_bound_artifact_hash(self)


ProofPayload: TypeAlias = (
    ExactNullProof
    | SupportGapProof
    | LinearBoundProof
    | ScalarBoundProof
    | AnchoredEnvelopeProof
    | TemporalBoundProof
    | TemporalDesignBoundProof
)


@dataclass(frozen=True)
class ProofVerification:
    """Sealed result of independently checking one proof payload."""

    valid: bool
    proof_type: str
    reason: str = ""
    recomputed_interval: "TargetInterval | None" = None
    proves_identification: bool = False
    proves_nonidentification: bool = False
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    required_failure_allocations: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class FailureBudget:
    """One uniquely named allocation in a joint failure-probability ledger."""

    component: str
    delta: float
    provenance: str

    def __post_init__(self) -> None:
        if not self.component.strip():
            raise ValueError("failure-budget component must be non-empty")
        if not math.isfinite(self.delta) or not 0.0 < self.delta < 1.0:
            raise ValueError("failure-budget delta must lie strictly between zero and one")
        if not self.provenance.strip():
            raise ValueError("failure-budget provenance must be non-empty")


@dataclass(frozen=True)
class FailureBudgetLedger:
    """Top-level union-bound ledger shared by all conditional certificates."""

    total_delta: float
    allocations: tuple[FailureBudget, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.total_delta) or not 0.0 < self.total_delta < 1.0:
            raise ValueError("total_delta must lie strictly between zero and one")
        names = [allocation.component for allocation in self.allocations]
        if len(names) != len(set(names)):
            raise ValueError("failure-budget component names must be unique")
        if self.spent_delta > self.total_delta + 1e-15:
            raise ValueError("failure-budget allocations exceed total_delta")

    @property
    def spent_delta(self) -> float:
        return float(sum(allocation.delta for allocation in self.allocations))

    @property
    def components(self) -> frozenset[str]:
        return frozenset(allocation.component for allocation in self.allocations)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_delta": self.total_delta,
            "spent_delta": self.spent_delta,
            "allocations": [
                {
                    "component": allocation.component,
                    "delta": allocation.delta,
                    "provenance": allocation.provenance,
                }
                for allocation in self.allocations
            ],
        }


@dataclass(frozen=True)
class TargetInterval:
    """Closed scalar target interval; infinite endpoints denote no finite bound."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if math.isnan(self.lower) or math.isnan(self.upper):
            raise ValueError("target interval endpoints cannot be NaN")
        if self.lower > self.upper:
            raise ValueError("target interval lower endpoint exceeds upper endpoint")

    @classmethod
    def unbounded(cls) -> "TargetInterval":
        return cls(-math.inf, math.inf)

    @property
    def width(self) -> float:
        return float(self.upper - self.lower)

    @property
    def midpoint(self) -> float:
        if not self.is_bounded:
            raise ValueError("an unbounded target interval has no finite midpoint")
        return float((self.lower + self.upper) / 2.0)

    @property
    def is_bounded(self) -> bool:
        return math.isfinite(self.lower) and math.isfinite(self.upper)

    def intersect(self, other: "TargetInterval") -> "TargetInterval | None":
        lower = max(self.lower, other.lower)
        upper = min(self.upper, other.upper)
        if lower > upper:
            return None
        return TargetInterval(float(lower), float(upper))

    def as_dict(self) -> dict[str, float | bool | None]:
        return {
            "lower": self.lower if math.isfinite(self.lower) else None,
            "upper": self.upper if math.isfinite(self.upper) else None,
            "width": self.width if math.isfinite(self.width) else None,
            "bounded": self.is_bounded,
        }


@dataclass(frozen=True)
class TargetCertificate:
    """One module's constraint or failure witness for a common scalar target."""

    source: str
    mode: CertificationMode
    scope: CertificateScope
    valid: bool
    feasible_set: TargetInterval | None = None
    point_estimate: float | None = None
    error_radius: float | None = None
    null_witness: tuple[float, ...] | None = None
    assumptions: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""
    claim_type: CertificateClaimType = CertificateClaimType.NUMERICAL_DIAGNOSTIC
    budget_components: tuple[str, ...] = ()
    failure_allocations: tuple[tuple[str, float], ...] = ()
    failure_semantics: FailureSemantics = FailureSemantics.DETERMINISTIC_BOUND
    numerical_slack: float = 0.0
    provenance: str = ""
    proof: ProofPayload | None = None
    set_subtype: SetSubtype | None = None
    target_id: str = ""


@dataclass(frozen=True)
class CromeDecision:
    """Safe public decision; only the point status may expose a point estimate."""

    status: DecisionStatus
    certificate_scope: CertificateScope
    feasible_set: TargetInterval
    point_estimate: float | None
    uncertainty: TargetInterval | None
    assumptions: tuple[str, ...]
    reasons: tuple[str, ...]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    set_subtype: SetSubtype | None = None
    joint_failure_probability: float | None = None
    numerical_slack: float = 0.0
    structural_status: StructuralStatus = StructuralStatus.UNKNOWN
    operational_status: OperationalStatus = OperationalStatus.INCONCLUSIVE

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "certificate_scope": self.certificate_scope.value,
            "feasible_set": self.feasible_set.as_dict(),
            "point_estimate": self.point_estimate,
            "uncertainty": self.uncertainty.as_dict() if self.uncertainty else None,
            "assumptions": list(self.assumptions),
            "reasons": list(self.reasons),
            "diagnostics": dict(self.diagnostics),
            "set_subtype": self.set_subtype.value if self.set_subtype else None,
            "joint_failure_probability": self.joint_failure_probability,
            "numerical_slack": self.numerical_slack,
            "structural_status": self.structural_status.value,
            "operational_status": self.operational_status.value,
        }
