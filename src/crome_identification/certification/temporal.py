"""Temporal support certificates with explicit population-assumption scope."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from .types import (
    AnchoredEnvelopeProof,
    CertificateClaimType,
    CertificateScope,
    CertificationMode,
    FailureSemantics,
    SetSubtype,
    SupportGapProof,
    TargetCertificate,
    TargetInterval,
    TemporalBoundProof,
    TemporalDesignBoundProof,
)


def _proof_number(value: float) -> str:
    return repr(float(value))


def _temporal_bound_proof(
    lags: np.ndarray,
    responses: np.ndarray,
    *,
    bandwidth: float,
    n_units: int,
    lower_mass_c: float,
    lower_mass_beta: float,
    holder_L: float,
    holder_alpha: float,
    noise_scale: float,
    noise_failure_budget: float,
    baseline_error: float,
    declared_failure_budget: float,
    center: float,
    radius: float,
    failure_allocations: tuple[tuple[str, float], ...],
    target_id: str,
    provenance: str,
) -> TemporalBoundProof:
    proof = TemporalBoundProof(
        lags=tuple(_proof_number(value) for value in lags),
        responses=tuple(_proof_number(value) for value in responses),
        bandwidth=_proof_number(bandwidth),
        n_units=n_units,
        lower_mass_c=_proof_number(lower_mass_c),
        lower_mass_beta=_proof_number(lower_mass_beta),
        holder_L=_proof_number(holder_L),
        holder_alpha=_proof_number(holder_alpha),
        noise_scale=_proof_number(noise_scale),
        noise_failure_budget=_proof_number(noise_failure_budget),
        baseline_error=_proof_number(baseline_error),
        declared_failure_budget=_proof_number(declared_failure_budget),
        claimed_center=_proof_number(center),
        claimed_radius=_proof_number(radius),
        failure_allocations=tuple(
            (name, _proof_number(delta)) for name, delta in failure_allocations
        ),
        schema_version="crome.temporal-bound.v1",
        producer_version="crome-identification/0.1.0",
        artifact_hash="sha256:" + "0" * 64,
        units="declared scalar-target units",
        premise="trajectory-level lower-mass and simultaneous boundary-error event",
        provenance=provenance,
        target_id=target_id,
    )
    return replace(proof, artifact_hash=proof.recompute_artifact_hash())


def _anchored_envelope_proof(
    lags: np.ndarray,
    responses: np.ndarray,
    *,
    holder_L: float,
    holder_alpha: float,
    response_error_bound: float,
    interval: TargetInterval,
    failure_allocations: tuple[tuple[str, float], ...],
    target_id: str,
    provenance: str,
) -> AnchoredEnvelopeProof:
    proof = AnchoredEnvelopeProof(
        lags=tuple(_proof_number(value) for value in lags),
        responses=tuple(_proof_number(value) for value in responses),
        holder_L=_proof_number(holder_L),
        holder_alpha=_proof_number(holder_alpha),
        response_error_bound=_proof_number(response_error_bound),
        claimed_lower=_proof_number(interval.lower),
        claimed_upper=_proof_number(interval.upper),
        schema_version="crome.anchored-envelope.v1",
        producer_version="crome-identification/0.1.0",
        artifact_hash="sha256:" + "0" * 64,
        units="declared scalar-target units",
        premise="noise-expanded anchored Holder-envelope intersection",
        provenance=provenance,
        failure_allocations=tuple(
            (name, _proof_number(delta)) for name, delta in failure_allocations
        ),
        target_id=target_id,
    )
    return replace(proof, artifact_hash=proof.recompute_artifact_hash())


def _temporal_design_bound_proof(
    lags: np.ndarray,
    responses: np.ndarray,
    *,
    bandwidth: float,
    holder_L: float,
    holder_alpha: float,
    response_error_bound: float,
    center: float,
    radius: float,
    failure_allocations: tuple[tuple[str, float], ...],
    target_id: str,
    provenance: str,
) -> TemporalDesignBoundProof:
    proof = TemporalDesignBoundProof(
        lags=tuple(_proof_number(value) for value in lags),
        responses=tuple(_proof_number(value) for value in responses),
        bandwidth=_proof_number(bandwidth),
        holder_L=_proof_number(holder_L),
        holder_alpha=_proof_number(holder_alpha),
        response_error_bound=_proof_number(response_error_bound),
        claimed_center=_proof_number(center),
        claimed_radius=_proof_number(radius),
        failure_allocations=tuple(
            (name, _proof_number(delta)) for name, delta in failure_allocations
        ),
        schema_version="crome.temporal-design-bound.v1",
        producer_version="crome-identification/0.1.0",
        artifact_hash="sha256:" + "0" * 64,
        units="declared scalar-target units",
        premise="frozen support accumulation and simultaneous response-error bound",
        provenance=provenance,
        target_id=target_id,
    )
    return replace(proof, artifact_hash=proof.recompute_artifact_hash())


def _positive_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _nonnegative_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def certify_temporal_design_known(
    support_accumulates: bool | None,
    *,
    lags: np.ndarray | None = None,
    responses: np.ndarray | None = None,
    bandwidth: float | None = None,
    holder_L: float | None = None,
    holder_alpha: float = 1.0,
    response_error_bound: float | None = None,
    identified_set: TargetInterval | None = None,
    budget_components: tuple[str, ...] = (),
    component_delta: float | None = None,
    provenance: str = "",
    support_gap_proof: SupportGapProof | None = None,
    target_id: str = "scalar_target",
) -> TargetCertificate:
    """Certify a target when the endpoint support mechanism is known by design."""

    target_id = target_id.strip()
    if not target_id:
        raise ValueError("target_id must be non-empty")
    failure_allocations: tuple[tuple[str, float], ...] = ()
    if budget_components:
        if component_delta is None:
            raise ValueError("probabilistic design-known bounds require component_delta")
        if not math.isfinite(component_delta) or not 0.0 < component_delta < 1.0:
            raise ValueError("component_delta must lie strictly between zero and one")
        if any(not component.strip() for component in budget_components):
            raise ValueError("budget_components must contain non-empty names")
        failure_allocations = tuple(
            (component, float(component_delta)) for component in budget_components
        )
        failure_semantics = FailureSemantics.PROBABILITY_LEDGER
    else:
        if component_delta is not None:
            raise ValueError("component_delta requires named budget_components")
        failure_semantics = FailureSemantics.DETERMINISTIC_BOUND

    if support_accumulates is None:
        return TargetCertificate(
            source="temporal_support",
            mode=CertificationMode.DESIGN_KNOWN,
            scope=(
                CertificateScope.ASSUMPTION_CONDITIONAL
                if budget_components
                else CertificateScope.POPULATION_EXACT
            ),
            valid=False,
            assumptions=("known endpoint support mechanism",),
            reason="design-known support status was not supplied",
            claim_type=CertificateClaimType.NUMERICAL_DIAGNOSTIC,
            budget_components=budget_components,
            failure_allocations=failure_allocations,
            failure_semantics=failure_semantics,
            provenance=provenance,
        )

    if support_accumulates:
        if (
            lags is None
            or responses is None
            or bandwidth is None
            or holder_L is None
            or response_error_bound is None
        ):
            return TargetCertificate(
                source="temporal_support",
                mode=CertificationMode.DESIGN_KNOWN,
                scope=(
                    CertificateScope.ASSUMPTION_CONDITIONAL
                    if budget_components
                    else CertificateScope.POPULATION_EXACT
                ),
                valid=False,
                assumptions=("known support accumulation at zero",),
                reason=(
                    "data-bound lags, responses, bandwidth, Holder parameters, and "
                    "response-error bound are required under accumulation"
                ),
                claim_type=CertificateClaimType.NUMERICAL_DIAGNOSTIC,
                budget_components=budget_components,
                failure_allocations=failure_allocations,
                failure_semantics=failure_semantics,
                provenance=provenance,
            )
        lag_values = np.asarray(lags, dtype=float)
        response_values = np.asarray(responses, dtype=float)
        if (
            lag_values.ndim != 1
            or response_values.ndim != 1
            or lag_values.shape != response_values.shape
            or lag_values.size == 0
        ):
            raise ValueError("lags and responses must be non-empty equal-length vectors")
        if (
            not np.all(np.isfinite(lag_values))
            or not np.all(np.isfinite(response_values))
            or np.any(lag_values <= 0)
        ):
            raise ValueError("lags must be positive and all design-bound data must be finite")
        bandwidth = _positive_finite("bandwidth", bandwidth)
        holder_L = _nonnegative_finite("holder_L", holder_L)
        holder_alpha = _positive_finite("holder_alpha", holder_alpha)
        response_error_bound = _nonnegative_finite(
            "response_error_bound", response_error_bound
        )
        selected = lag_values <= bandwidth
        n_selected = int(np.sum(selected))
        if n_selected == 0:
            return TargetCertificate(
                source="temporal_support",
                mode=CertificationMode.DESIGN_KNOWN,
                scope=CertificateScope.FINITE_SAMPLE_ONLY,
                valid=False,
                assumptions=("known support accumulation at zero",),
                reason="the frozen design selects no near-zero row",
                claim_type=CertificateClaimType.NUMERICAL_DIAGNOSTIC,
                budget_components=budget_components,
                failure_allocations=failure_allocations,
                failure_semantics=failure_semantics,
                provenance=provenance,
            )
        estimate = float(np.mean(response_values[selected]))
        radius = float(
            holder_L * bandwidth**holder_alpha
            + response_error_bound / math.sqrt(n_selected)
        )
        interval = TargetInterval(estimate - radius, estimate + radius)
        return TargetCertificate(
            source="temporal_support",
            mode=CertificationMode.DESIGN_KNOWN,
            scope=(
                CertificateScope.ASSUMPTION_CONDITIONAL
                if budget_components
                else CertificateScope.POPULATION_EXACT
            ),
            valid=True,
            feasible_set=interval,
            point_estimate=estimate,
            error_radius=radius,
            assumptions=("known support accumulation at zero",),
            diagnostics={"support_accumulates": True, "n_selected": n_selected},
            reason="design-known accumulation supplies a boundary constraint",
            claim_type=(
                CertificateClaimType.STRUCTURAL_EVIDENCE
                if radius == 0.0
                else CertificateClaimType.ROBUST_TARGET_BOUND
            ),
            budget_components=budget_components,
            failure_allocations=failure_allocations,
            failure_semantics=failure_semantics,
            provenance=provenance,
            proof=_temporal_design_bound_proof(
                lag_values,
                response_values,
                bandwidth=bandwidth,
                holder_L=holder_L,
                holder_alpha=holder_alpha,
                response_error_bound=response_error_bound,
                center=estimate,
                radius=radius,
                failure_allocations=failure_allocations,
                target_id=target_id,
                provenance=provenance,
            ),
            set_subtype=(
                SetSubtype.STRUCTURAL_IDENTIFIED
                if radius == 0.0
                else SetSubtype.CONFIDENCE_OUTER
            ),
            target_id=target_id,
        )

    if identified_set is not None:
        return TargetCertificate(
            source="temporal_support",
            mode=CertificationMode.DESIGN_KNOWN,
            scope=(
                CertificateScope.ASSUMPTION_CONDITIONAL
                if budget_components
                else CertificateScope.POPULATION_EXACT
            ),
            valid=False,
            assumptions=("known positive support gap", "declared target envelope"),
            diagnostics={"support_accumulates": False},
            reason=(
                "a caller-declared identified_set is diagnostic-only; use a "
                "data-bound anchored-envelope proof"
            ),
            claim_type=CertificateClaimType.NUMERICAL_DIAGNOSTIC,
            budget_components=budget_components,
            failure_allocations=failure_allocations,
            failure_semantics=failure_semantics,
            provenance=provenance,
            target_id=target_id,
        )

    if support_gap_proof is None:
        return TargetCertificate(
            source="temporal_support",
            mode=CertificationMode.DESIGN_KNOWN,
            scope=CertificateScope.FINITE_SAMPLE_ONLY,
            valid=False,
            assumptions=("caller-declared positive support gap",),
            diagnostics={"support_accumulates": False},
            reason="a Boolean support-gap label is not a verifiable proof payload",
            claim_type=CertificateClaimType.NUMERICAL_DIAGNOSTIC,
            budget_components=budget_components,
            provenance=provenance,
        )
    if support_gap_proof.target_id != target_id:
        return TargetCertificate(
            source="temporal_support",
            mode=CertificationMode.DESIGN_KNOWN,
            scope=CertificateScope.POPULATION_EXACT,
            valid=False,
            assumptions=("verified positive support gap",),
            reason="support-gap proof target id does not match the requested target",
            claim_type=CertificateClaimType.NUMERICAL_DIAGNOSTIC,
            provenance=provenance,
            target_id=target_id,
        )
    return TargetCertificate(
        source="temporal_support",
        mode=CertificationMode.DESIGN_KNOWN,
        scope=CertificateScope.POPULATION_EXACT,
        valid=True,
        null_witness=(
            float(support_gap_proof.target_value_right)
            - float(support_gap_proof.target_value_left),
        ),
        assumptions=("verified positive support gap", "verified equivalence construction"),
        diagnostics={"support_accumulates": False},
        reason="support gap admits a target-changing observational-equivalence bump",
        claim_type=CertificateClaimType.EXACT_WITNESS,
        budget_components=budget_components,
        provenance=provenance,
        proof=support_gap_proof,
        target_id=target_id,
    )


def certify_temporal_conditional(
    lags: np.ndarray,
    responses: np.ndarray,
    *,
    bandwidth: float,
    n_units: int,
    lower_mass_c: float,
    lower_mass_beta: float,
    holder_L: float,
    holder_alpha: float,
    noise_scale: float,
    delta: float,
    baseline_error: float = 0.0,
    independent_units: bool = True,
    budget_components: tuple[str, ...] = (),
    external_failure_allocations: tuple[tuple[str, float], ...] = (),
    target_id: str = "scalar_target",
    provenance: str = "",
) -> TargetCertificate:
    """Conditional near-zero certificate under a declared lower-mass assumption.

    The simplified proof route treats the supplied rows as independent units. Clustered
    trajectory data must first be reduced to independent trajectory-level contributions.
    """

    lag_values = np.asarray(lags, dtype=float)
    trace_values = np.asarray(responses, dtype=float)
    if lag_values.ndim != 1 or trace_values.ndim != 1 or lag_values.shape != trace_values.shape:
        raise ValueError("lags and responses must be one-dimensional arrays of equal length")
    if not np.all(np.isfinite(lag_values)) or not np.all(np.isfinite(trace_values)):
        raise ValueError("lags and responses must contain finite values")
    if np.any(lag_values <= 0):
        raise ValueError("lags must be strictly positive")
    target_id = target_id.strip()
    if not target_id:
        raise ValueError("target_id must be non-empty")

    bandwidth = _positive_finite("bandwidth", bandwidth)
    lower_mass_c = _positive_finite("lower_mass_c", lower_mass_c)
    lower_mass_beta = _positive_finite("lower_mass_beta", lower_mass_beta)
    holder_L = _nonnegative_finite("holder_L", holder_L)
    holder_alpha = _positive_finite("holder_alpha", holder_alpha)
    noise_scale = _nonnegative_finite("noise_scale", noise_scale)
    baseline_error = _nonnegative_finite("baseline_error", baseline_error)
    if not isinstance(n_units, (int, np.integer)) or int(n_units) <= 0:
        raise ValueError("n_units must be a positive integer")
    n_units = int(n_units)
    if lag_values.size > n_units:
        raise ValueError(
            "the number of temporal rows cannot exceed n_units; reduce clustered "
            "trajectories to at most one effective contribution per independent unit"
        )
    if not math.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("delta must lie strictly between zero and one")

    near_zero = lag_values <= bandwidth
    n_near_zero = int(np.sum(near_zero))
    expected_mass = n_units * lower_mass_c * bandwidth**lower_mass_beta
    required_count = expected_mass / 2.0
    count_failure_bound = math.exp(-expected_mass / 8.0)
    noise_failure_budget = delta / 2.0
    count_failure_budget = delta - noise_failure_budget
    total_failure_bound = count_failure_bound + noise_failure_budget
    diagnostics = {
        "n_near_zero": n_near_zero,
        "required_near_zero_count": required_count,
        "count_failure_bound": count_failure_bound,
        "failure_probability_bound": total_failure_bound,
        "count_failure_budget": count_failure_budget,
        "noise_failure_budget": noise_failure_budget,
        "bandwidth": bandwidth,
        "lower_mass_c": lower_mass_c,
        "lower_mass_beta": lower_mass_beta,
    }
    assumptions = (
        f"P(0<Q<=h) >= {lower_mass_c:g} h^{lower_mass_beta:g}",
        "independent effective units",
        "anchored response envelope",
        "sub-Gaussian response noise",
    )
    if not budget_components:
        budget_components = ("temporal_count", "trace_noise")
    if len(budget_components) != 2 or any(not name.strip() for name in budget_components):
        raise ValueError("conditional temporal certification requires count and trace-noise components")
    external_names = [name for name, _ in external_failure_allocations]
    if (
        any(not name.strip() for name in external_names)
        or len(external_names) != len(set(external_names))
        or set(external_names) & set(budget_components)
        or any(
            not math.isfinite(component_delta) or not 0.0 < component_delta < 1.0
            for _, component_delta in external_failure_allocations
        )
    ):
        raise ValueError("external failure allocations require unique names and deltas in (0, 1)")
    internal_allocations = (
        (budget_components[0], count_failure_budget),
        (budget_components[1], noise_failure_budget),
    )
    certificate_allocations = internal_allocations + tuple(external_failure_allocations)
    certificate_components = budget_components + tuple(external_names)

    if not independent_units:
        return TargetCertificate(
            source="temporal_support",
            mode=CertificationMode.ASSUMPTION_CONDITIONAL,
            scope=CertificateScope.ASSUMPTION_CONDITIONAL,
            valid=False,
            assumptions=assumptions,
            diagnostics=diagnostics,
            reason="independent effective units were not established",
            claim_type=CertificateClaimType.NUMERICAL_DIAGNOSTIC,
            budget_components=budget_components,
            provenance=provenance,
        )
    if n_near_zero < required_count or n_near_zero == 0 or total_failure_bound > delta:
        return TargetCertificate(
            source="temporal_support",
            mode=CertificationMode.ASSUMPTION_CONDITIONAL,
            scope=CertificateScope.ASSUMPTION_CONDITIONAL,
            valid=False,
            assumptions=assumptions,
            diagnostics=diagnostics,
            reason="near-zero mass is insufficient for the declared failure budget",
            claim_type=CertificateClaimType.NUMERICAL_DIAGNOSTIC,
            budget_components=budget_components,
            provenance=provenance,
        )

    estimate = float(np.mean(trace_values[near_zero]))
    bias_radius = holder_L * bandwidth**holder_alpha
    noise_radius = noise_scale * math.sqrt(
        2.0 * math.log(2.0 / noise_failure_budget) / n_near_zero
    )
    radius = float(bias_radius + noise_radius + baseline_error)
    interval = TargetInterval(estimate - radius, estimate + radius)
    diagnostics.update(
        {
            "boundary_estimate": estimate,
            "bias_radius": bias_radius,
            "noise_error_radius": noise_radius,
            "baseline_error": baseline_error,
            "target_error_radius": radius,
        }
    )
    return TargetCertificate(
        source="temporal_support",
        mode=CertificationMode.ASSUMPTION_CONDITIONAL,
        scope=CertificateScope.ASSUMPTION_CONDITIONAL,
        valid=True,
        feasible_set=interval,
        point_estimate=estimate,
        error_radius=radius,
        assumptions=assumptions,
        diagnostics=diagnostics,
        reason="conditional lower-mass and boundary-error certificate",
        claim_type=CertificateClaimType.ROBUST_TARGET_BOUND,
        budget_components=certificate_components,
        failure_allocations=certificate_allocations,
        failure_semantics=FailureSemantics.PROBABILITY_LEDGER,
        provenance=provenance,
        proof=_temporal_bound_proof(
            lag_values,
            trace_values,
            bandwidth=bandwidth,
            n_units=n_units,
            lower_mass_c=lower_mass_c,
            lower_mass_beta=lower_mass_beta,
            holder_L=holder_L,
            holder_alpha=holder_alpha,
            noise_scale=noise_scale,
            noise_failure_budget=noise_failure_budget,
            baseline_error=baseline_error,
            declared_failure_budget=delta,
            center=estimate,
            radius=radius,
            failure_allocations=internal_allocations,
            target_id=target_id,
            provenance=provenance,
        ),
        set_subtype=SetSubtype.ASSUMPTION_CONDITIONAL,
        target_id=target_id,
    )


def certify_temporal_empirical(lags: np.ndarray) -> TargetCertificate:
    """Report finite-sample lag diagnostics without a population support claim."""

    lag_values = np.asarray(lags, dtype=float)
    if lag_values.ndim != 1 or not np.all(np.isfinite(lag_values)):
        raise ValueError("lags must be a one-dimensional finite array")
    positive = lag_values[lag_values > 0]
    minimum = float(np.min(positive)) if positive.size else None
    return TargetCertificate(
        source="temporal_support",
        mode=CertificationMode.EMPIRICAL_ONLY,
        scope=CertificateScope.FINITE_SAMPLE_ONLY,
        valid=True,
        assumptions=(),
        diagnostics={
            "n_positive_lags": int(positive.size),
            "minimum_positive_lag": minimum,
        },
        reason="finite-sample lag diagnostics only",
        claim_type=CertificateClaimType.FINITE_SAMPLE_DIAGNOSTIC,
    )


def certify_temporal_gap_conditional(
    lags: np.ndarray,
    responses: np.ndarray,
    *,
    holder_L: float,
    holder_alpha: float,
    response_error_bound: float,
    budget_components: tuple[str, ...] = (),
    component_delta: float | None = None,
    deterministic_uncertainty: bool = False,
    target_id: str = "scalar_target",
    provenance: str = "",
) -> TargetCertificate:
    """Build a noise-expanded anchored-envelope set under a positive-lag design."""

    lag_values = np.asarray(lags, dtype=float)
    response_values = np.asarray(responses, dtype=float)
    if (
        lag_values.ndim != 1
        or response_values.ndim != 1
        or lag_values.shape != response_values.shape
        or lag_values.size == 0
    ):
        raise ValueError("lags and responses must be non-empty equal-length vectors")
    if not np.all(np.isfinite(lag_values)) or not np.all(np.isfinite(response_values)):
        raise ValueError("lags and responses must contain finite values")
    if np.any(lag_values <= 0):
        raise ValueError("lags must be strictly positive")
    target_id = target_id.strip()
    if not target_id:
        raise ValueError("target_id must be non-empty")
    holder_L = _nonnegative_finite("holder_L", holder_L)
    holder_alpha = _positive_finite("holder_alpha", holder_alpha)
    response_error_bound = _nonnegative_finite(
        "response_error_bound", response_error_bound
    )
    if deterministic_uncertainty:
        if component_delta is not None or budget_components:
            raise ValueError(
                "deterministic_uncertainty cannot carry probability-ledger components"
            )
        failure_allocations: tuple[tuple[str, float], ...] = ()
        failure_semantics = FailureSemantics.DETERMINISTIC_BOUND
    else:
        if component_delta is None:
            raise ValueError(
                "probabilistic response-error bounds require component_delta and "
                "named budget_components"
            )
        if not math.isfinite(component_delta) or not 0.0 < component_delta < 1.0:
            raise ValueError("component_delta must lie strictly between zero and one")
        if not budget_components or any(not component.strip() for component in budget_components):
            raise ValueError("component_delta requires named budget_components")
        failure_allocations = tuple(
            (component, float(component_delta)) for component in budget_components
        )
        failure_semantics = FailureSemantics.PROBABILITY_LEDGER
    half_widths = holder_L * np.power(lag_values, holder_alpha) + response_error_bound
    lower = float(np.max(response_values - half_widths))
    upper = float(np.min(response_values + half_widths))
    assumptions = (
        "anchored Holder response envelope",
        "simultaneous response-error bound",
        "positive-lag observation design",
    )
    diagnostics = {
        "minimum_lag": float(np.min(lag_values)),
        "n_lags": int(lag_values.size),
        "holder_L": holder_L,
        "holder_alpha": holder_alpha,
        "response_error_bound": response_error_bound,
        "raw_lower": lower,
        "raw_upper": upper,
    }
    if lower > upper:
        return TargetCertificate(
            source="temporal_gap",
            mode=CertificationMode.ASSUMPTION_CONDITIONAL,
            scope=CertificateScope.ASSUMPTION_CONDITIONAL,
            valid=False,
            assumptions=assumptions,
            diagnostics=diagnostics,
            reason="noise-expanded anchored constraints have an empty intersection",
            claim_type=CertificateClaimType.NUMERICAL_DIAGNOSTIC,
            budget_components=budget_components,
            failure_allocations=failure_allocations,
            failure_semantics=failure_semantics,
            provenance=provenance,
        )
    interval = TargetInterval(lower, upper)
    return TargetCertificate(
        source="temporal_gap",
        mode=CertificationMode.ASSUMPTION_CONDITIONAL,
        scope=CertificateScope.ASSUMPTION_CONDITIONAL,
        valid=True,
        feasible_set=interval,
        assumptions=assumptions,
        diagnostics=diagnostics,
        reason="conditional noise-expanded anchored-envelope target set",
        claim_type=CertificateClaimType.ROBUST_TARGET_BOUND,
        budget_components=budget_components,
        failure_allocations=failure_allocations,
        failure_semantics=failure_semantics,
        provenance=provenance,
        proof=_anchored_envelope_proof(
            lag_values,
            response_values,
            holder_L=holder_L,
            holder_alpha=holder_alpha,
            response_error_bound=response_error_bound,
            interval=interval,
            failure_allocations=failure_allocations,
            target_id=target_id,
            provenance=provenance,
        ),
        set_subtype=SetSubtype.ASSUMPTION_CONDITIONAL,
        target_id=target_id,
    )
