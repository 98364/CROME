"""Independent verification for proof-carrying CROME certificates."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
import re

from .types import (
    AnchoredEnvelopeProof,
    ExactNullProof,
    LinearBoundProof,
    ProofPayload,
    ProofVerification,
    RationalInput,
    ScalarBoundProof,
    SupportGapProof,
    TemporalBoundProof,
    TemporalDesignBoundProof,
    TargetInterval,
)


_TEMPORAL_BOUND_SCHEMA = "crome.temporal-bound.v1"
_LINEAR_BOUND_SCHEMA = "crome.linear-bound.v1"
_SUPPORT_GAP_SCHEMA = "crome.support-gap.v1"
_ANCHORED_ENVELOPE_SCHEMA = "crome.anchored-envelope.v1"
_TEMPORAL_DESIGN_BOUND_SCHEMA = "crome.temporal-design-bound.v1"
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _fraction(value: RationalInput) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise ValueError("Boolean values are not rational proof entries")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, str) and value.strip():
        return Fraction(value)
    raise ValueError("Exact proof entries must be integers, decimal strings, or Fractions")


def _vector(values: tuple[RationalInput, ...], name: str) -> tuple[Fraction, ...]:
    if not values:
        raise ValueError(f"{name} must be non-empty")
    return tuple(_fraction(value) for value in values)


def _matrix(
    values: tuple[tuple[RationalInput, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    if not values or not values[0]:
        raise ValueError("operator must be a non-empty rectangular matrix")
    width = len(values[0])
    if any(len(row) != width for row in values):
        raise ValueError("operator must be rectangular")
    return tuple(tuple(_fraction(value) for value in row) for row in values)


def _dot(left: tuple[Fraction, ...], right: tuple[Fraction, ...]) -> Fraction:
    return sum((a * b for a, b in zip(left, right, strict=True)), Fraction(0))


def _nonempty_metadata(*values: str) -> bool:
    return all(isinstance(value, str) and bool(value.strip()) for value in values)


def _invalid(proof: ProofPayload, reason: str) -> ProofVerification:
    return ProofVerification(valid=False, proof_type=type(proof).__name__, reason=reason)


def _canonical_rational(value: RationalInput) -> tuple[int, int]:
    rational = _fraction(value)
    return rational.numerator, rational.denominator


def temporal_bound_artifact_hash(proof: TemporalBoundProof) -> str:
    """Hash every decisive temporal-proof field except the digest itself."""

    payload = {
        "lags": [_canonical_rational(value) for value in proof.lags],
        "responses": [_canonical_rational(value) for value in proof.responses],
        "bandwidth": _canonical_rational(proof.bandwidth),
        "n_units": proof.n_units,
        "lower_mass_c": _canonical_rational(proof.lower_mass_c),
        "lower_mass_beta": _canonical_rational(proof.lower_mass_beta),
        "holder_L": _canonical_rational(proof.holder_L),
        "holder_alpha": _canonical_rational(proof.holder_alpha),
        "noise_scale": _canonical_rational(proof.noise_scale),
        "noise_failure_budget": _canonical_rational(proof.noise_failure_budget),
        "baseline_error": _canonical_rational(proof.baseline_error),
        "declared_failure_budget": _canonical_rational(proof.declared_failure_budget),
        "claimed_center": _canonical_rational(proof.claimed_center),
        "claimed_radius": _canonical_rational(proof.claimed_radius),
        "failure_allocations": [
            (name, _canonical_rational(value))
            for name, value in proof.failure_allocations
        ],
        "schema_version": proof.schema_version,
        "producer_version": proof.producer_version,
        "units": proof.units,
        "premise": proof.premise,
        "provenance": proof.provenance,
        "target_id": proof.target_id,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def linear_bound_artifact_hash(proof: LinearBoundProof) -> str:
    """Hash the linear operator, sufficient statistics, budgets, and metadata."""

    payload = {
        "operator": [
            [_canonical_rational(value) for value in row] for row in proof.operator
        ],
        "outcome": [_canonical_rational(value) for value in proof.outcome],
        "target": [_canonical_rational(value) for value in proof.target],
        "weights": [_canonical_rational(value) for value in proof.weights],
        "theta_radius": _canonical_rational(proof.theta_radius),
        "design_error_bound": _canonical_rational(proof.design_error_bound),
        "noise_radius": _canonical_rational(proof.noise_radius),
        "baseline_support_radius": _canonical_rational(proof.baseline_support_radius),
        "approximation_support_radius": _canonical_rational(
            proof.approximation_support_radius
        ),
        "additive_baseline_error": _canonical_rational(proof.additive_baseline_error),
        "additive_approximation_error": _canonical_rational(
            proof.additive_approximation_error
        ),
        "center": _canonical_rational(proof.center),
        "claimed_radius": _canonical_rational(proof.claimed_radius),
        "units": proof.units,
        "uncertainty_definition": proof.uncertainty_definition,
        "provenance": proof.provenance,
        "schema_version": proof.schema_version,
        "producer_version": proof.producer_version,
        "failure_allocations": [
            (name, _canonical_rational(value))
            for name, value in proof.failure_allocations
        ],
        "target_id": proof.target_id,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def support_gap_artifact_hash(proof: SupportGapProof) -> str:
    """Hash the finite-support local-bump construction and its metadata."""

    payload = {
        "observed_support": [_canonical_rational(value) for value in proof.observed_support],
        "gap": [_canonical_rational(value) for value in proof.gap],
        "observed_signature_left": [
            _canonical_rational(value) for value in proof.observed_signature_left
        ],
        "observed_signature_right": [
            _canonical_rational(value) for value in proof.observed_signature_right
        ],
        "target_value_left": _canonical_rational(proof.target_value_left),
        "target_value_right": _canonical_rational(proof.target_value_right),
        "baseline_level": _canonical_rational(proof.baseline_level),
        "bump_amplitude": _canonical_rational(proof.bump_amplitude),
        "bump_power": proof.bump_power,
        "response_class_premise": proof.response_class_premise,
        "support_specification": proof.support_specification,
        "units": proof.units,
        "provenance": proof.provenance,
        "schema_version": proof.schema_version,
        "producer_version": proof.producer_version,
        "target_id": proof.target_id,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def anchored_envelope_artifact_hash(proof: AnchoredEnvelopeProof) -> str:
    payload = {
        "lags": [_canonical_rational(value) for value in proof.lags],
        "responses": [_canonical_rational(value) for value in proof.responses],
        "holder_L": _canonical_rational(proof.holder_L),
        "holder_alpha": _canonical_rational(proof.holder_alpha),
        "response_error_bound": _canonical_rational(proof.response_error_bound),
        "claimed_lower": _canonical_rational(proof.claimed_lower),
        "claimed_upper": _canonical_rational(proof.claimed_upper),
        "schema_version": proof.schema_version,
        "producer_version": proof.producer_version,
        "units": proof.units,
        "premise": proof.premise,
        "provenance": proof.provenance,
        "failure_allocations": [
            (name, _canonical_rational(value))
            for name, value in proof.failure_allocations
        ],
        "target_id": proof.target_id,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def temporal_design_bound_artifact_hash(proof: TemporalDesignBoundProof) -> str:
    """Hash the frozen selector, observations, radius formula, and failure split."""

    payload = {
        "lags": [_canonical_rational(value) for value in proof.lags],
        "responses": [_canonical_rational(value) for value in proof.responses],
        "bandwidth": _canonical_rational(proof.bandwidth),
        "holder_L": _canonical_rational(proof.holder_L),
        "holder_alpha": _canonical_rational(proof.holder_alpha),
        "response_error_bound": _canonical_rational(proof.response_error_bound),
        "claimed_center": _canonical_rational(proof.claimed_center),
        "claimed_radius": _canonical_rational(proof.claimed_radius),
        "failure_allocations": [
            (name, _canonical_rational(value))
            for name, value in proof.failure_allocations
        ],
        "schema_version": proof.schema_version,
        "producer_version": proof.producer_version,
        "units": proof.units,
        "premise": proof.premise,
        "provenance": proof.provenance,
        "target_id": proof.target_id,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _failure_allocations(
    values: tuple[tuple[str, RationalInput], ...],
) -> tuple[dict[str, float] | None, str | None]:
    names = [name for name, _ in values]
    if any(not isinstance(name, str) or not name.strip() for name in names):
        return None, "failure allocations require non-empty component names"
    if len(names) != len(set(names)):
        return None, "failure-allocation component names must be unique"
    try:
        parsed = {name: float(_fraction(delta)) for name, delta in values}
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        return None, str(exc)
    if any(not math.isfinite(delta) or not 0.0 < delta < 1.0 for delta in parsed.values()):
        return None, "failure-allocation deltas must lie strictly between zero and one"
    return parsed, None


def _verify_anchored_envelope(proof: AnchoredEnvelopeProof) -> ProofVerification:
    if proof.schema_version != _ANCHORED_ENVELOPE_SCHEMA:
        return _invalid(proof, f"unsupported anchored-envelope proof schema: {proof.schema_version!r}")
    if not isinstance(proof.artifact_hash, str) or not _SHA256_PATTERN.fullmatch(
        proof.artifact_hash
    ):
        return _invalid(proof, "artifact hash must be a lowercase sha256 digest")
    try:
        recomputed_hash = anchored_envelope_artifact_hash(proof)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        return _invalid(proof, str(exc))
    if proof.artifact_hash != recomputed_hash:
        return _invalid(proof, "artifact hash does not match the anchored-envelope payload")
    if not _nonempty_metadata(
        proof.producer_version,
        proof.units,
        proof.premise,
        proof.provenance,
        proof.target_id,
    ):
        return _invalid(
            proof,
            "producer version, units, premise, provenance, and target id are required",
        )
    allocations, allocation_reason = _failure_allocations(proof.failure_allocations)
    if allocation_reason:
        return _invalid(proof, allocation_reason)
    try:
        lags = _vector(proof.lags, "lags")
        responses = _vector(proof.responses, "responses")
        holder_L = _fraction(proof.holder_L)
        holder_alpha = _fraction(proof.holder_alpha)
        response_error = _fraction(proof.response_error_bound)
        claimed_lower = _fraction(proof.claimed_lower)
        claimed_upper = _fraction(proof.claimed_upper)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        return _invalid(proof, str(exc))
    if len(lags) != len(responses):
        return _invalid(proof, "lags and responses must have equal non-zero length")
    if any(value <= 0 for value in lags) or holder_L < 0 or holder_alpha <= 0 or response_error < 0:
        return _invalid(proof, "lags and Holder/error parameters are outside their admissible ranges")
    half_widths = [
        float(holder_L) * float(lag) ** float(holder_alpha) + float(response_error)
        for lag in lags
    ]
    lower = max(float(response) - width for response, width in zip(responses, half_widths, strict=True))
    upper = min(float(response) + width for response, width in zip(responses, half_widths, strict=True))
    if lower > upper:
        return _invalid(proof, "anchored-envelope constraints have an empty intersection")
    if not math.isclose(lower, float(claimed_lower), rel_tol=1e-12, abs_tol=1e-12) or not math.isclose(
        upper, float(claimed_upper), rel_tol=1e-12, abs_tol=1e-12
    ):
        return _invalid(proof, "claimed interval does not match the recomputed anchored envelope")
    return ProofVerification(
        valid=True,
        proof_type=type(proof).__name__,
        recomputed_interval=TargetInterval(lower, upper),
        diagnostics={"artifact_hash": recomputed_hash, "n_lags": len(lags)},
        required_failure_allocations=allocations or {},
    )


def _verify_exact_null(proof: ExactNullProof) -> ProofVerification:
    try:
        operator = _matrix(proof.operator)
        target = _vector(proof.target, "target")
        witness = _vector(proof.witness, "witness")
    except (TypeError, ValueError) as exc:
        return _invalid(proof, str(exc))
    if not _nonempty_metadata(
        proof.operator_id, proof.target_id, proof.units, proof.provenance
    ):
        return _invalid(proof, "operator id, target id, units, and provenance are required")
    if len(target) != len(operator[0]) or len(witness) != len(target):
        return _invalid(proof, "operator, target, and witness dimensions do not agree")
    residual = tuple(_dot(row, witness) for row in operator)
    if any(value != 0 for value in residual):
        return _invalid(proof, "exact witness residual A v is nonzero")
    target_variation = _dot(target, witness)
    if target_variation == 0:
        return _invalid(proof, "exact witness does not change the target")
    if proof.parameter_space not in {"unrestricted", "box"}:
        return _invalid(proof, "parameter_space must be unrestricted or box")
    if proof.parameter_space == "box":
        required = (
            proof.base_parameter,
            proof.step,
            proof.parameter_lower,
            proof.parameter_upper,
        )
        if any(value is None for value in required):
            return _invalid(proof, "box feasibility requires base, step, lower, and upper")
        try:
            base = _vector(proof.base_parameter or (), "base_parameter")
            lower = _vector(proof.parameter_lower or (), "parameter_lower")
            upper = _vector(proof.parameter_upper or (), "parameter_upper")
            step = _fraction(proof.step)
        except (TypeError, ValueError) as exc:
            return _invalid(proof, str(exc))
        if not (len(base) == len(lower) == len(upper) == len(witness)):
            return _invalid(proof, "box feasibility dimensions do not agree")
        if step == 0:
            return _invalid(proof, "box witness step must be nonzero")
        shifted = tuple(value + step * direction for value, direction in zip(base, witness))
        if any(lo > hi for lo, hi in zip(lower, upper)) or any(
            not (lo <= value <= hi and lo <= moved <= hi)
            for value, moved, lo, hi in zip(base, shifted, lower, upper)
        ):
            return _invalid(proof, "witness is not feasible in the declared parameter box")
    return ProofVerification(
        valid=True,
        proof_type=type(proof).__name__,
        proves_nonidentification=True,
        diagnostics={
            "exact_residual": residual,
            "target_variation": target_variation,
            "operator_id": proof.operator_id,
        },
    )


def _verify_support_gap(proof: SupportGapProof) -> ProofVerification:
    if proof.schema_version != _SUPPORT_GAP_SCHEMA:
        return _invalid(proof, f"unsupported support-gap proof schema: {proof.schema_version!r}")
    if not isinstance(proof.artifact_hash, str) or not _SHA256_PATTERN.fullmatch(
        proof.artifact_hash
    ):
        return _invalid(proof, "artifact hash must be a lowercase sha256 digest")
    try:
        recomputed_hash = support_gap_artifact_hash(proof)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        return _invalid(proof, str(exc))
    if proof.artifact_hash != recomputed_hash:
        return _invalid(proof, "artifact hash does not match the support-gap proof payload")
    try:
        support = _vector(proof.observed_support, "observed_support")
        gap_lower, gap_upper = (_fraction(value) for value in proof.gap)
        signature_left = _vector(proof.observed_signature_left, "observed_signature_left")
        signature_right = _vector(proof.observed_signature_right, "observed_signature_right")
        target_left = _fraction(proof.target_value_left)
        target_right = _fraction(proof.target_value_right)
        baseline = _fraction(proof.baseline_level)
        amplitude = _fraction(proof.bump_amplitude)
    except (TypeError, ValueError) as exc:
        return _invalid(proof, str(exc))
    if not _nonempty_metadata(
        proof.response_class_premise,
        proof.support_specification,
        proof.units,
        proof.provenance,
        proof.producer_version,
        proof.target_id,
    ):
        return _invalid(proof, "support specification, premises, units, and provenance are required")
    if gap_lower != 0 or gap_upper <= 0:
        return _invalid(
            proof,
            "support-gap nonidentification requires a zero-anchored boundary gap (0, a)",
        )
    if any(point <= 0 for point in support):
        return _invalid(proof, "observed support must contain strictly positive lags")
    if isinstance(proof.bump_power, bool) or not isinstance(proof.bump_power, int) or proof.bump_power <= 0:
        return _invalid(proof, "bump_power must be a positive integer")
    if amplitude == 0:
        return _invalid(proof, "the local bump amplitude must be nonzero")
    if any(gap_lower < point < gap_upper for point in support):
        return _invalid(proof, "declared observed support intersects the open support gap")
    width = gap_upper - gap_lower
    expected_left = tuple(baseline for _ in support)
    expected_right = tuple(
        baseline
        + (
            amplitude * ((gap_upper - point) / width) ** proof.bump_power
            if gap_lower <= point < gap_upper
            else 0
        )
        for point in support
    )
    if signature_left != expected_left or signature_right != expected_right:
        return _invalid(
            proof,
            "supplied signatures are not observationally equivalent under the declared local-bump construction",
        )
    if signature_left != signature_right:
        return _invalid(proof, "the supplied pair is not observationally equivalent")
    if target_left != baseline or target_right != baseline + amplitude:
        return _invalid(proof, "target values do not match the declared local-bump construction")
    if target_left == target_right:
        return _invalid(proof, "the observationally equivalent pair does not change the target")
    return ProofVerification(
        valid=True,
        proof_type=type(proof).__name__,
        proves_nonidentification=True,
        diagnostics={
            "gap_width": gap_upper - gap_lower,
            "target_variation": target_right - target_left,
            "artifact_hash": recomputed_hash,
        },
    )


def _verify_linear_bound(proof: LinearBoundProof) -> ProofVerification:
    if proof.schema_version != _LINEAR_BOUND_SCHEMA:
        return _invalid(proof, f"unsupported linear proof schema: {proof.schema_version!r}")
    if not _nonempty_metadata(
        proof.producer_version,
        proof.units,
        proof.uncertainty_definition,
        proof.provenance,
        proof.target_id,
    ):
        return _invalid(
            proof,
            "producer version, units, uncertainty-set definition, and provenance are required",
        )
    if not isinstance(proof.artifact_hash, str) or not _SHA256_PATTERN.fullmatch(
        proof.artifact_hash
    ):
        return _invalid(proof, "artifact hash must be a lowercase sha256 digest")
    try:
        recomputed_hash = linear_bound_artifact_hash(proof)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        return _invalid(proof, str(exc))
    if proof.artifact_hash != recomputed_hash:
        return _invalid(proof, "artifact hash does not match the linear proof payload")
    allocations, allocation_reason = _failure_allocations(proof.failure_allocations)
    if allocation_reason:
        return _invalid(proof, allocation_reason)
    try:
        operator = _matrix(proof.operator)
        outcome = _vector(proof.outcome, "outcome")
        target = _vector(proof.target, "target")
        weights = _vector(proof.weights, "weights")
        theta_radius = _fraction(proof.theta_radius)
        design_error = _fraction(proof.design_error_bound)
        noise_radius = _fraction(proof.noise_radius)
        baseline_support = _fraction(proof.baseline_support_radius)
        approximation_support = _fraction(proof.approximation_support_radius)
        additive_baseline = _fraction(proof.additive_baseline_error)
        additive_approximation = _fraction(proof.additive_approximation_error)
        center = _fraction(proof.center)
        claimed_radius = _fraction(proof.claimed_radius)
    except (TypeError, ValueError) as exc:
        return _invalid(proof, str(exc))
    if (
        len(target) != len(operator[0])
        or len(weights) != len(operator)
        or len(outcome) != len(operator)
    ):
        return _invalid(proof, "operator, outcome, target, and weight dimensions do not agree")
    budgets = (
        theta_radius,
        design_error,
        noise_radius,
        baseline_support,
        approximation_support,
        additive_baseline,
        additive_approximation,
        claimed_radius,
    )
    if any(value < 0 for value in budgets):
        return _invalid(proof, "all bound radii and budgets must be non-negative")
    weighted_operator = tuple(
        sum((weights[row] * operator[row][column] for row in range(len(operator))), Fraction(0))
        for column in range(len(target))
    )
    residual = tuple(value - fitted for value, fitted in zip(target, weighted_operator))
    residual_norm = math.sqrt(sum(float(value * value) for value in residual))
    amplification = math.sqrt(sum(float(value * value) for value in weights))
    recomputed = (
        float(theta_radius) * residual_norm
        + (
            float(theta_radius * design_error)
            + float(noise_radius + baseline_support + approximation_support)
        )
        * amplification
        + float(additive_baseline + additive_approximation)
    )
    recomputed_center = _dot(weights, outcome)
    if not math.isclose(
        float(recomputed_center), float(center), rel_tol=1e-12, abs_tol=1e-12
    ):
        return _invalid(proof, "claimed center does not equal the recomputed weight outcome")
    if not math.isclose(recomputed, float(claimed_radius), rel_tol=1e-12, abs_tol=1e-12):
        return _invalid(proof, "claimed radius does not match the recomputed radius")
    center_value = float(center)
    interval = TargetInterval(center_value - recomputed, center_value + recomputed)
    return ProofVerification(
        valid=True,
        proof_type=type(proof).__name__,
        recomputed_interval=interval,
        proves_identification=(
            all(value == 0 for value in residual) and design_error == 0
        ),
        diagnostics={
            "exact_target_residual": residual,
            "recomputed_radius": recomputed,
            "recomputed_center": recomputed_center,
            "weight_norm": amplification,
            "artifact_hash": recomputed_hash,
        },
        required_failure_allocations=allocations or {},
    )


def _verify_scalar_bound(proof: ScalarBoundProof) -> ProofVerification:
    return _invalid(
        proof,
        "generic scalar bounds are diagnostic-only; use a data-bound typed proof",
    )


def _verify_temporal_bound(proof: TemporalBoundProof) -> ProofVerification:
    if proof.schema_version != _TEMPORAL_BOUND_SCHEMA:
        return _invalid(proof, f"unsupported temporal proof schema: {proof.schema_version!r}")
    if not _nonempty_metadata(
        proof.producer_version,
        proof.units,
        proof.premise,
        proof.provenance,
        proof.target_id,
    ):
        return _invalid(proof, "producer version, units, premise, and provenance are required")
    if not isinstance(proof.artifact_hash, str) or not _SHA256_PATTERN.fullmatch(
        proof.artifact_hash
    ):
        return _invalid(proof, "artifact hash must be a lowercase sha256 digest")
    try:
        recomputed_hash = temporal_bound_artifact_hash(proof)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        return _invalid(proof, str(exc))
    if proof.artifact_hash != recomputed_hash:
        return _invalid(proof, "artifact hash does not match the temporal proof payload")

    try:
        lags = _vector(proof.lags, "lags")
        responses = _vector(proof.responses, "responses")
        bandwidth = _fraction(proof.bandwidth)
        lower_mass_c = _fraction(proof.lower_mass_c)
        lower_mass_beta = _fraction(proof.lower_mass_beta)
        holder_L = _fraction(proof.holder_L)
        holder_alpha = _fraction(proof.holder_alpha)
        noise_scale = _fraction(proof.noise_scale)
        noise_delta = _fraction(proof.noise_failure_budget)
        baseline_error = _fraction(proof.baseline_error)
        declared_delta = _fraction(proof.declared_failure_budget)
        claimed_center = _fraction(proof.claimed_center)
        claimed_radius = _fraction(proof.claimed_radius)
        allocations = tuple(
            (name, _fraction(value)) for name, value in proof.failure_allocations
        )
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        return _invalid(proof, str(exc))

    if len(lags) != len(responses):
        return _invalid(proof, "lags and responses must have equal non-zero length")
    if isinstance(proof.n_units, bool) or not isinstance(proof.n_units, int) or proof.n_units <= 0:
        return _invalid(proof, "n_units must be a positive integer")
    if len(lags) > proof.n_units:
        return _invalid(
            proof,
            "the number of temporal rows cannot exceed n_units; the proof does not "
            "establish at most one effective contribution per independent unit",
        )
    positive = (bandwidth, lower_mass_c, lower_mass_beta, holder_alpha, noise_delta)
    if any(value <= 0 for value in positive):
        return _invalid(proof, "bandwidth, mass parameters, Holder alpha, and noise delta must be positive")
    nonnegative = (holder_L, noise_scale, baseline_error, claimed_radius)
    if any(value < 0 for value in nonnegative):
        return _invalid(proof, "Holder L, noise scale, baseline error, and radius must be non-negative")
    if any(value <= 0 for value in lags):
        return _invalid(proof, "lags must be strictly positive")
    if not (0 < noise_delta < 1 and 0 < declared_delta < 1):
        return _invalid(proof, "failure budgets must lie strictly between zero and one")

    component_names = [name for name, _ in allocations]
    if (
        not component_names
        or any(not isinstance(name, str) or not name.strip() for name in component_names)
        or len(component_names) != len(set(component_names))
        or any(value <= 0 or value >= 1 for _, value in allocations)
    ):
        return _invalid(proof, "failure allocations require unique names and deltas in (0, 1)")

    bandwidth_value = float(bandwidth)
    mass_probability = float(lower_mass_c) * bandwidth_value ** float(lower_mass_beta)
    if not math.isfinite(mass_probability) or not 0 < mass_probability <= 1:
        return _invalid(proof, "the declared one-scale lower mass must lie in (0, 1]")
    expected_mass = proof.n_units * mass_probability
    required_count = expected_mass / 2.0
    selected = [
        float(response)
        for lag, response in zip(lags, responses, strict=True)
        if lag <= bandwidth
    ]
    n_near_zero = len(selected)
    if n_near_zero == 0 or n_near_zero < required_count:
        return _invalid(proof, "near-zero count does not meet the declared availability threshold")

    count_failure = math.exp(-expected_mass / 8.0)
    total_failure = count_failure + float(noise_delta)
    if total_failure > float(declared_delta) + 1e-15:
        return _invalid(proof, "count and noise failure bounds exceed the declared budget")
    allocation_map = {name: float(value) for name, value in allocations}
    if len(allocation_map) != 2:
        return _invalid(proof, "temporal proof requires count and trace-noise allocations")
    count_allocation = float(allocations[0][1])
    noise_allocation = float(allocations[1][1])
    if count_allocation + 1e-15 < count_failure:
        return _invalid(proof, "count failure allocation is smaller than the recomputed bound")
    if not math.isclose(
        noise_allocation, float(noise_delta), rel_tol=1e-12, abs_tol=1e-15
    ):
        return _invalid(proof, "trace-noise allocation does not match its declared budget")
    if count_allocation + noise_allocation > float(declared_delta) + 1e-15:
        return _invalid(proof, "failure allocations exceed the declared temporal budget")

    center = sum(selected) / n_near_zero
    bias_radius = float(holder_L) * bandwidth_value ** float(holder_alpha)
    noise_radius = float(noise_scale) * math.sqrt(
        2.0 * math.log(2.0 / float(noise_delta)) / n_near_zero
    )
    radius = bias_radius + noise_radius + float(baseline_error)
    if not math.isclose(center, float(claimed_center), rel_tol=1e-12, abs_tol=1e-12):
        return _invalid(proof, "claimed center does not match the recomputed near-zero mean")
    if not math.isclose(radius, float(claimed_radius), rel_tol=1e-12, abs_tol=1e-12):
        return _invalid(proof, "claimed radius does not match the recomputed temporal radius")

    return ProofVerification(
        valid=True,
        proof_type=type(proof).__name__,
        recomputed_interval=TargetInterval(center - radius, center + radius),
        diagnostics={
            "artifact_hash": recomputed_hash,
            "n_near_zero": n_near_zero,
            "expected_near_zero_count": expected_mass,
            "required_near_zero_count": required_count,
            "count_failure_bound": count_failure,
            "noise_failure_budget": float(noise_delta),
            "failure_probability_bound": total_failure,
            "bias_radius": bias_radius,
            "noise_error_radius": noise_radius,
            "baseline_error": float(baseline_error),
            "recomputed_radius": radius,
        },
        required_failure_allocations=allocation_map,
    )


def _verify_temporal_design_bound(
    proof: TemporalDesignBoundProof,
) -> ProofVerification:
    if proof.schema_version != _TEMPORAL_DESIGN_BOUND_SCHEMA:
        return _invalid(
            proof,
            f"unsupported temporal design-bound schema: {proof.schema_version!r}",
        )
    if not _nonempty_metadata(
        proof.producer_version,
        proof.units,
        proof.premise,
        proof.provenance,
        proof.target_id,
    ):
        return _invalid(
            proof,
            "producer version, units, premise, provenance, and target id are required",
        )
    if not isinstance(proof.artifact_hash, str) or not _SHA256_PATTERN.fullmatch(
        proof.artifact_hash
    ):
        return _invalid(proof, "artifact hash must be a lowercase sha256 digest")
    try:
        recomputed_hash = temporal_design_bound_artifact_hash(proof)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        return _invalid(proof, str(exc))
    if proof.artifact_hash != recomputed_hash:
        return _invalid(
            proof,
            "artifact hash does not match the temporal design-bound payload",
        )
    allocations, allocation_reason = _failure_allocations(proof.failure_allocations)
    if allocation_reason:
        return _invalid(proof, allocation_reason)
    try:
        lags = _vector(proof.lags, "lags")
        responses = _vector(proof.responses, "responses")
        bandwidth = _fraction(proof.bandwidth)
        holder_L = _fraction(proof.holder_L)
        holder_alpha = _fraction(proof.holder_alpha)
        response_error = _fraction(proof.response_error_bound)
        claimed_center = _fraction(proof.claimed_center)
        claimed_radius = _fraction(proof.claimed_radius)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        return _invalid(proof, str(exc))
    if len(lags) != len(responses):
        return _invalid(proof, "lags and responses must have equal non-zero length")
    if (
        bandwidth <= 0
        or holder_L < 0
        or holder_alpha <= 0
        or response_error < 0
        or claimed_radius < 0
        or any(lag <= 0 for lag in lags)
    ):
        return _invalid(
            proof,
            "temporal design-bound parameters are outside their admissible ranges",
        )
    selected = [
        float(response)
        for lag, response in zip(lags, responses, strict=True)
        if lag <= bandwidth
    ]
    if not selected:
        return _invalid(proof, "the frozen temporal design selects no near-zero row")
    center = sum(selected) / len(selected)
    radius = (
        float(holder_L) * float(bandwidth) ** float(holder_alpha)
        + float(response_error) / math.sqrt(len(selected))
    )
    if not math.isclose(center, float(claimed_center), rel_tol=1e-12, abs_tol=1e-12):
        return _invalid(proof, "claimed center does not match the selected response mean")
    if not math.isclose(radius, float(claimed_radius), rel_tol=1e-12, abs_tol=1e-12):
        return _invalid(
            proof,
            "claimed radius does not match the temporal design-bound formula",
        )
    return ProofVerification(
        valid=True,
        proof_type=type(proof).__name__,
        recomputed_interval=TargetInterval(center - radius, center + radius),
        diagnostics={
            "artifact_hash": recomputed_hash,
            "n_selected": len(selected),
            "recomputed_center": center,
            "recomputed_radius": radius,
        },
        required_failure_allocations=allocations or {},
    )


def verify_proof(proof: ProofPayload) -> ProofVerification:
    """Verify one supported proof payload without consulting caller validity labels."""

    if isinstance(proof, ExactNullProof):
        return _verify_exact_null(proof)
    if isinstance(proof, SupportGapProof):
        return _verify_support_gap(proof)
    if isinstance(proof, LinearBoundProof):
        return _verify_linear_bound(proof)
    if isinstance(proof, ScalarBoundProof):
        return _verify_scalar_bound(proof)
    if isinstance(proof, AnchoredEnvelopeProof):
        return _verify_anchored_envelope(proof)
    if isinstance(proof, TemporalBoundProof):
        return _verify_temporal_bound(proof)
    if isinstance(proof, TemporalDesignBoundProof):
        return _verify_temporal_design_bound(proof)
    raise TypeError(f"unsupported proof payload: {type(proof).__name__}")
