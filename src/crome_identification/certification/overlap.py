"""Target-direct overlap certificates for exact and perturbed designs."""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from fractions import Fraction

import numpy as np
from scipy.optimize import minimize_scalar

from ..inverse.functionals import null_space_basis
from .types import (
    CertificateClaimType,
    CertificateScope,
    CertificationMode,
    ExactNullProof,
    FailureSemantics,
    LinearBoundProof,
    SetSubtype,
    TargetCertificate,
    TargetInterval,
)
from .verification import verify_proof


def _exact_number(value: float) -> str:
    """Serialize the binary-float value as an exact decimal proof input."""

    return repr(float(value))


def _linear_bound_proof(
    design: np.ndarray,
    outcome: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    theta_radius: float,
    design_error_bound: float,
    noise_radius: float,
    baseline_support_radius: float,
    approximation_support_radius: float,
    baseline_error: float,
    approximation_error: float,
    center: float,
    radius: float,
    failure_allocations: tuple[tuple[str, float], ...],
    target_id: str,
    provenance: str,
) -> LinearBoundProof:
    proof = LinearBoundProof(
        operator=tuple(tuple(_exact_number(value) for value in row) for row in design),
        outcome=tuple(_exact_number(value) for value in outcome),
        target=tuple(_exact_number(value) for value in target),
        weights=tuple(_exact_number(value) for value in weights),
        theta_radius=_exact_number(theta_radius),
        design_error_bound=_exact_number(design_error_bound),
        noise_radius=_exact_number(noise_radius),
        baseline_support_radius=_exact_number(baseline_support_radius),
        approximation_support_radius=_exact_number(approximation_support_radius),
        additive_baseline_error=_exact_number(baseline_error),
        additive_approximation_error=_exact_number(approximation_error),
        center=_exact_number(center),
        claimed_radius=_exact_number(radius),
        units="declared scalar-target units",
        uncertainty_definition=(
            "Euclidean parameter, operator, noise, baseline, and approximation budgets"
        ),
        provenance=provenance,
        schema_version="crome.linear-bound.v1",
        producer_version="crome-identification/0.1.0",
        artifact_hash="sha256:" + "0" * 64,
        failure_allocations=tuple(
            (name, _exact_number(delta)) for name, delta in failure_allocations
        ),
        target_id=target_id,
    )
    return replace(proof, artifact_hash=proof.recompute_artifact_hash())


def _proof_matches_operator(
    proof: ExactNullProof,
    design: np.ndarray,
    target: np.ndarray,
) -> bool:
    try:
        proof_design = np.asarray(
            [[float(Fraction(value)) for value in row] for row in proof.operator],
            dtype=float,
        )
        proof_target = np.asarray(
            [float(Fraction(value)) for value in proof.target],
            dtype=float,
        )
    except (TypeError, ValueError, ZeroDivisionError):
        return False
    return np.array_equal(proof_design, design) and np.array_equal(proof_target, target)


def _default_target_id(target: np.ndarray) -> str:
    encoded = "|".join(_exact_number(value) for value in target).encode("ascii")
    return f"linear-target-sha256:{hashlib.sha256(encoded).hexdigest()}"


def _scalar_target_row(C: np.ndarray, parameter_dim: int) -> np.ndarray:
    target = np.asarray(C, dtype=float)
    if target.ndim == 2 and target.shape[0] == 1:
        target = target.reshape(-1)
    if target.ndim != 1:
        raise ValueError("minimal prototype supports one scalar target")
    if target.shape[0] != parameter_dim:
        raise ValueError("C must match the design parameter dimension")
    if not np.all(np.isfinite(target)):
        raise ValueError("C must contain finite values")
    return target


def target_direct_weights(
    Ahat: np.ndarray,
    C: np.ndarray,
    *,
    rcond: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """Solve the dual equation ``Ahat.T w ~= C.T`` without estimating theta."""

    design = np.asarray(Ahat, dtype=float)
    if design.ndim != 2:
        raise ValueError("Ahat must be a two-dimensional matrix")
    if not np.all(np.isfinite(design)):
        raise ValueError("Ahat must contain finite values")
    target = _scalar_target_row(C, design.shape[1])
    weights = np.linalg.lstsq(design.T, target, rcond=rcond)[0]
    residual = target - weights @ design
    return weights, float(np.linalg.norm(residual, ord=2))


def certificate_radius(
    Ahat: np.ndarray,
    C: np.ndarray,
    weights: np.ndarray,
    *,
    theta_radius: float,
    design_error_bound: float,
    noise_radius: float,
    baseline_support_radius: float = 0.0,
    approximation_support_radius: float = 0.0,
) -> float:
    """Evaluate the fixed-weight rectangular-set certificate objective."""

    design = np.asarray(Ahat, dtype=float)
    if design.ndim != 2 or not np.all(np.isfinite(design)):
        raise ValueError("Ahat must be a finite two-dimensional matrix")
    target = _scalar_target_row(C, design.shape[1])
    weight = np.asarray(weights, dtype=float)
    if weight.ndim != 1 or weight.shape[0] != design.shape[0]:
        raise ValueError("weights must contain one finite value per design row")
    if not np.all(np.isfinite(weight)):
        raise ValueError("weights must contain one finite value per design row")
    theta_radius = _nonnegative_finite("theta_radius", theta_radius)
    design_error_bound = _nonnegative_finite(
        "design_error_bound", design_error_bound
    )
    noise_radius = _nonnegative_finite("noise_radius", noise_radius)
    baseline_support_radius = _nonnegative_finite(
        "baseline_support_radius", baseline_support_radius
    )
    approximation_support_radius = _nonnegative_finite(
        "approximation_support_radius", approximation_support_radius
    )
    residual = float(np.linalg.norm(target - weight @ design, ord=2))
    amplification = float(np.linalg.norm(weight, ord=2))
    support_scale = (
        theta_radius * design_error_bound
        + noise_radius
        + baseline_support_radius
        + approximation_support_radius
    )
    return float(theta_radius * residual + support_scale * amplification)


def certificate_optimized_weights(
    Ahat: np.ndarray,
    C: np.ndarray,
    *,
    theta_radius: float,
    design_error_bound: float,
    noise_radius: float,
    baseline_support_radius: float = 0.0,
    approximation_support_radius: float = 0.0,
    solver_tolerance: float = 1e-10,
) -> tuple[np.ndarray, dict[str, float | bool | str]]:
    """Minimize the convex target-certificate radius over linear weights.

    The primal is a two-cone epigraph problem.  A separately solved feasible
    dual supplies a numerical lower bound and an auditable optimality gap.  The
    returned fixed-weight radius remains valid even when the solver gap is not
    negligible.
    """

    design = np.asarray(Ahat, dtype=float)
    if design.ndim != 2 or not np.all(np.isfinite(design)):
        raise ValueError("Ahat must be a finite two-dimensional matrix")
    target = _scalar_target_row(C, design.shape[1])
    theta_radius = _nonnegative_finite("theta_radius", theta_radius)
    design_error_bound = _nonnegative_finite(
        "design_error_bound", design_error_bound
    )
    noise_radius = _nonnegative_finite("noise_radius", noise_radius)
    baseline_support_radius = _nonnegative_finite(
        "baseline_support_radius", baseline_support_radius
    )
    approximation_support_radius = _nonnegative_finite(
        "approximation_support_radius", approximation_support_radius
    )
    if not math.isfinite(solver_tolerance) or solver_tolerance <= 0.0:
        raise ValueError("solver_tolerance must be finite and positive")

    support_scale = (
        theta_radius * design_error_bound
        + noise_radius
        + baseline_support_radius
        + approximation_support_radius
    )
    least_squares, _ = target_direct_weights(design, target, rcond=solver_tolerance)
    zero = np.zeros(design.shape[0], dtype=float)

    def radius(weight: np.ndarray) -> float:
        return certificate_radius(
            design,
            target,
            weight,
            theta_radius=theta_radius,
            design_error_bound=design_error_bound,
            noise_radius=noise_radius,
            baseline_support_radius=baseline_support_radius,
            approximation_support_radius=approximation_support_radius,
        )

    if support_scale == 0.0:
        candidates = (least_squares, zero)
        weight = min(candidates, key=lambda candidate: (radius(candidate), np.linalg.norm(candidate)))
        primal = radius(weight)
        return weight, {
            "primal_radius": primal,
            "dual_lower_bound": primal,
            "duality_gap": 0.0,
            "solver_success": True,
            "solver_message": "closed minimal-norm zero-support case",
        }

    # A minimum-norm optimizer lies in Col(A).  Compact SVD coordinates reduce
    # the primal from one variable per observation row to rank(A) variables.
    left, singular_values, right_t = np.linalg.svd(design, full_matrices=False)
    sv_threshold = solver_tolerance * max(
        1.0, float(singular_values[0]) if singular_values.size else 1.0
    )
    rank = int(np.sum(singular_values > sv_threshold))
    left = left[:, :rank]
    singular_values = singular_values[:rank]
    right = right_t[:rank].T
    if rank:
        rhs = singular_values * (right.T @ target)
        squared = np.square(singular_values)
        log_scale = math.log(max(float(np.max(squared)), solver_tolerance))

        def weights_at(log_regularizer: float) -> np.ndarray:
            regularizer = math.exp(log_regularizer)
            coordinates = rhs / (squared + regularizer)
            return left @ coordinates

        scalar_result = minimize_scalar(
            lambda value: radius(weights_at(float(value))),
            bounds=(log_scale - 30.0, log_scale + 30.0),
            method="bounded",
            options={"xatol": solver_tolerance, "maxiter": 300},
        )
        optimized = weights_at(float(scalar_result.x))
    else:
        scalar_result = None
        optimized = zero
    candidates = (optimized, least_squares, zero)
    weight = min(candidates, key=lambda candidate: (radius(candidate), np.linalg.norm(candidate)))
    primal = radius(weight)

    residual = target - weight @ design
    residual_norm = float(np.linalg.norm(residual))
    if residual_norm > solver_tolerance and theta_radius > 0.0:
        dual_u = theta_radius * residual / residual_norm
    else:
        dual_u = np.zeros(design.shape[1], dtype=float)
    scales = [1.0]
    norm_u = float(np.linalg.norm(dual_u))
    norm_au = float(np.linalg.norm(design @ dual_u))
    if norm_u > theta_radius and norm_u > 0.0:
        scales.append(theta_radius / norm_u)
    if norm_au > support_scale and norm_au > 0.0:
        scales.append(support_scale / norm_au)
    dual_u *= min(scales)
    dual_bound = max(0.0, float(target @ dual_u))
    dual_bound = min(dual_bound, primal)
    gap = max(0.0, primal - dual_bound)
    return weight, {
        "primal_radius": primal,
        "dual_lower_bound": dual_bound,
        "duality_gap": gap,
        "solver_success": bool(scalar_result is None or scalar_result.success),
        "solver_message": (
            "rank-zero closed solution"
            if scalar_result is None
            else str(scalar_result.message)
        ),
    }


def _nonnegative_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _target_changing_null_witness(
    design: np.ndarray,
    target: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray | None:
    basis = null_space_basis(design, tol=tolerance)
    if basis.shape[1] == 0:
        return None
    changes = target @ basis
    index = int(np.argmax(np.abs(changes)))
    if abs(float(changes[index])) <= tolerance * max(1.0, float(np.linalg.norm(target))):
        return None
    witness = basis[:, index]
    if float(target @ witness) < 0:
        witness = -witness
    return witness


def certify_overlap_target(
    Ahat: np.ndarray,
    y: np.ndarray,
    C: np.ndarray,
    *,
    noise_radius: float,
    design_error_bound: float,
    theta_radius: float,
    baseline_error: float = 0.0,
    approximation_error: float = 0.0,
    baseline_support_radius: float = 0.0,
    approximation_support_radius: float = 0.0,
    weight_strategy: str = "certificate_optimal",
    exact_design: bool,
    exact_null_proof: ExactNullProof | None = None,
    algebraic_tolerance: float = 1e-10,
    budget_components: tuple[str, ...] = (),
    component_delta: float | None = None,
    deterministic_uncertainty: bool = False,
    target_id: str | None = None,
    provenance: str = "",
) -> TargetCertificate:
    """Construct a scalar target certificate from a direct dual estimator.

    The default ``certificate_optimal`` strategy minimizes the declared
    certificate radius.  ``least_squares`` remains available as an explicit
    baseline/ablation strategy.

    For estimated designs the reported radius is

    ``(rho + kappa * eps_A) * R_theta + kappa * eps_noise
       + eps_baseline + eps_approx``.

    This is a conditional target-error budget, not an exact row-space claim.
    """

    design = np.asarray(Ahat, dtype=float)
    outcome = np.asarray(y, dtype=float)
    if design.ndim != 2:
        raise ValueError("Ahat must be a two-dimensional matrix")
    if outcome.ndim != 1 or outcome.shape[0] != design.shape[0]:
        raise ValueError("y must be a vector with one entry per design row")
    if not np.all(np.isfinite(design)) or not np.all(np.isfinite(outcome)):
        raise ValueError("Ahat and y must contain finite values")
    target = _scalar_target_row(C, design.shape[1])
    resolved_target_id = _default_target_id(target) if target_id is None else target_id.strip()
    if not resolved_target_id:
        raise ValueError("target_id must be non-empty when supplied")

    noise_radius = _nonnegative_finite("noise_radius", noise_radius)
    design_error_bound = _nonnegative_finite("design_error_bound", design_error_bound)
    theta_radius = _nonnegative_finite("theta_radius", theta_radius)
    baseline_error = _nonnegative_finite("baseline_error", baseline_error)
    approximation_error = _nonnegative_finite(
        "approximation_error", approximation_error
    )
    baseline_support_radius = _nonnegative_finite(
        "baseline_support_radius", baseline_support_radius
    )
    approximation_support_radius = _nonnegative_finite(
        "approximation_support_radius", approximation_support_radius
    )
    if weight_strategy not in {"least_squares", "certificate_optimal"}:
        raise ValueError("weight_strategy must be least_squares or certificate_optimal")
    if not math.isfinite(algebraic_tolerance) or algebraic_tolerance <= 0:
        raise ValueError("algebraic_tolerance must be finite and positive")
    if exact_design and design_error_bound != 0.0:
        raise ValueError("design_error_bound must be zero when exact_design is true")
    failure_allocations: tuple[tuple[str, float], ...] = ()
    if component_delta is not None:
        if not math.isfinite(component_delta) or not 0.0 < component_delta < 1.0:
            raise ValueError("component_delta must lie strictly between zero and one")
        if not budget_components or any(not component.strip() for component in budget_components):
            raise ValueError("component_delta requires named budget_components")
        failure_allocations = tuple(
            (component, float(component_delta)) for component in budget_components
        )

    optimization_diagnostics: dict[str, float | bool | str] = {}
    if weight_strategy == "certificate_optimal":
        weights, optimization_diagnostics = certificate_optimized_weights(
            design,
            target,
            theta_radius=theta_radius,
            design_error_bound=design_error_bound,
            noise_radius=noise_radius,
            baseline_support_radius=baseline_support_radius,
            approximation_support_radius=approximation_support_radius,
            solver_tolerance=algebraic_tolerance,
        )
        target_residual = float(np.linalg.norm(target - weights @ design, ord=2))
    else:
        weights, target_residual = target_direct_weights(
            design,
            target,
            rcond=algebraic_tolerance,
        )
    amplification = float(np.linalg.norm(weights, ord=2))
    target_scale = max(1.0, float(np.linalg.norm(target, ord=2)))
    is_algebraically_identified = target_residual <= algebraic_tolerance * target_scale

    mode = (
        CertificationMode.DESIGN_KNOWN
        if exact_design
        else CertificationMode.ASSUMPTION_CONDITIONAL
    )
    has_precision_uncertainty = any(
        value > 0.0
        for value in (
            noise_radius,
            design_error_bound,
            baseline_error,
            approximation_error,
            baseline_support_radius,
            approximation_support_radius,
        )
    )
    scope = (
        CertificateScope.POPULATION_EXACT
        if exact_design and not has_precision_uncertainty
        else CertificateScope.ASSUMPTION_CONDITIONAL
    )
    if scope is CertificateScope.ASSUMPTION_CONDITIONAL:
        if deterministic_uncertainty:
            if component_delta is not None or budget_components:
                raise ValueError(
                    "deterministic_uncertainty cannot carry probability-ledger components"
                )
            failure_semantics = FailureSemantics.DETERMINISTIC_BOUND
        else:
            if component_delta is None:
                raise ValueError(
                    "probabilistic overlap uncertainty requires component_delta and "
                    "named budget_components; set deterministic_uncertainty=True only "
                    "for externally guaranteed norm bounds"
                )
            failure_semantics = FailureSemantics.PROBABILITY_LEDGER
    else:
        if component_delta is not None or budget_components:
            raise ValueError("population-exact overlap evidence cannot spend failure budget")
        failure_semantics = FailureSemantics.DETERMINISTIC_BOUND
    diagnostics = {
        "target_residual": target_residual,
        "target_amplification": amplification,
        "design_error_bound": design_error_bound,
        "theta_radius": theta_radius,
        "noise_radius": noise_radius,
        "baseline_error": baseline_error,
        "approximation_error": approximation_error,
        "baseline_support_radius": baseline_support_radius,
        "approximation_support_radius": approximation_support_radius,
        "weight_strategy": weight_strategy,
        "dual_weights": weights.tolist(),
        **optimization_diagnostics,
    }

    if exact_design and not is_algebraically_identified:
        witness = _target_changing_null_witness(
            design,
            target,
            tolerance=algebraic_tolerance,
        )
        if witness is None:
            return TargetCertificate(
                source="overlap",
                mode=mode,
                scope=scope,
                valid=False,
                assumptions=("exact fixed design",),
                diagnostics=diagnostics,
                reason="numerical residual is nonzero but no stable null witness was found",
                claim_type=CertificateClaimType.NUMERICAL_DIAGNOSTIC,
                budget_components=budget_components,
                provenance=provenance,
            )
        diagnostics["null_target_variation"] = float(target @ witness)
        diagnostics["candidate_null_witness"] = witness.tolist()
        if exact_null_proof is None:
            return TargetCertificate(
                source="overlap",
                mode=mode,
                scope=CertificateScope.FINITE_SAMPLE_ONLY,
                valid=True,
                assumptions=("floating-point design diagnostic",),
                diagnostics=diagnostics,
                reason=(
                    "floating SVD found a target-changing near-null direction; "
                    "no exact proof payload was supplied"
                ),
                claim_type=CertificateClaimType.NUMERICAL_DIAGNOSTIC,
                budget_components=budget_components,
                provenance=provenance,
            )
        verification = verify_proof(exact_null_proof)
        if (
            not verification.valid
            or exact_null_proof.target_id != resolved_target_id
            or not _proof_matches_operator(exact_null_proof, design, target)
        ):
            return TargetCertificate(
                source="overlap",
                mode=mode,
                scope=CertificateScope.POPULATION_EXACT,
                valid=False,
                assumptions=("exact fixed design",),
                diagnostics={
                    **diagnostics,
                    "exact_proof_verification": verification.reason,
                },
                reason="exact null proof is invalid or does not match A and c",
                claim_type=CertificateClaimType.NUMERICAL_DIAGNOSTIC,
                budget_components=budget_components,
                provenance=provenance,
            )
        exact_witness = tuple(float(Fraction(value)) for value in exact_null_proof.witness)
        return TargetCertificate(
            source="overlap",
            mode=mode,
            scope=CertificateScope.POPULATION_EXACT,
            valid=True,
            null_witness=exact_witness,
            assumptions=("exact fixed design", "unrestricted parameter direction"),
            diagnostics=diagnostics,
            reason="exact design admits a target-changing null direction",
            claim_type=CertificateClaimType.EXACT_WITNESS,
            provenance=provenance,
            proof=exact_null_proof,
            target_id=resolved_target_id,
        )

    estimate = float(weights @ outcome)
    radius = float(
        certificate_radius(
            design,
            target,
            weights,
            theta_radius=theta_radius,
            design_error_bound=design_error_bound,
            noise_radius=noise_radius,
            baseline_support_radius=baseline_support_radius,
            approximation_support_radius=approximation_support_radius,
        )
        + baseline_error
        + approximation_error
    )
    interval = TargetInterval(estimate - radius, estimate + radius)
    proof = _linear_bound_proof(
        design,
        outcome,
        target,
        weights,
        theta_radius=theta_radius,
        design_error_bound=design_error_bound,
        noise_radius=noise_radius,
        baseline_support_radius=baseline_support_radius,
        approximation_support_radius=approximation_support_radius,
        baseline_error=baseline_error,
        approximation_error=approximation_error,
        center=estimate,
        radius=radius,
        failure_allocations=failure_allocations,
        target_id=resolved_target_id,
        provenance=provenance,
    )
    assumptions = (
        ("exact fixed design", "bounded outcome-noise norm")
        if exact_design
        else (
            "operator-norm design error bound",
            "bounded parameter norm",
            "bounded outcome-noise norm",
        )
    )
    return TargetCertificate(
        source="overlap",
        mode=mode,
        scope=scope,
        valid=True,
        feasible_set=interval,
        point_estimate=estimate,
        error_radius=radius,
        assumptions=assumptions,
        diagnostics=diagnostics,
        reason=(
            "exact target-direct overlap certificate"
            if exact_design
            else "assumption-conditional robust target-error certificate"
        ),
        claim_type=(
            CertificateClaimType.STRUCTURAL_EVIDENCE
            if scope is CertificateScope.POPULATION_EXACT
            else CertificateClaimType.ROBUST_TARGET_BOUND
        ),
        budget_components=budget_components,
        failure_allocations=failure_allocations,
        failure_semantics=failure_semantics,
        provenance=provenance,
        proof=proof,
        set_subtype=(
            SetSubtype.CONFIDENCE_OUTER
            if exact_design
            else SetSubtype.ASSUMPTION_CONDITIONAL
        ),
        target_id=resolved_target_id,
    )
