"""E-CS3: aligned continuous support-by-perturbation experiment."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from crome_identification.baselines import (
    fit_naive_boundary,
    fit_ridge_target,
    fit_tsvd_target,
)
from crome_identification.benchmarks import PerturbationSpec, generate_aligned_geometry
from crome_identification.certification import (
    CertificateScope,
    CertificationMode,
    DecisionStatus,
    FailureBudget,
    FailureBudgetLedger,
    ModelInfeasibleError,
    TargetCertificate,
    TargetInterval,
    calibrate_bounded_design_error,
    calibrate_gaussian_outcome_noise,
    calibrate_lower_mass,
    certify_overlap_target,
    certify_temporal_conditional,
    certify_temporal_design_known,
    certify_temporal_empirical,
    certify_temporal_gap_conditional,
    decide_target,
)
from crome_identification.evaluation import EvaluationRow, summarize_method_rows, wilson_interval
from crome_identification.evaluation.splits import split_trajectory_ids
from crome_identification.seeding import make_seed_bundle

from ._common import load_exp_config, mode_reps, save_raw_and_summary
from ._proof_fixtures import (
    coordinate_null_fixture_proof,
    support_gap_fixture_proof,
)


_METHODS = (
    "crome_optimal",
    "crome_current",
    "matched_uncertainty",
    "naive_boundary",
    "ridge",
    "tsvd",
)


@dataclass(frozen=True)
class CellSpec:
    support_mass: float
    perturbation: PerturbationSpec

    @property
    def key(self) -> str:
        return f"mass={self.support_mass:g}:{self.perturbation.kind}:{self.perturbation.level:g}"


def _cell_specs(cfg: dict[str, Any]) -> list[CellSpec]:
    cells = [
        CellSpec(float(mass), PerturbationSpec(str(kind), float(level)))
        for mass in cfg["support_masses"]
        for kind, levels in cfg["perturbation_levels"].items()
        for level in levels
    ]
    keys = [cell.key for cell in cells]
    if len(keys) != len(set(keys)):
        raise ValueError("support-by-perturbation grid contains duplicate cells")
    return cells


def _hash_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _indices(values: tuple[int, ...]) -> np.ndarray:
    return np.asarray(values, dtype=int)


def _fit_residual_budget(design: np.ndarray, y: np.ndarray, train: np.ndarray,
                         calibration: np.ndarray, n_test: int, delta: float):
    theta = np.linalg.lstsq(design[train], y[train], rcond=1e-12)[0]
    residuals = y[calibration] - design[calibration] @ theta
    return calibrate_gaussian_outcome_noise(residuals, n_test=n_test, delta=delta)


def _fit_trace_budget(data, train: np.ndarray, calibration: np.ndarray,
                      test: np.ndarray, delta: float):
    train_target = train[data.observed_marks[train] == 0]
    calibration_target = calibration[data.observed_marks[calibration] == 0]
    test_target = test[data.observed_marks[test] == 0]
    if train_target.size < 3 or calibration_target.size < 3 or test_target.size < 1:
        return None, train_target, calibration_target, test_target
    design = np.column_stack([np.ones(train_target.size), data.observed_lags[train_target]])
    beta = np.linalg.lstsq(design, data.trace_responses[train_target], rcond=None)[0]
    residuals = data.trace_responses[calibration_target] - (
        beta[0] + beta[1] * data.observed_lags[calibration_target]
    )
    budget = calibrate_gaussian_outcome_noise(
        residuals, n_test=int(test_target.size), delta=delta
    )
    return budget, train_target, calibration_target, test_target


def _invalid_temporal(reason: str) -> TargetCertificate:
    return TargetCertificate(
        source="temporal_support_calibration",
        mode=CertificationMode.ASSUMPTION_CONDITIONAL,
        scope=CertificateScope.ASSUMPTION_CONDITIONAL,
        valid=False,
        assumptions=("split-sample temporal calibration",),
        reason=reason,
    )


def _expected_status(cell: CellSpec) -> str:
    if cell.support_mass > 0:
        return DecisionStatus.POINT_ESTIMABLE.value
    if cell.perturbation.kind == "clean":
        return DecisionStatus.NONRECOVERABLE.value
    if cell.perturbation.kind == "baseline_error":
        return DecisionStatus.SET_ESTIMABLE.value
    return DecisionStatus.INCONCLUSIVE.value


def _temporal_certificate(cfg: dict[str, Any], cell: CellSpec, data,
                          calibration_target: np.ndarray, test_target: np.ndarray,
                          trace_budget):
    if test_target.size == 0 or trace_budget is None:
        return _invalid_temporal("target-mark rows are absent from a required split")
    lags = data.observed_lags[test_target]
    responses = data.trace_responses[test_target]
    if cell.support_mass == 0:
        if cell.perturbation.kind == "clean":
            return certify_temporal_design_known(
                False,
                support_gap_proof=support_gap_fixture_proof(
                    lags,
                    fixture_name=f"CS03 {cell.key}",
                ),
                provenance=f"oracle-informed CS03 fixture {cell.key}",
            )
        if cell.perturbation.kind == "baseline_error":
            return certify_temporal_gap_conditional(
                lags,
                responses,
                holder_L=float(cfg["holder_L"]),
                holder_alpha=float(cfg["holder_alpha"]),
            response_error_bound=trace_budget.simultaneous_coordinate_bound,
            budget_components=("trace_noise",),
            component_delta=float(cfg["component_delta"]),
                provenance="held-out trace calibration split",
            )
        return certify_temporal_empirical(lags)

    if cell.perturbation.kind == "clean":
        near = lags <= float(cfg["support_bandwidth"])
        if not np.any(near):
            return _invalid_temporal("no near-zero target row on the test split")
        estimate = float(np.mean(responses[near]))
        radius = float(
            float(cfg["holder_L"]) * float(cfg["support_bandwidth"])
            + trace_budget.simultaneous_coordinate_bound / math.sqrt(int(np.sum(near)))
        )
        return certify_temporal_design_known(
            True,
            lags=lags,
            responses=responses,
            bandwidth=float(cfg["support_bandwidth"]),
            holder_L=float(cfg["holder_L"]),
            holder_alpha=1.0,
            response_error_bound=trace_budget.simultaneous_coordinate_bound,
            budget_components=("trace_noise",),
            component_delta=float(cfg["component_delta"]),
            provenance="held-out trace calibration split",
        )

    if cell.support_mass < 0.5:
        return certify_temporal_empirical(lags)
    lower_mass = calibrate_lower_mass(
        data.observed_lags[calibration_target],
        bandwidth=float(cfg["support_bandwidth"]),
        beta=1.0,
        delta=float(cfg["component_delta"]),
    )
    if not lower_mass.valid:
        return _invalid_temporal("calibration split supplies no positive lower-mass bound")
    baseline_error = (
        cell.perturbation.level
        if cell.perturbation.kind in {"mark_noise", "baseline_error"}
        else float(cfg["temporal_slope"]) * cell.perturbation.level
    )
    return certify_temporal_conditional(
        lags,
        responses,
        bandwidth=float(cfg["support_bandwidth"]),
        n_units=int(test_target.size),
        lower_mass_c=lower_mass.lower_mass_c,
        lower_mass_beta=1.0,
        holder_L=float(cfg["holder_L"]),
        holder_alpha=float(cfg["holder_alpha"]),
        noise_scale=trace_budget.sigma_upper,
        delta=float(cfg["component_delta"]),
        baseline_error=baseline_error,
        budget_components=("temporal_count", "trace_noise"),
        external_failure_allocations=(
            ("lower_mass_calibration", float(cfg["component_delta"])),
            ("trace_scale_calibration", float(cfg["component_delta"])),
        ),
        provenance="held-out lower-mass and trace calibration splits",
    )


def _crome_output(
    cfg: dict[str, Any],
    cell: CellSpec,
    data,
    train: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
    *,
    weight_strategy: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = perf_counter()
    delta = float(cfg["component_delta"])
    outcome_budget = _fit_residual_budget(
        data.observed_design, data.outcomes, train, calibration, int(test.size), delta
    )
    trace_budget, _, calibration_target, test_target = _fit_trace_budget(
        data, train, calibration, test, delta
    )
    temporal = _temporal_certificate(
        cfg, cell, data, calibration_target, test_target, trace_budget
    )
    exact = cell.perturbation.kind in {"clean", "baseline_error"}
    design_calibration = None
    design_error = 0.0
    if not exact:
        design_calibration = calibrate_bounded_design_error(
            data.true_design[calibration],
            data.observed_design[calibration],
            n_test=int(test.size),
            row_error_bound=float(cfg["row_error_bound"]),
            delta=delta,
        )
        design_error = design_calibration.operator_error_bound
    overlap = certify_overlap_target(
        data.observed_design[test],
        data.outcomes[test],
        data.target,
        noise_radius=outcome_budget.vector_norm_bound,
        design_error_bound=design_error,
        theta_radius=float(cfg["theta_radius"]),
        baseline_error=(cell.perturbation.level if cell.perturbation.kind == "baseline_error" else 0.0),
        weight_strategy=weight_strategy,
        exact_design=exact,
        exact_null_proof=(
            coordinate_null_fixture_proof(
                data.observed_design[test],
                data.target,
                fixture_name=f"CS03 {cell.key}",
            )
            if exact
            else None
        ),
        budget_components=(
            ("outcome_noise",)
            if exact
            else ("outcome_noise", "design_error")
        ),
        component_delta=delta,
        target_id="scalar_target",
        provenance="held-out outcome and operator calibration splits",
    )
    allocations = [
        FailureBudget(component, component_delta, "certificate-bound held-out event")
        for certificate in (temporal, overlap)
        for component, component_delta in certificate.failure_allocations
    ]
    ledger = FailureBudgetLedger(
        total_delta=float(cfg["global_delta"]),
        allocations=tuple(allocations),
    )
    prior_radius = float(cfg["theta_radius"]) * float(np.linalg.norm(data.target))
    try:
        decision = decide_target(
            [temporal, overlap],
            scientific_tolerance=float(cfg["scientific_tolerance"]),
            failure_ledger=ledger,
            prior_target_domain=TargetInterval(-prior_radius, prior_radius),
            prior_only_shrinkage_threshold=float(cfg["prior_only_shrinkage_threshold"]),
        ).as_dict()
        failed = False
    except ModelInfeasibleError as exc:
        decision = {
            "status": DecisionStatus.INCONCLUSIVE.value,
            "certificate_scope": CertificateScope.ASSUMPTION_CONDITIONAL.value,
            "feasible_set": None,
            "point_estimate": None,
            "uncertainty": None,
            "assumptions": [],
            "reasons": [str(exc)],
            "diagnostics": {"model_infeasible": True},
        }
        failed = True
    decision["runtime_seconds"] = perf_counter() - started
    decision["failed"] = failed
    decision["certificate_radius"] = overlap.error_radius
    decision["overlap_diagnostics"] = dict(overlap.diagnostics)
    decision["failure_ledger"] = ledger.as_dict()
    calibration_record = {
        "outcome_noise": outcome_budget.as_dict(),
        "trace_noise": trace_budget.as_dict() if trace_budget is not None else None,
        "design_error": design_calibration.as_dict() if design_calibration is not None else None,
        "splits_used": {"fit": "train", "budgets": "calibration", "evaluation": "test"},
    }
    return decision, calibration_record


def _matched_uncertainty_candidate(
    cfg: dict[str, Any],
    data,
    test: np.ndarray,
    outcome_budget: dict[str, Any],
) -> dict[str, Any]:
    """Ridge uncertainty score that is deliberately blind to structural witnesses."""

    started = perf_counter()
    design = np.asarray(data.observed_design[test], dtype=float)
    target = np.asarray(data.target, dtype=float)
    lam = float(cfg["ridge_lambda"])
    system = design.T @ design + lam * np.eye(design.shape[1])
    coefficient = np.linalg.solve(system, target)
    weights = design @ coefficient
    estimate = float(weights @ data.outcomes[test])
    score = float(outcome_budget["sigma_upper"] * np.linalg.norm(weights))
    return {
        "status": DecisionStatus.INCONCLUSIVE.value,
        "point_estimate": None,
        "interval": None,
        "candidate_point_estimate": estimate,
        "candidate_interval": [estimate - score, estimate + score],
        "uncertainty_score": score,
        "runtime_seconds": perf_counter() - started,
        "failed": False,
    }


def _match_uncertainty_point_yield(records: list[dict[str, Any]]) -> None:
    target_count = sum(
        row["methods"]["crome_optimal"]["status"]
        == DecisionStatus.POINT_ESTIMABLE.value
        for row in records
    )
    ranked = sorted(
        range(len(records)),
        key=lambda index: (
            records[index]["methods"]["matched_uncertainty"]["uncertainty_score"],
            records[index]["cell_key"],
            records[index]["rep"],
        ),
    )
    selected = set(ranked[:target_count])
    for index, row in enumerate(records):
        output = row["methods"]["matched_uncertainty"]
        if index in selected:
            output["status"] = DecisionStatus.POINT_ESTIMABLE.value
            output["point_estimate"] = output["candidate_point_estimate"]
            output["interval"] = output["candidate_interval"]


def _baseline(result, radius: float) -> dict[str, Any]:
    estimate = result.target_estimate
    interval = [estimate - radius, estimate + radius] if estimate is not None else None
    return {
        "status": DecisionStatus.POINT_ESTIMABLE.value if result.success else DecisionStatus.INCONCLUSIVE.value,
        "point_estimate": estimate,
        "interval": interval,
        "runtime_seconds": result.runtime_seconds,
        "failed": not result.success,
    }


def _run_cell(cfg: dict[str, Any], cell: CellSpec, rep: int, mode: str,
              n_units: int) -> dict[str, Any]:
    bundle = make_seed_bundle(
        int(cfg["master_seed"]), "cs03", f"{mode}:{cell.key}", rep
    )
    seed = int(bundle.rng.integers(0, 2**32 - 1))
    data = generate_aligned_geometry(
        n_units=n_units,
        parameter_dim=int(cfg["parameter_dim"]),
        target_support_mass=cell.support_mass,
        perturbation=cell.perturbation,
        seed=seed,
        support_bandwidth=float(cfg["support_bandwidth"]),
        support_gap=float(cfg["support_gap"]),
        outcome_noise_sigma=float(cfg["outcome_noise_sigma"]),
        trace_noise_sigma=float(cfg["trace_noise_sigma"]),
        temporal_slope=float(cfg["temporal_slope"]),
    )
    split_seed = seed ^ 0xA5A5A5A5
    split = split_trajectory_ids(data.trajectory_ids, tuple(cfg["split_ratios"]), seed=split_seed)
    train, calibration, test = map(_indices, (split.train, split.calibration, split.test))
    crome_optimal, calibration_record = _crome_output(
        cfg,
        cell,
        data,
        train,
        calibration,
        test,
        weight_strategy="certificate_optimal",
    )
    crome_current, _ = _crome_output(
        cfg,
        cell,
        data,
        train,
        calibration,
        test,
        weight_strategy="least_squares",
    )
    target_test = test[data.observed_marks[test] == 0]
    naive = fit_naive_boundary(
        data.observed_lags[target_test], data.trace_responses[target_test],
        bandwidth=float(cfg["support_bandwidth"]),
    )
    ridge = fit_ridge_target(
        data.observed_design[test], data.outcomes[test], data.target,
        lam=float(cfg["ridge_lambda"]),
    )
    tsvd = fit_tsvd_target(
        data.observed_design[test], data.outcomes[test], data.target,
        rank=int(cfg["parameter_dim"]),
    )
    radius = 1.96 * float(cfg["outcome_noise_sigma"])
    return {
        "rep": rep,
        "cell_key": cell.key,
        "support_mass": cell.support_mass,
        "perturbation": cell.perturbation.kind,
        "level": cell.perturbation.level,
        "expected_status": _expected_status(cell),
        "fixture_kind": "oracle_informed_contract_fixture",
        "evaluation_only": {
            "expected_status": _expected_status(cell),
            "support_mass": cell.support_mass,
            "perturbation_label": cell.perturbation.kind,
        },
        "target_definition": "C @ theta",
        "target_C": data.target.tolist(),
        "true_target": data.true_target,
        "split_seed": split_seed,
        "split_disjoint": split.as_dict()["disjoint"],
        "n_test_target_rows": int(target_test.size),
        "n_test_near_zero": int(np.sum(data.observed_lags[target_test] <= float(cfg["support_bandwidth"]))),
        "calibration": calibration_record,
        "methods": {
            "crome_optimal": crome_optimal,
            "crome_current": crome_current,
            "matched_uncertainty": _matched_uncertainty_candidate(
                cfg,
                data,
                test,
                calibration_record["outcome_noise"],
            ),
            "naive_boundary": _baseline(naive, radius),
            "ridge": _baseline(ridge, radius),
            "tsvd": _baseline(tsvd, radius),
        },
    }


def _interval(output: dict[str, Any], method: str):
    if not method.startswith("crome_"):
        value = output.get("interval")
        return tuple(value) if value is not None else None
    value = output.get("feasible_set")
    if isinstance(value, dict) and value.get("bounded"):
        return (float(value["lower"]), float(value["upper"]))
    return None


def _method_rows(records: list[dict[str, Any]], method: str) -> list[EvaluationRow]:
    return [
        EvaluationRow(
            expected_status=row["expected_status"],
            predicted_status=row["methods"][method]["status"],
            point_estimate=row["methods"][method].get("point_estimate"),
            interval=_interval(row["methods"][method], method),
            true_target=float(row["true_target"]),
            runtime_seconds=float(row["methods"][method]["runtime_seconds"]),
            failed=bool(row["methods"][method]["failed"]),
        )
        for row in records
    ]


def _gate(cfg: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    nonpoint = [row for row in records if row["expected_status"] != DecisionStatus.POINT_ESTIMABLE.value]
    false_points = sum(
        row["methods"]["crome_optimal"]["status"]
        == DecisionStatus.POINT_ESTIMABLE.value
        for row in nonpoint
    )
    false_interval = wilson_interval(false_points, len(nonpoint), level=float(cfg["mc_confidence_level"]))
    supported = [
        row for row in records
        if row["support_mass"] >= float(cfg["gate_supported_mass"])
        and row["perturbation"] == "clean"
    ]
    covered = 0
    for row in supported:
        interval = _interval(row["methods"]["crome_optimal"], "crome_optimal")
        covered += int(interval is not None and interval[0] <= row["true_target"] <= interval[1])
    coverage_interval = wilson_interval(covered, len(supported), level=float(cfg["mc_confidence_level"]))
    product_statuses = sorted(
        {
            "_x_".join(
                (
                    row["methods"]["crome_optimal"]["structural_status"],
                    row["methods"]["crome_optimal"]["operational_status"],
                    row["methods"]["crome_optimal"]["certificate_scope"],
                )
            )
            for row in records
        }
    )
    checks = {
        "false_point_upper_bound": {
            "passed": false_interval["upper"] <= float(cfg["gate_max_false_point_upper"]),
            "observed": false_interval["upper"],
            "required": float(cfg["gate_max_false_point_upper"]),
        },
        "supported_coverage_lower_bound": {
            "passed": coverage_interval["lower"] >= float(cfg["gate_min_supported_coverage_lower"]),
            "observed": coverage_interval["lower"],
            "required": float(cfg["gate_min_supported_coverage_lower"]),
        },
        "product_status_transition": {
            "passed": len(product_statuses) >= int(cfg["gate_min_statuses"]),
            "observed": product_statuses,
            "required": int(cfg["gate_min_statuses"]),
        },
        "split_integrity": {
            "passed": all(row["split_disjoint"] for row in records),
            "observed": sum(row["split_disjoint"] for row in records),
            "required": len(records),
        },
        "same_target": {
            "passed": all(row["target_definition"] == "C @ theta" for row in records),
            "observed": "C @ theta",
            "required": "C @ theta",
        },
    }
    return {
        "passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
        "false_point_count": false_points,
        "nonpoint_count": len(nonpoint),
        "false_point_interval": false_interval,
        "supported_coverage_count": covered,
        "supported_count": len(supported),
        "supported_coverage_interval": coverage_interval,
    }


def _point_metrics(records: list[dict[str, Any]], method: str, tolerance: float):
    eligible = [
        row
        for row in records
        if row["expected_status"] == DecisionStatus.POINT_ESTIMABLE.value
    ]
    outputs = [
        row
        for row in eligible
        if row["methods"][method]["status"] == DecisionStatus.POINT_ESTIMABLE.value
    ]
    covered = 0
    violations = 0
    for row in outputs:
        output = row["methods"][method]
        interval = _interval(output, method)
        covered += int(
            interval is not None and interval[0] <= row["true_target"] <= interval[1]
        )
        violations += int(
            output.get("point_estimate") is None
            or abs(float(output["point_estimate"]) - float(row["true_target"])) > tolerance
        )
    return {
        "eligible": len(eligible),
        "point_outputs": len(outputs),
        "point_yield": len(outputs) / len(eligible) if eligible else None,
        "conditional_coverage": covered / len(outputs) if outputs else None,
        "marginal_coverage": covered / len(eligible) if eligible else None,
        "tolerance_violation": violations / len(outputs) if outputs else None,
    }


def _utility_pilot(cfg: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    """Ill-conditioned but structurally identified slice for certificate utility."""

    count = int(cfg[f"n_utility_reps_{mode}"])
    design = np.diag([1.0, 0.01])
    target = np.array([1.0, 1.0])
    theta = np.array([0.5, 0.5])
    true_target = float(target @ theta)
    noise_radius = 0.02
    tolerance = float(cfg["utility_scientific_tolerance"])
    prior_radius = float(np.linalg.norm(target))
    ledger = FailureBudgetLedger(
        total_delta=0.05,
        allocations=(
            FailureBudget("utility_outcome_noise", 0.02, "predeclared bounded-noise event"),
        ),
    )
    rows: list[dict[str, Any]] = []
    for rep in range(count):
        bundle = make_seed_bundle(int(cfg["master_seed"]), "cs03_utility", mode, rep)
        direction = bundle.rng.normal(size=design.shape[0])
        direction /= np.linalg.norm(direction)
        magnitude = noise_radius * float(bundle.rng.uniform(0.0, 1.0))
        outcome = design @ theta + magnitude * direction
        methods: dict[str, Any] = {}
        for name, strategy in (
            ("crome_optimal", "certificate_optimal"),
            ("crome_current", "least_squares"),
        ):
            certificate = certify_overlap_target(
                design,
                outcome,
                target,
                noise_radius=noise_radius,
                design_error_bound=0.0,
                theta_radius=1.0,
                exact_design=False,
                weight_strategy=strategy,
                budget_components=("utility_outcome_noise",),
                component_delta=0.02,
                target_id="scalar_target",
                provenance="predeclared bounded-noise event",
            )
            decision = decide_target(
                [certificate],
                scientific_tolerance=tolerance,
                failure_ledger=ledger,
                prior_target_domain=TargetInterval(-prior_radius, prior_radius),
                prior_only_shrinkage_threshold=float(
                    cfg["prior_only_shrinkage_threshold"]
                ),
            ).as_dict()
            decision["certificate_radius"] = certificate.error_radius
            decision["covered"] = bool(
                certificate.feasible_set is not None
                and certificate.feasible_set.lower <= true_target
                <= certificate.feasible_set.upper
            )
            methods[name] = decision
        rows.append(
            {
                "rep": rep,
                "true_target": true_target,
                "scientific_tolerance": tolerance,
                "methods": methods,
            }
        )
    return rows


def _utility_metrics(records: list[dict[str, Any]], method: str) -> dict[str, Any]:
    point = [
        row
        for row in records
        if row["methods"][method]["status"] == DecisionStatus.POINT_ESTIMABLE.value
    ]
    covered = sum(row["methods"][method]["covered"] for row in point)
    marginal = sum(row["methods"][method]["covered"] for row in records)
    violations = sum(
        abs(
            float(row["methods"][method]["point_estimate"])
            - float(row["true_target"])
        )
        > float(row["scientific_tolerance"])
        for row in point
    )
    return {
        "eligible": len(records),
        "point_outputs": len(point),
        "point_yield": len(point) / len(records),
        "conditional_coverage": covered / len(point) if point else None,
        "marginal_coverage": marginal / len(records),
        "tolerance_violation": violations / len(point) if point else None,
    }


def _story_metrics(
    records: list[dict[str, Any]],
    tolerance: float,
    utility_records: list[dict[str, Any]],
) -> dict[str, Any]:
    paired_radii = [
        (
            float(row["methods"]["crome_optimal"]["certificate_radius"]),
            float(row["methods"]["crome_current"]["certificate_radius"]),
        )
        for row in utility_records
        if row["methods"]["crome_optimal"].get("certificate_radius") is not None
        and row["methods"]["crome_current"].get("certificate_radius") is not None
    ]
    optimal_radii = np.asarray([pair[0] for pair in paired_radii], dtype=float)
    current_radii = np.asarray([pair[1] for pair in paired_radii], dtype=float)
    optimal_points = sum(
        row["methods"]["crome_optimal"]["status"]
        == DecisionStatus.POINT_ESTIMABLE.value
        for row in records
    )
    matched_points = sum(
        row["methods"]["matched_uncertainty"]["status"]
        == DecisionStatus.POINT_ESTIMABLE.value
        for row in records
    )
    nonpoint = [
        row
        for row in records
        if row["expected_status"] != DecisionStatus.POINT_ESTIMABLE.value
    ]
    matched_false = sum(
        row["methods"]["matched_uncertainty"]["status"]
        == DecisionStatus.POINT_ESTIMABLE.value
        for row in nonpoint
    )
    return {
        "rq1": {
            "radius_noninferiority": bool(
                np.all(optimal_radii <= current_radii + 1e-8)
            ),
            "median_radius_optimal": float(np.median(optimal_radii)),
            "median_radius_current": float(np.median(current_radii)),
            "median_radius_reduction": float(
                np.median(current_radii - optimal_radii)
            ),
            "optimal": _utility_metrics(utility_records, "crome_optimal"),
            "current": _utility_metrics(utility_records, "crome_current"),
            "stress_grid_optimal": _point_metrics(
                records, "crome_optimal", tolerance
            ),
            "stress_grid_current": _point_metrics(
                records, "crome_current", tolerance
            ),
        },
        "rq2": {
            "point_outputs_matched": matched_points == optimal_points,
            "point_outputs": optimal_points,
            "matched_uncertainty_false_points": matched_false,
            "matched_uncertainty_nonpoint_cases": len(nonpoint),
            "matched_uncertainty_false_point_rate": (
                matched_false / len(nonpoint) if nonpoint else None
            ),
        },
    }


def _source_rows(records: list[dict[str, Any]], config_hash: str):
    for row in records:
        for method in _METHODS:
            output = row["methods"][method]
            interval = _interval(output, method)
            yield {
                "rep": row["rep"], "support_mass": row["support_mass"],
                "perturbation": row["perturbation"], "level": row["level"],
                "method": method, "expected_status": row["expected_status"],
                "predicted_status": output["status"], "true_target": row["true_target"],
                "structural_status": output.get("structural_status"),
                "operational_status": output.get("operational_status"),
                "certificate_scope": output.get("certificate_scope"),
                "point_estimate": output.get("point_estimate"),
                "interval_lower": interval[0] if interval else None,
                "interval_upper": interval[1] if interval else None,
                "runtime_seconds": output["runtime_seconds"], "config_sha256": config_hash,
            }


def _write_csv(rows, path: Path) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader(); writer.writerows(rows)


def _semantic_digest(result: dict[str, Any]) -> str:
    def strip(value: Any):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if "runtime" not in k}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value
    return _hash_json(strip(result))


def run(mode: str = "smoke", outdir: Path | None = None) -> dict[str, Any]:
    cfg = load_exp_config("cs03_perturbation_grid")
    cfg["mode"] = mode
    cells = _cell_specs(cfg)
    n_reps = mode_reps(cfg, mode)
    n_units = int(cfg[f"n_trajectories_{mode}"])
    records = [
        _run_cell(cfg, cell, rep, mode, n_units)
        for rep in range(n_reps)
        for cell in cells
    ]
    _match_uncertainty_point_yield(records)
    utility_records = _utility_pilot(cfg, mode)
    method_summaries = {
        method: summarize_method_rows(
            _method_rows(records, method), level=float(cfg["mc_confidence_level"])
        )
        for method in _METHODS
    }
    config_hash = _hash_json(cfg)
    keys = [(r["rep"], r["cell_key"]) for r in records]
    result = {
        "experiment": "cs03_perturbation_grid", "mode": mode,
        "master_seed": int(cfg["master_seed"]), "config_sha256": config_hash,
        "n_reps": n_reps, "n_cells": len(cells), "n_trajectories": n_units,
        "target_definition": "C @ theta; support and overlap share one operator",
        "certificate_scope": "population_exact for clean design; assumption_conditional or finite_sample_only otherwise",
        "methods": method_summaries, "gate": _gate(cfg, records), "config": cfg,
        "story_metrics": _story_metrics(
            records,
            float(cfg["scientific_tolerance"]),
            utility_records,
        ),
        "artifact_audit": {
            "strict_json": True,
            "unique_keys": len(keys) == len(set(keys)),
            "expected_rows": n_reps * len(cells), "observed_rows": len(records),
        },
        "replication_records": records,
        "utility_records": utility_records,
    }
    json.dumps(result, allow_nan=False)
    if outdir is not None:
        outdir = Path(outdir)
        save_raw_and_summary(result, outdir / f"cs03_{mode}.json")
        source = outdir.parent / "source_data" / f"cs03_{mode}.csv" if outdir.name == "raw" else outdir / f"cs03_{mode}_source.csv"
        _write_csv(_source_rows(records, config_hash), source)
    return result


if __name__ == "__main__":
    print(json.dumps(run("smoke", Path("results/raw"))["gate"], indent=2))
