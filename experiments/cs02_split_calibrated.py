"""CS02: split-calibrated, same-target CROME pilot with an automatic main Gate."""

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
    BaselineTargetResult,
    fit_naive_boundary,
    fit_ridge_target,
    fit_tsvd_target,
    select_inverse_hyperparameters,
)
from crome_identification.certification import (
    CertificateScope,
    CertificationMode,
    DecisionStatus,
    FailureBudget,
    FailureBudgetLedger,
    ModelInfeasibleError,
    TargetCertificate,
    calibrate_gaussian_design_error,
    calibrate_gaussian_outcome_noise,
    calibrate_lower_mass,
    certify_overlap_target,
    certify_temporal_conditional,
    certify_temporal_design_known,
    certify_temporal_gap_conditional,
    decide_target,
)
from crome_identification.evaluation import (
    EvaluationRow,
    product_truth_allows_public_point,
    split_trajectory_ids,
    summarize_method_rows,
)
from crome_identification.inference.boundary_regression import (
    one_sided_local_linear,
    select_bandwidth_group_kfold,
)
from crome_identification.seeding import make_seed_bundle

from ._common import load_exp_config, mode_reps, save_raw_and_summary
from ._proof_fixtures import coordinate_null_fixture_proof, support_gap_fixture_proof


_REGIMES = (
    ("strong_support_full_rank", DecisionStatus.POINT_ESTIMABLE),
    ("gap_estimated_design", DecisionStatus.SET_ESTIMABLE),
    ("structural_target_null", DecisionStatus.NONRECOVERABLE),
    ("sparse_near_zero", DecisionStatus.INCONCLUSIVE),
)
_EXPECTED_PRODUCT = {
    "strong_support_full_rank": ("UNKNOWN", "POINT_AT_TAU"),
    "gap_estimated_design": ("UNKNOWN", "SET"),
    "structural_target_null": ("NONIDENTIFIED", "INCONCLUSIVE"),
    "sparse_near_zero": ("UNKNOWN", "INCONCLUSIVE"),
}
_METHODS = ("crome", "naive_boundary", "ridge", "tsvd")


@dataclass(frozen=True)
class _RegimeData:
    trajectory_ids: np.ndarray
    lags: np.ndarray
    responses: np.ndarray
    true_design: np.ndarray
    first_design: np.ndarray
    second_design: np.ndarray
    outcomes: np.ndarray
    theta: np.ndarray
    target: np.ndarray
    true_target: float
    design_sigma: float


def _config_hash(cfg: dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _id_hash(values: tuple[int, ...]) -> str:
    payload = ",".join(str(value) for value in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _generate_regime(
    cfg: dict[str, Any],
    rng: np.random.Generator,
    n_trajectories: int,
    regime: str,
) -> _RegimeData:
    parameter_dim = int(cfg["parameter_dim"])
    theta = np.asarray(cfg["theta"], dtype=float)
    target = np.asarray(cfg["target_C"], dtype=float)
    if theta.shape != (parameter_dim,) or target.shape != (parameter_dim,):
        raise ValueError("theta and target_C must match parameter_dim")

    true_design = rng.normal(size=(n_trajectories, parameter_dim))
    if regime == "structural_target_null":
        true_design[:, 1] = true_design[:, 0]
        design_sigma = 0.0
    elif regime == "gap_estimated_design":
        design_sigma = float(cfg["design_noise_sigma_set"])
    else:
        design_sigma = float(cfg["design_noise_sigma_point"])

    first_design = true_design + rng.normal(0.0, design_sigma, size=true_design.shape)
    second_design = true_design + rng.normal(0.0, design_sigma, size=true_design.shape)
    outcomes = true_design @ theta + rng.normal(
        0.0, float(cfg["outcome_noise_sigma"]), size=n_trajectories
    )
    if regime == "strong_support_full_rank":
        lags = rng.uniform(0.001, 1.0, size=n_trajectories)
    elif regime == "sparse_near_zero":
        lags = rng.uniform(0.45, 1.0, size=n_trajectories)
    else:
        lags = rng.uniform(0.5, 1.0, size=n_trajectories)
    true_target = float(target @ theta)
    responses = (
        true_target
        + float(cfg["temporal_slope"]) * lags
        + rng.normal(0.0, float(cfg["response_noise_sigma"]), size=n_trajectories)
    )
    return _RegimeData(
        trajectory_ids=np.arange(n_trajectories),
        lags=lags,
        responses=responses,
        true_design=true_design,
        first_design=first_design,
        second_design=second_design,
        outcomes=outcomes,
        theta=theta,
        target=target,
        true_target=true_target,
        design_sigma=design_sigma,
    )


def _indices(values: tuple[int, ...]) -> np.ndarray:
    return np.asarray(values, dtype=int)


def _fit_response_calibration(
    data: _RegimeData,
    train: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
    bandwidth: float,
    delta: float,
):
    fit = one_sided_local_linear(data.lags[train], data.responses[train], bandwidth)
    if not math.isfinite(fit["estimate"]):
        design = np.column_stack([np.ones(train.size), data.lags[train]])
        beta = np.linalg.lstsq(design, data.responses[train], rcond=None)[0]
        intercept, slope = map(float, beta)
    else:
        intercept = float(fit["estimate"])
        slope = float(fit.get("slope", 0.0))
    residuals = data.responses[calibration] - (
        intercept + slope * data.lags[calibration]
    )
    return calibrate_gaussian_outcome_noise(
        residuals, n_test=int(test.size), delta=delta
    )


def _fit_outcome_calibration(
    data: _RegimeData,
    train: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
    delta: float,
):
    theta_pilot = np.linalg.lstsq(
        data.first_design[train], data.outcomes[train], rcond=1e-12
    )[0]
    residuals = data.outcomes[calibration] - data.first_design[calibration] @ theta_pilot
    return calibrate_gaussian_outcome_noise(
        residuals, n_test=int(test.size), delta=delta
    )


def _invalid_support_certificate(lower_mass) -> TargetCertificate:
    return TargetCertificate(
        source="temporal_support_calibration",
        mode=CertificationMode.ASSUMPTION_CONDITIONAL,
        scope=CertificateScope.ASSUMPTION_CONDITIONAL,
        valid=False,
        assumptions=tuple(lower_mass.assumptions),
        diagnostics=lower_mass.as_dict(),
        reason="calibration split supplied no positive near-zero lower-mass bound",
    )


def _crome_decision(
    cfg: dict[str, Any],
    regime: str,
    data: _RegimeData,
    calibration: np.ndarray,
    test: np.ndarray,
    response_noise,
    outcome_noise,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = perf_counter()
    delta = float(cfg["component_delta"])
    calibration_record: dict[str, Any] = {
        "response_noise": response_noise.as_dict(),
        "outcome_noise": outcome_noise.as_dict(),
        "joint_error_budget_upper": 5.0 * delta,
    }

    if regime == "structural_target_null":
        temporal = certify_temporal_design_known(
            False,
            support_gap_proof=support_gap_fixture_proof(
                data.lags[test], fixture_name="CS02 structural-target-null regime"
            ),
            provenance="oracle-informed CS02 structural support-gap fixture",
        )
        overlap = certify_overlap_target(
            data.first_design[test],
            data.outcomes[test],
            data.target,
            noise_radius=0.0,
            design_error_bound=0.0,
            theta_radius=float(cfg["theta_radius"]),
            weight_strategy="certificate_optimal",
            exact_design=True,
            exact_null_proof=coordinate_null_fixture_proof(
                data.first_design[test],
                data.target,
                fixture_name="CS02 structural-target-null regime",
            ),
            target_id="scalar_target",
            provenance="oracle-informed CS02 exact-null fixture",
        )
    else:
        design_error = calibrate_gaussian_design_error(
            data.first_design[calibration],
            data.second_design[calibration],
            n_test=int(test.size),
            delta=delta,
        )
        calibration_record["design_error"] = design_error.as_dict()
        overlap = certify_overlap_target(
            data.first_design[test],
            data.outcomes[test],
            data.target,
            noise_radius=outcome_noise.vector_norm_bound,
            design_error_bound=design_error.operator_error_bound,
            theta_radius=float(cfg["theta_radius"]),
            weight_strategy="certificate_optimal",
            exact_design=False,
            budget_components=("outcome_noise", "design_error"),
            component_delta=delta,
            target_id="scalar_target",
            provenance="held-out CS02 outcome and operator calibration splits",
        )
        if regime == "gap_estimated_design":
            temporal = certify_temporal_gap_conditional(
                data.lags[test],
                data.responses[test],
                holder_L=float(cfg["holder_L_gap"]),
                holder_alpha=float(cfg["holder_alpha"]),
                response_error_bound=response_noise.simultaneous_coordinate_bound,
                budget_components=("response_noise",),
                component_delta=delta,
                provenance="held-out CS02 response calibration split",
            )
        else:
            lower_mass = calibrate_lower_mass(
                data.lags[calibration],
                bandwidth=float(cfg["support_bandwidth"]),
                beta=float(cfg["lower_mass_beta"]),
                delta=delta,
            )
            calibration_record["lower_mass"] = lower_mass.as_dict()
            temporal = (
                _invalid_support_certificate(lower_mass)
                if not lower_mass.valid
                else certify_temporal_conditional(
                    data.lags[test],
                    data.responses[test],
                    bandwidth=float(cfg["support_bandwidth"]),
                    n_units=int(test.size),
                    lower_mass_c=lower_mass.lower_mass_c,
                    lower_mass_beta=float(cfg["lower_mass_beta"]),
                    holder_L=float(cfg["holder_L_point"]),
                    holder_alpha=float(cfg["holder_alpha"]),
                    noise_scale=response_noise.sigma_upper,
                    delta=delta,
                    budget_components=("temporal_count", "trace_mean_noise"),
                    external_failure_allocations=(
                        ("lower_mass_calibration", delta),
                        ("response_scale_calibration", delta),
                    ),
                    provenance="held-out CS02 lower-mass and response calibration splits",
                )
            )

    allocations = tuple(
        FailureBudget(component, component_delta, "certificate-bound held-out event")
        for certificate in (temporal, overlap)
        for component, component_delta in certificate.failure_allocations
    )
    ledger = (
        FailureBudgetLedger(total_delta=5.0 * delta, allocations=allocations)
        if allocations
        else None
    )
    try:
        decision = decide_target(
            [temporal, overlap],
            scientific_tolerance=float(cfg["scientific_tolerance"]),
            failure_ledger=ledger,
        )
        output = decision.as_dict()
        output.update(
            weight_strategy=overlap.diagnostics["weight_strategy"],
            runtime_seconds=perf_counter() - started,
            failed=False,
        )
    except ModelInfeasibleError as exc:
        output = {
            "status": DecisionStatus.INCONCLUSIVE.value,
            "certificate_scope": CertificateScope.ASSUMPTION_CONDITIONAL.value,
            "feasible_set": None,
            "point_estimate": None,
            "uncertainty": None,
            "assumptions": [],
            "reasons": [str(exc)],
            "diagnostics": {"model_infeasible": True},
            "weight_strategy": overlap.diagnostics["weight_strategy"],
            "runtime_seconds": perf_counter() - started,
            "failed": True,
        }
    return output, calibration_record


def _inverse_interval(
    design: np.ndarray,
    target: np.ndarray,
    estimate: float | None,
    noise_norm_bound: float,
    *,
    method: str,
    ridge_lambda: float,
    tsvd_rank: int,
) -> tuple[float, float] | None:
    if estimate is None:
        return None
    if method == "ridge":
        gram = design.T @ design + ridge_lambda * np.eye(design.shape[1])
        coefficients = (
            np.linalg.pinv(gram, rcond=1e-12) @ target
            if ridge_lambda == 0.0
            else np.linalg.solve(gram, target)
        )
        weights = design @ coefficients
    else:
        U, singular_values, Vt = np.linalg.svd(design, full_matrices=False)
        retained = singular_values[:tsvd_rank]
        tolerance = 1e-12 * max(1.0, float(singular_values[0]))
        inverse = np.where(retained > tolerance, 1.0 / retained, 0.0)
        weights = U[:, :tsvd_rank] @ (inverse * (Vt[:tsvd_rank] @ target))
    radius = float(np.linalg.norm(weights) * noise_norm_bound)
    return (float(estimate - radius), float(estimate + radius))


def _baseline_output(
    result: BaselineTargetResult,
    interval: tuple[float, float] | None,
) -> dict[str, Any]:
    return {
        "status": (
            DecisionStatus.POINT_ESTIMABLE.value
            if result.success
            else DecisionStatus.INCONCLUSIVE.value
        ),
        "point_estimate": result.target_estimate,
        "interval": list(interval) if interval is not None else None,
        "success": result.success,
        "failed": not result.success,
        "failure_reason": result.failure_reason,
        "hyperparameters": dict(result.hyperparameters),
        "runtime_seconds": result.runtime_seconds,
    }


def _run_regime(
    cfg: dict[str, Any],
    regime: str,
    expected_status: DecisionStatus,
    split,
    rng: np.random.Generator,
    n_trajectories: int,
) -> dict[str, Any]:
    data = _generate_regime(cfg, rng, n_trajectories, regime)
    train = _indices(split.train)
    calibration = _indices(split.calibration)
    test = _indices(split.test)
    bandwidth = select_bandwidth_group_kfold(
        data.lags[train],
        data.responses[train],
        data.trajectory_ids[train],
        [float(value) for value in cfg["bandwidth_grid"]],
        n_splits=int(cfg["cv_folds"]),
    )
    inverse_hyper = select_inverse_hyperparameters(
        data.first_design[train],
        data.outcomes[train],
        [float(value) for value in cfg["ridge_grid"]],
        [int(value) for value in cfg["rank_grid"]],
        n_splits=int(cfg["cv_folds"]),
    )
    delta = float(cfg["component_delta"])
    response_noise = _fit_response_calibration(
        data, train, calibration, test, bandwidth, delta
    )
    outcome_noise = _fit_outcome_calibration(
        data, train, calibration, test, delta
    )
    crome, calibration_record = _crome_decision(
        cfg, regime, data, calibration, test, response_noise, outcome_noise
    )

    naive = fit_naive_boundary(data.lags[test], data.responses[test], bandwidth=bandwidth)
    n_effective = max(1, int(np.sum(data.lags[test] <= bandwidth)))
    holder_L = float(
        cfg["holder_L_point"]
        if regime == "strong_support_full_rank"
        else cfg["holder_L_gap"]
    )
    naive_radius = (
        holder_L * bandwidth ** float(cfg["holder_alpha"])
        + response_noise.sigma_upper
        * math.sqrt(2.0 * math.log(2.0 / delta) / n_effective)
    )
    naive_interval = (
        (naive.target_estimate - naive_radius, naive.target_estimate + naive_radius)
        if naive.target_estimate is not None
        else None
    )
    ridge = fit_ridge_target(
        data.first_design[test], data.outcomes[test], data.target,
        lam=inverse_hyper.ridge_lambda,
    )
    tsvd = fit_tsvd_target(
        data.first_design[test], data.outcomes[test], data.target,
        rank=inverse_hyper.tsvd_rank,
    )
    ridge_interval = _inverse_interval(
        data.first_design[test], data.target, ridge.target_estimate,
        outcome_noise.vector_norm_bound, method="ridge",
        ridge_lambda=inverse_hyper.ridge_lambda, tsvd_rank=inverse_hyper.tsvd_rank,
    )
    tsvd_interval = _inverse_interval(
        data.first_design[test], data.target, tsvd.target_estimate,
        outcome_noise.vector_norm_bound, method="tsvd",
        ridge_lambda=inverse_hyper.ridge_lambda, tsvd_rank=inverse_hyper.tsvd_rank,
    )
    methods = {
        "crome": crome,
        "naive_boundary": _baseline_output(naive, naive_interval),
        "ridge": _baseline_output(ridge, ridge_interval),
        "tsvd": _baseline_output(tsvd, tsvd_interval),
    }
    expected_structural, expected_operational = _EXPECTED_PRODUCT[regime]
    return {
        "regime": regime,
        "expected_status": expected_status.value,
        "expected_structural_status": expected_structural,
        "expected_operational_status": expected_operational,
        "target_definition": "C @ theta",
        "target_C": data.target.tolist(),
        "true_target": data.true_target,
        "data_roles": {
            "hyperparameter_selection": "train",
            "error_budget_calibration": "calibration",
            "final_estimation": "test",
        },
        "selected_hyperparameters": {
            "boundary_bandwidth": bandwidth,
            "ridge_lambda": inverse_hyper.ridge_lambda,
            "tsvd_rank": inverse_hyper.tsvd_rank,
            "ridge_validation_mse": inverse_hyper.ridge_validation_mse,
            "tsvd_validation_mse": inverse_hyper.tsvd_validation_mse,
        },
        "calibration": calibration_record,
        "dgp_audit": {
            "n_trajectories": n_trajectories,
            "design_noise_sigma": data.design_sigma,
            "minimum_test_lag": float(np.min(data.lags[test])),
            "test_near_zero_count": int(
                np.sum(data.lags[test] <= float(cfg["support_bandwidth"]))
            ),
        },
        "methods": methods,
    }


def _crome_interval(output: dict[str, Any]) -> tuple[float, float] | None:
    feasible = output.get("feasible_set")
    if not isinstance(feasible, dict) or not feasible.get("bounded", False):
        return None
    return (float(feasible["lower"]), float(feasible["upper"]))


def _evaluation_rows(records: list[dict[str, Any]], method: str) -> list[EvaluationRow]:
    rows: list[EvaluationRow] = []
    for replication in records:
        for regime in replication["regimes"]:
            output = regime["methods"][method]
            interval = (
                _crome_interval(output)
                if method == "crome"
                else tuple(output["interval"]) if output.get("interval") is not None else None
            )
            rows.append(
                EvaluationRow(
                    expected_status=regime["expected_status"],
                    predicted_status=output["status"],
                    point_estimate=output.get("point_estimate"),
                    interval=interval,
                    true_target=float(regime["true_target"]),
                    runtime_seconds=float(output.get("runtime_seconds", 0.0)),
                    failed=bool(output.get("failed", False)),
                    expected_structural_status=regime["expected_structural_status"],
                    expected_operational_status=regime["expected_operational_status"],
                )
            )
    return rows


def _finite_or_null(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, np.integer)):
        return True
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite_or_null(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_or_null(item) for item in value)
    return True


def _gate(
    cfg: dict[str, Any],
    records: list[dict[str, Any]],
    method_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    crome = method_summaries["crome"]
    coverage_lower = float(crome["point_coverage_interval"]["lower"])
    false_point_upper = float(crome["false_point_interval"]["upper"])
    expected_rows = len(records) * len(_REGIMES)
    no_point_leakage = all(
        output["status"] == DecisionStatus.POINT_ESTIMABLE.value
        or output.get("point_estimate") is None
        for replication in records
        for regime in replication["regimes"]
        for output in [regime["methods"]["crome"]]
    )
    checks = {
        "split_integrity": {
            "passed": all(record["split_audit"]["disjoint"] for record in records),
            "observed": sum(record["split_audit"]["disjoint"] for record in records),
            "required": len(records),
        },
        "record_completeness": {
            "passed": all(summary["n_total"] == expected_rows for summary in method_summaries.values()),
            "observed": {name: summary["n_total"] for name, summary in method_summaries.items()},
            "required": expected_rows,
        },
        "coverage_lower_bound": {
            "passed": coverage_lower >= float(cfg["gate_min_coverage_lower"]),
            "observed": coverage_lower,
            "required": float(cfg["gate_min_coverage_lower"]),
        },
        "false_point_upper_bound": {
            "passed": false_point_upper <= float(cfg["gate_max_false_point_upper"]),
            "observed": false_point_upper,
            "required": float(cfg["gate_max_false_point_upper"]),
        },
        "status_accuracy": {
            "passed": crome["status_accuracy"] >= float(cfg["gate_min_status_accuracy"]),
            "observed": crome["status_accuracy"],
            "required": float(cfg["gate_min_status_accuracy"]),
        },
        "no_point_leakage": {
            "passed": no_point_leakage,
            "observed": no_point_leakage,
            "required": True,
        },
        "false_point_improvement": {
            "passed": all(
                crome["false_point_rate"] < method_summaries[name]["false_point_rate"]
                for name in _METHODS if name != "crome"
            ),
            "observed": {
                name: summary["false_point_rate"]
                for name, summary in method_summaries.items()
            },
            "required": "CROME strictly lower than every point baseline",
        },
        "finite_or_explicit_null": {
            "passed": _finite_or_null(records) and _finite_or_null(method_summaries),
            "observed": _finite_or_null(records) and _finite_or_null(method_summaries),
            "required": True,
        },
    }
    return {
        "main_ready": all(item["passed"] for item in checks.values()),
        "checks": checks,
        "interpretation": (
            "A passed pilot Gate authorizes the predeclared main run; it does not by itself "
            "establish external validity or venue-level novelty."
        ),
    }


def _source_rows(records: list[dict[str, Any]], config_hash: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for replication in records:
        for regime in replication["regimes"]:
            for method in _METHODS:
                output = regime["methods"][method]
                interval = (
                    _crome_interval(output)
                    if method == "crome"
                    else tuple(output["interval"]) if output.get("interval") is not None else None
                )
                rows.append(
                    {
                        "rep": replication["rep"],
                        "regime": regime["regime"],
                        "method": method,
                        "expected_status": regime["expected_status"],
                        "expected_structural_status": regime["expected_structural_status"],
                        "expected_operational_status": regime["expected_operational_status"],
                        "predicted_status": output["status"],
                        "true_target": regime["true_target"],
                        "point_estimate": output.get("point_estimate"),
                        "interval_lower": interval[0] if interval is not None else None,
                        "interval_upper": interval[1] if interval is not None else None,
                        "false_point": int(
                            not product_truth_allows_public_point(
                                regime["expected_structural_status"],
                                regime["expected_operational_status"],
                            )
                            and output["status"] == DecisionStatus.POINT_ESTIMABLE.value
                        ),
                        "abstained": int(output["status"] == DecisionStatus.INCONCLUSIVE.value),
                        "failed": int(bool(output.get("failed", False))),
                        "runtime_seconds": output.get("runtime_seconds", 0.0),
                        "weight_strategy": output.get("weight_strategy"),
                        "config_sha256": config_hash,
                    }
                )
    return rows


def _write_source_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(mode: str = "smoke", outdir: Path | None = None) -> dict[str, Any]:
    cfg = load_exp_config("cs_split_calibrated")
    cfg["mode"] = mode
    n_reps = mode_reps(cfg, mode)
    n_trajectories = int(cfg[f"n_trajectories_{mode}"])
    split_ratios = tuple(float(value) for value in cfg["split_ratios"])
    replication_records: list[dict[str, Any]] = []

    for rep in range(n_reps):
        bundle = make_seed_bundle(int(cfg["master_seed"]), "cs02", mode, rep)
        split_seed = int(bundle.rng.integers(0, 2**32 - 1))
        split = split_trajectory_ids(np.arange(n_trajectories), split_ratios, seed=split_seed)
        split_audit = {
            "seed": split_seed,
            "train_count": len(split.train),
            "calibration_count": len(split.calibration),
            "test_count": len(split.test),
            "train_sha256": _id_hash(split.train),
            "calibration_sha256": _id_hash(split.calibration),
            "test_sha256": _id_hash(split.test),
            "disjoint": bool(split.as_dict()["disjoint"]),
            "complete": len(split.train) + len(split.calibration) + len(split.test) == n_trajectories,
        }
        regimes = []
        for regime_index, (regime, expected_status) in enumerate(_REGIMES):
            regime_rng = make_seed_bundle(
                int(cfg["master_seed"]), "cs02_regime",
                f"{mode}:rep={rep}:{regime}", regime_index,
            ).rng
            regimes.append(
                _run_regime(
                    cfg, regime, expected_status, split, regime_rng, n_trajectories
                )
            )
        replication_records.append(
            {"rep": rep, "split_audit": split_audit, "regimes": regimes}
        )

    method_summaries = {
        method: summarize_method_rows(
            _evaluation_rows(replication_records, method),
            level=float(cfg["mc_confidence_level"]),
            missing_point_penalty=float(cfg["missing_point_penalty"]),
        )
        for method in _METHODS
    }
    config_hash = _config_hash(cfg)
    crome_weight_strategies = sorted({
        regime["methods"]["crome"]["weight_strategy"]
        for replication in replication_records
        for regime in replication["regimes"]
    })
    summary = {
        "experiment": "cs02_split_calibrated",
        "mode": mode,
        "master_seed": int(cfg["master_seed"]),
        "config_sha256": config_hash,
        "crome_weight_strategy": (
            crome_weight_strategies[0]
            if len(crome_weight_strategies) == 1
            else crome_weight_strategies
        ),
        "n_reps": n_reps,
        "n_trajectories_per_regime": n_trajectories,
        "n_regimes": len(_REGIMES),
        "target_definition": "scalar C @ theta, fixed before each regime fit",
        "certificate_scope": "assumption_conditional except structural exact-null fixture",
        "methods": method_summaries,
        "config": cfg,
        "main_readiness_gate": None,
        "replication_records": replication_records,
    }
    summary["main_readiness_gate"] = _gate(cfg, replication_records, method_summaries)
    if outdir is not None:
        outdir = Path(outdir)
        save_raw_and_summary(summary, outdir / f"cs02_{mode}.json")
        source_path = (
            outdir.parent / "source_data" / f"cs02_{mode}.csv"
            if outdir.name == "raw"
            else outdir / f"cs02_{mode}_source.csv"
        )
        _write_source_csv(_source_rows(replication_records, config_hash), source_path)
    return summary


if __name__ == "__main__":
    print(run("smoke", Path("results/raw")))
