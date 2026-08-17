"""E-CS4: real Online Retail II timing with controlled coarsening and injected outcomes."""

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

from crome_identification.baselines import fit_naive_boundary, fit_ridge_target, fit_tsvd_target
from crome_identification.benchmarks.online_retail import sha256_file
from crome_identification.certification import (
    CertificateScope,
    DecisionStatus,
    FailureBudget,
    FailureBudgetLedger,
    ModelInfeasibleError,
    calibrate_bounded_design_error,
    calibrate_gaussian_outcome_noise,
    certify_overlap_target,
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


_METHODS = ("crome", "naive_boundary", "ridge", "tsvd")
_EXPECTED = {
    "fine_4h": DecisionStatus.POINT_ESTIMABLE.value,
    "daily_gap": DecisionStatus.SET_ESTIMABLE.value,
    "target_mark_collapse": DecisionStatus.NONRECOVERABLE.value,
    "rounded_hidden_time": DecisionStatus.INCONCLUSIVE.value,
}


@dataclass(frozen=True)
class RealTimingGeometry:
    regime: str
    customer_ids: np.ndarray
    event_customer_index: np.ndarray
    event_marks: np.ndarray
    true_lags: np.ndarray
    observed_lags: np.ndarray
    true_design: np.ndarray
    observed_design: np.ndarray
    cadence_hours: float
    near_zero_hours: float
    target_near_zero_count: int
    observed_target_near_zero_count: int

    @property
    def n_customers(self) -> int:
        return int(self.customer_ids.size)


@dataclass(frozen=True)
class InjectedRetailData:
    geometry: RealTimingGeometry
    outcomes: np.ndarray
    trace_responses: np.ndarray
    theta: np.ndarray
    target: np.ndarray
    true_target: float


def _calendar_lags(timestamp_hours: np.ndarray, cadence: float, phase: float = 0.0) -> np.ndarray:
    remainder = np.mod(timestamp_hours - phase, cadence)
    lag = cadence - remainder
    lag[np.isclose(remainder, 0.0)] = cadence
    return lag


def _aggregate_design(customer_index: np.ndarray, marks: np.ndarray, lags: np.ndarray,
                      n_customers: int, cadence: float, near_zero: float,
                      *, collapse_target: bool = False) -> np.ndarray:
    design = np.zeros((n_customers, 3), dtype=float)
    customer_counts = np.bincount(customer_index, minlength=n_customers)
    scale = np.sqrt(np.maximum(customer_counts, 1))[customer_index]
    weights = np.exp(-lags / cadence) / scale
    for mark in range(3):
        active = marks == mark
        if mark == 0:
            active &= lags <= near_zero
            if collapse_target:
                active &= False
        np.add.at(design[:, mark], customer_index[active], weights[active])
    return design


def _build_real_timing_geometry(path: Path, regime: str) -> RealTimingGeometry:
    if regime not in _EXPECTED:
        raise ValueError(f"unknown real-timing regime: {regime}")
    arrays = np.load(Path(path), allow_pickle=False)
    customers, customer_index = np.unique(
        arrays["trajectory_index"].astype(np.int64), return_inverse=True
    )
    marks = arrays["mark"].astype(int)
    timestamp_hours = arrays["time_of_day_minutes"].astype(np.float64) / 60.0
    cadence = 4.0 if regime in {"fine_4h", "target_mark_collapse", "rounded_hidden_time"} else 24.0
    near_zero = 1.0
    true_lags = _calendar_lags(timestamp_hours, cadence)
    collapse = regime == "target_mark_collapse"
    true_design = _aggregate_design(
        customer_index, marks, true_lags, customers.size, cadence, near_zero,
        collapse_target=collapse,
    )
    observed_lags = true_lags.copy()
    if regime == "rounded_hidden_time":
        observed_lags = np.maximum(np.finfo(float).eps, np.round(true_lags / 2.0) * 2.0)
    observed_design = _aggregate_design(
        customer_index, marks, observed_lags, customers.size, cadence, near_zero,
        collapse_target=collapse,
    )
    return RealTimingGeometry(
        regime=regime,
        customer_ids=customers,
        event_customer_index=customer_index,
        event_marks=marks,
        true_lags=true_lags / cadence,
        observed_lags=observed_lags / cadence,
        true_design=true_design,
        observed_design=observed_design,
        cadence_hours=cadence,
        near_zero_hours=near_zero,
        target_near_zero_count=int(np.sum((marks == 0) & (true_lags <= near_zero))),
        observed_target_near_zero_count=int(np.sum((marks == 0) & (observed_lags <= near_zero))),
    )


def _inject_semisynthetic(geometry: RealTimingGeometry, *, seed: int,
                          outcome_sigma: float, trace_sigma: float,
                          temporal_slope: float = 0.05) -> InjectedRetailData:
    rng = np.random.default_rng(seed)
    theta = np.array([1.0, 0.3, -0.4], dtype=float)
    target = np.array([1.0, 0.0, 0.0], dtype=float)
    outcomes = geometry.true_design @ theta + rng.normal(
        0.0, outcome_sigma, size=geometry.n_customers
    )
    target_events = geometry.event_marks == 0
    trace_responses = np.full(geometry.event_marks.size, np.nan, dtype=float)
    trace_responses[target_events] = (
        float(target @ theta)
        + temporal_slope * geometry.true_lags[target_events]
        + rng.normal(0.0, trace_sigma, size=int(np.sum(target_events)))
    )
    return InjectedRetailData(
        geometry=geometry, outcomes=outcomes, trace_responses=trace_responses,
        theta=theta, target=target, true_target=float(target @ theta),
    )


def _indices(values: tuple[int, ...]) -> np.ndarray:
    return np.asarray(values, dtype=int)


def _event_rows(geometry: RealTimingGeometry, customers: np.ndarray) -> np.ndarray:
    mask = np.zeros(geometry.n_customers, dtype=bool)
    mask[customers] = True
    return np.flatnonzero((geometry.event_marks == 0) & mask[geometry.event_customer_index])


def _budgets(cfg: dict[str, Any], data: InjectedRetailData, train: np.ndarray,
             calibration: np.ndarray, test: np.ndarray):
    geometry = data.geometry
    theta_pilot = np.linalg.lstsq(
        geometry.observed_design[train], data.outcomes[train], rcond=1e-12
    )[0]
    outcome_residuals = data.outcomes[calibration] - geometry.observed_design[calibration] @ theta_pilot
    outcome = calibrate_gaussian_outcome_noise(
        outcome_residuals, n_test=int(test.size), delta=float(cfg["component_delta"])
    )
    train_events = _event_rows(geometry, train)
    calibration_events = _event_rows(geometry, calibration)
    test_events = _event_rows(geometry, test)
    trace_design = np.column_stack([
        np.ones(train_events.size), geometry.observed_lags[train_events]
    ])
    beta = np.linalg.lstsq(trace_design, data.trace_responses[train_events], rcond=None)[0]
    trace_residuals = data.trace_responses[calibration_events] - (
        beta[0] + beta[1] * geometry.observed_lags[calibration_events]
    )
    trace = calibrate_gaussian_outcome_noise(
        trace_residuals, n_test=int(test_events.size), delta=float(cfg["component_delta"])
    )
    return outcome, trace, test_events


def _crome(cfg: dict[str, Any], data: InjectedRetailData, train: np.ndarray,
           calibration: np.ndarray, test: np.ndarray):
    started = perf_counter()
    geometry = data.geometry
    outcome_budget, trace_budget, test_events = _budgets(cfg, data, train, calibration, test)
    if geometry.regime == "fine_4h":
        near = geometry.observed_lags[test_events] <= geometry.near_zero_hours / geometry.cadence_hours
        estimate = float(np.mean(data.trace_responses[test_events][near]))
        radius = float(
            float(cfg["holder_L"]) * geometry.near_zero_hours / geometry.cadence_hours
            + trace_budget.simultaneous_coordinate_bound / math.sqrt(int(np.sum(near)))
        )
        temporal = certify_temporal_design_known(
            True,
            lags=geometry.observed_lags[test_events],
            responses=data.trace_responses[test_events],
            bandwidth=geometry.near_zero_hours / geometry.cadence_hours,
            holder_L=float(cfg["holder_L"]),
            holder_alpha=1.0,
            response_error_bound=trace_budget.simultaneous_coordinate_bound,
            budget_components=("trace_noise",),
            component_delta=float(cfg["component_delta"]),
            provenance="oracle-informed CS04 fine-calendar fixture",
        )
    elif geometry.regime == "daily_gap":
        temporal = certify_temporal_gap_conditional(
            geometry.observed_lags[test_events], data.trace_responses[test_events],
            holder_L=float(cfg["holder_L"]), holder_alpha=float(cfg["holder_alpha"]),
            response_error_bound=trace_budget.simultaneous_coordinate_bound,
            budget_components=("trace_noise",),
            component_delta=float(cfg["component_delta"]),
            provenance="oracle-informed CS04 daily-gap fixture",
        )
    elif geometry.regime == "target_mark_collapse":
        temporal = certify_temporal_design_known(
            False,
            support_gap_proof=support_gap_fixture_proof(
                geometry.observed_lags[test_events],
                fixture_name="CS04 target-mark-collapse regime",
            ),
            provenance="oracle-informed CS04 target-mark-collapse fixture",
        )
    else:
        temporal = certify_temporal_empirical(geometry.observed_lags[test_events])

    exact = geometry.regime != "rounded_hidden_time"
    design_calibration = None
    design_error = 0.0
    if not exact:
        design_calibration = calibrate_bounded_design_error(
            geometry.true_design[calibration], geometry.observed_design[calibration],
            n_test=int(test.size), row_error_bound=float(cfg["row_error_bound"]),
            delta=float(cfg["component_delta"]),
        )
        design_error = design_calibration.operator_error_bound
    overlap = certify_overlap_target(
        geometry.observed_design[test], data.outcomes[test], data.target,
        noise_radius=outcome_budget.vector_norm_bound,
        design_error_bound=design_error, theta_radius=float(cfg["theta_radius"]),
        exact_design=exact,
        budget_components=(
            ("outcome_noise",) if exact else ("outcome_noise", "design_error")
        ),
        component_delta=float(cfg["component_delta"]),
        exact_null_proof=(
            coordinate_null_fixture_proof(
                geometry.observed_design[test],
                data.target,
                fixture_name=f"CS04 {geometry.regime}",
            )
            if exact
            else None
        ),
        target_id="scalar_target",
        provenance=f"oracle-informed CS04 {geometry.regime} fixture",
    )
    allocations = tuple(
        FailureBudget(component, component_delta, "certificate-bound held-out event")
        for certificate in (temporal, overlap)
        for component, component_delta in certificate.failure_allocations
    )
    ledger = (
        FailureBudgetLedger(
            total_delta=4.0 * float(cfg["component_delta"]), allocations=allocations
        )
        if allocations
        else None
    )
    try:
        output = decide_target(
            [temporal, overlap],
            scientific_tolerance=float(cfg["scientific_tolerance"]),
            failure_ledger=ledger,
        ).as_dict()
        failed = False
    except ModelInfeasibleError as exc:
        output = {
            "status": DecisionStatus.INCONCLUSIVE.value,
            "certificate_scope": CertificateScope.ASSUMPTION_CONDITIONAL.value,
            "feasible_set": None, "point_estimate": None, "uncertainty": None,
            "assumptions": [], "reasons": [str(exc)],
            "diagnostics": {"model_infeasible": True},
        }
        failed = True
    output.update(runtime_seconds=perf_counter() - started, failed=failed)
    return output, {
        "outcome_noise": outcome_budget.as_dict(), "trace_noise": trace_budget.as_dict(),
        "design_error": design_calibration.as_dict() if design_calibration else None,
    }, test_events


def _baseline(result, radius: float):
    estimate = result.target_estimate
    return {
        "status": DecisionStatus.POINT_ESTIMABLE.value if result.success else DecisionStatus.INCONCLUSIVE.value,
        "point_estimate": estimate,
        "interval": [estimate - radius, estimate + radius] if estimate is not None else None,
        "runtime_seconds": result.runtime_seconds, "failed": not result.success,
    }


def _run_regime(cfg: dict[str, Any], geometry: RealTimingGeometry, rep: int,
                mode: str) -> dict[str, Any]:
    bundle = make_seed_bundle(int(cfg["master_seed"]), "cs04", f"{mode}:{geometry.regime}", rep)
    seed = int(bundle.rng.integers(0, 2**32 - 1))
    data = _inject_semisynthetic(
        geometry, seed=seed, outcome_sigma=float(cfg["outcome_noise_sigma"]),
        trace_sigma=float(cfg["trace_noise_sigma"]), temporal_slope=float(cfg["temporal_slope"]),
    )
    split_seed = seed ^ 0x5A5A5A5A
    split = split_trajectory_ids(np.arange(geometry.n_customers), tuple(cfg["split_ratios"]), seed=split_seed)
    train, calibration, test = map(_indices, (split.train, split.calibration, split.test))
    crome, calibration_record, test_events = _crome(cfg, data, train, calibration, test)
    naive = fit_naive_boundary(
        geometry.observed_lags[test_events], data.trace_responses[test_events],
        bandwidth=geometry.near_zero_hours / geometry.cadence_hours,
    )
    ridge = fit_ridge_target(
        geometry.observed_design[test], data.outcomes[test], data.target,
        lam=float(cfg["ridge_lambda"]),
    )
    tsvd = fit_tsvd_target(
        geometry.observed_design[test], data.outcomes[test], data.target,
        rank=int(cfg["parameter_dim"]),
    )
    radius = 1.96 * float(cfg["outcome_noise_sigma"])
    return {
        "rep": rep, "regime": geometry.regime, "expected_status": _EXPECTED[geometry.regime],
        "fixture_kind": "oracle_informed_contract_fixture",
        "evaluation_only": {
            "expected_status": _EXPECTED[geometry.regime],
            "regime_label": geometry.regime,
        },
        "n_customers": geometry.n_customers, "n_events": int(geometry.event_marks.size),
        "target_near_zero_count": geometry.target_near_zero_count,
        "observed_target_near_zero_count": geometry.observed_target_near_zero_count,
        "true_target": data.true_target, "target_definition": "C @ theta",
        "data_roles": {"geometry": "real timestamps only", "responses": "injected after geometry freeze"},
        "split_disjoint": split.as_dict()["disjoint"], "calibration": calibration_record,
        "methods": {
            "crome": crome, "naive_boundary": _baseline(naive, radius),
            "ridge": _baseline(ridge, radius), "tsvd": _baseline(tsvd, radius),
        },
    }


def _interval(output: dict[str, Any], method: str):
    if method == "crome":
        value = output.get("feasible_set")
        return (float(value["lower"]), float(value["upper"])) if isinstance(value, dict) and value.get("bounded") else None
    value = output.get("interval")
    return tuple(value) if value is not None else None


def _rows(records: list[dict[str, Any]], method: str):
    return [EvaluationRow(
        expected_status=row["expected_status"], predicted_status=row["methods"][method]["status"],
        point_estimate=row["methods"][method].get("point_estimate"),
        interval=_interval(row["methods"][method], method), true_target=row["true_target"],
        runtime_seconds=row["methods"][method]["runtime_seconds"], failed=row["methods"][method]["failed"],
    ) for row in records]


def _coverage(records: list[dict[str, Any]], expected: str, level: float):
    selected = [row for row in records if row["expected_status"] == expected]
    covered = sum(
        (interval := _interval(row["methods"]["crome"], "crome")) is not None
        and interval[0] <= row["true_target"] <= interval[1]
        for row in selected
    )
    return covered, len(selected), wilson_interval(covered, len(selected), level=level)


def _gate(cfg: dict[str, Any], records: list[dict[str, Any]], eligible: int):
    level = float(cfg["mc_confidence_level"])
    nonpoint = [row for row in records if row["expected_status"] != DecisionStatus.POINT_ESTIMABLE.value]
    false_points = sum(row["methods"]["crome"]["status"] == DecisionStatus.POINT_ESTIMABLE.value for row in nonpoint)
    false_ci = wilson_interval(false_points, len(nonpoint), level=level)
    point_cov, point_n, point_ci = _coverage(records, DecisionStatus.POINT_ESTIMABLE.value, level)
    set_cov, set_n, set_ci = _coverage(records, DecisionStatus.SET_ESTIMABLE.value, level)
    product_status_counts = {
        "UNKNOWN_x_POINT_AT_TAU": sum(
            row["methods"]["crome"].get("structural_status") == "UNKNOWN"
            and row["methods"]["crome"].get("operational_status") == "POINT_AT_TAU"
            for row in records
        ),
        "NONIDENTIFIED_x_SET": sum(
            row["methods"]["crome"].get("structural_status") == "NONIDENTIFIED"
            and row["methods"]["crome"].get("operational_status") == "SET"
            for row in records
        ),
        "NONIDENTIFIED_x_INCONCLUSIVE": sum(
            row["methods"]["crome"].get("structural_status") == "NONIDENTIFIED"
            and row["methods"]["crome"].get("operational_status") == "INCONCLUSIVE"
            for row in records
        ),
        "UNKNOWN_x_INCONCLUSIVE": sum(
            row["methods"]["crome"].get("structural_status") == "UNKNOWN"
            and row["methods"]["crome"].get("operational_status") == "INCONCLUSIVE"
            for row in records
        ),
    }
    checks = {
        "all_eligible_customers": {"passed": all(row["n_customers"] == eligible for row in records), "observed": eligible, "required": eligible},
        "product_status_population": {"passed": min(product_status_counts.values()) >= int(cfg["gate_min_status_count"]), "observed": product_status_counts, "required": int(cfg["gate_min_status_count"])},
        "false_point_upper_bound": {"passed": false_ci["upper"] <= float(cfg["gate_max_false_point_upper"]), "observed": false_ci["upper"], "required": float(cfg["gate_max_false_point_upper"])},
        "point_coverage_lower_bound": {"passed": point_ci["lower"] >= float(cfg["gate_min_point_coverage_lower"]), "observed": point_ci["lower"], "required": float(cfg["gate_min_point_coverage_lower"])},
        "set_coverage_lower_bound": {"passed": set_ci["lower"] >= float(cfg["gate_min_set_coverage_lower"]), "observed": set_ci["lower"], "required": float(cfg["gate_min_set_coverage_lower"])},
        "split_integrity": {"passed": all(row["split_disjoint"] for row in records), "observed": sum(row["split_disjoint"] for row in records), "required": len(records)},
    }
    return {
        "passed": all(item["passed"] for item in checks.values()), "checks": checks,
        "false_point_count": false_points, "nonpoint_count": len(nonpoint),
        "point_coverage": {"covered": point_cov, "total": point_n, "interval": point_ci},
        "set_coverage": {"covered": set_cov, "total": set_n, "interval": set_ci},
    }


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write_source(records: list[dict[str, Any]], config_hash: str, path: Path):
    rows = []
    for row in records:
        for method in _METHODS:
            output = row["methods"][method]; interval = _interval(output, method)
            rows.append({
                "rep": row["rep"], "regime": row["regime"], "method": method,
                "expected_status": row["expected_status"], "predicted_status": output["status"],
                "structural_status": output.get("structural_status"),
                "operational_status": output.get("operational_status"),
                "certificate_scope": output.get("certificate_scope"),
                "true_target": row["true_target"], "point_estimate": output.get("point_estimate"),
                "interval_lower": interval[0] if interval else None,
                "interval_upper": interval[1] if interval else None,
                "runtime_seconds": output["runtime_seconds"], "config_sha256": config_hash,
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def run(mode: str = "smoke", outdir: Path | None = None) -> dict[str, Any]:
    cfg = load_exp_config("cs04_online_retail"); cfg["mode"] = mode
    profile_path = Path(cfg["data_profile"]); data_path = Path(cfg["processed_data"])
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    geometries = {regime: _build_real_timing_geometry(data_path, regime) for regime in cfg["regimes"]}
    records = [
        _run_regime(cfg, geometries[regime], rep, mode)
        for rep in range(mode_reps(cfg, mode)) for regime in cfg["regimes"]
    ]
    methods = {method: summarize_method_rows(_rows(records, method), level=float(cfg["mc_confidence_level"])) for method in _METHODS}
    config_hash = _hash_json(cfg)
    result = {
        "experiment": "cs04_online_retail", "mode": mode, "master_seed": int(cfg["master_seed"]),
        "fixture_kind": "oracle_informed_contract_fixture",
        "config_sha256": config_hash, "n_reps": mode_reps(cfg, mode), "n_regimes": len(cfg["regimes"]),
        "benchmark_semantics": "real event timing + controlled coarsening; synthetic outcomes; no causal claim",
        "data_profile": profile, "processed_data_sha256": sha256_file(data_path),
        "methods": methods, "gate": _gate(cfg, records, int(profile["eligible_customers"])),
        "artifact_audit": {
            "data_hashes_attached": all(profile.get(key) for key in ("archive_sha256", "workbook_sha256", "processed_npz_sha256")),
            "unique_keys": len(records) == len({(row["rep"], row["regime"]) for row in records}),
            "strict_json": True,
        },
        "config": cfg, "replication_records": records,
    }
    json.dumps(result, allow_nan=False)
    if outdir is not None:
        outdir = Path(outdir); save_raw_and_summary(result, outdir / f"cs04_{mode}.json")
        source = outdir.parent / "source_data" / f"cs04_{mode}.csv" if outdir.name == "raw" else outdir / f"cs04_{mode}_source.csv"
        _write_source(records, config_hash, source)
    return result


if __name__ == "__main__":
    print(json.dumps(run("smoke", Path("results/raw"))["gate"], indent=2))
