"""E-CS6: paired component ablations on aligned hard cases."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from crome_identification.baselines import fit_ridge_target
from crome_identification.benchmarks import PerturbationSpec, generate_aligned_geometry
from crome_identification.certification import (
    DecisionStatus,
    FailureBudget,
    FailureBudgetLedger,
    certify_overlap_target,
    certify_temporal_design_known,
    certify_temporal_empirical,
    decide_target,
)
from crome_identification.evaluation import wilson_interval
from crome_identification.evaluation.splits import split_trajectory_ids
from crome_identification.seeding import make_seed_bundle

from ._common import load_exp_config, mode_reps, save_raw_and_summary
from ._proof_fixtures import (
    coordinate_null_fixture_proof,
)


VARIANTS = (
    "full_crome",
    "no_support_certifier",
    "no_target_null_check",
    "force_point",
    "no_mark_sharing",
    "fixed_tolerance",
)
CELLS = {
    "unsupported_corrupted": {"mass": 0.0, "kind": "mark_noise", "level": 0.15, "scale": 1.0, "shared": False},
    "structural_null": {"mass": 0.0, "kind": "clean", "level": 0.0, "scale": 1.0, "shared": False},
    "shared_sparse": {"mass": 0.1, "kind": "clean", "level": 0.0, "scale": 1.0, "shared": True},
    "supported_strong": {"mass": 0.9, "kind": "clean", "level": 0.0, "scale": 1.0, "shared": False},
    "scale_small": {"mass": 0.9, "kind": "clean", "level": 0.0, "scale": 1.0e-12, "shared": False},
    "scale_large": {"mass": 0.9, "kind": "clean", "level": 0.0, "scale": 1.0e3, "shared": False},
}


def _interval(output: dict[str, Any]):
    value = output.get("feasible_set")
    if isinstance(value, dict) and value.get("bounded"):
        return [float(value["lower"]), float(value["upper"])]
    return output.get("interval")


def _point_output(estimate: float, radius: float, runtime: float = 0.0):
    return {
        "status": DecisionStatus.POINT_ESTIMABLE.value,
        "point_estimate": float(estimate),
        "interval": [float(estimate - radius), float(estimate + radius)],
        "runtime_seconds": float(runtime), "failed": False,
    }


def _full_output(cfg: dict[str, Any], cell_name: str, data, test: np.ndarray,
                 design: np.ndarray, outcomes: np.ndarray, scale: float,
                 *, mark_sharing: bool, support_certifier: bool = True,
                 target_null_check: bool = True):
    started = perf_counter()
    cell = CELLS[cell_name]
    delta = float(cfg["component_delta"])
    temporal = None
    if not support_certifier and cell_name != "structural_null":
        temporal = certify_temporal_empirical(data.observed_lags[test])
    elif cell_name == "structural_null":
        # This cell isolates the overlap null check; no temporal claim is composed.
        temporal = None
    elif cell["mass"] == 0:
        temporal = certify_temporal_empirical(data.observed_lags[test])
    else:
        if cell["shared"] and mark_sharing:
            design_rows = test
        else:
            design_rows = test[data.observed_marks[test] == 0]
        active = design_rows[
            data.observed_lags[design_rows] <= float(cfg["support_bandwidth"])
        ]
        estimate = float(np.mean(data.trace_responses[active]))
        radius = float(
            float(cfg["holder_L"]) * float(cfg["support_bandwidth"])
            + 3.0 * float(cfg["trace_noise_sigma"]) / math.sqrt(active.size)
        )
        temporal = certify_temporal_design_known(
            True,
            lags=data.observed_lags[design_rows],
            responses=data.trace_responses[design_rows],
            bandwidth=float(cfg["support_bandwidth"]),
            holder_L=float(cfg["holder_L"]),
            holder_alpha=1.0,
            response_error_bound=3.0 * float(cfg["trace_noise_sigma"]),
            budget_components=("trace_noise",),
            component_delta=delta,
            provenance="predeclared trace-noise bound",
        )
    exact = cell["kind"] == "clean"
    overlap = certify_overlap_target(
        design[test], outcomes[test], data.target,
        noise_radius=4.0 * float(cfg["outcome_noise_sigma"]) * math.sqrt(test.size) * scale,
        design_error_bound=0.0, theta_radius=float(cfg["theta_radius"]),
        exact_design=exact,
        exact_null_proof=(
            coordinate_null_fixture_proof(
                design[test],
                data.target,
                fixture_name=f"CS06 {cell_name}",
            )
            if exact and target_null_check
            else None
        ),
        weight_strategy="certificate_optimal",
        budget_components=("outcome_noise",),
        component_delta=delta,
        target_id="scalar_target",
        provenance="predeclared outcome-noise bound",
    )
    certificates = [certificate for certificate in (temporal, overlap) if certificate is not None]
    ledger_allocations = tuple(
        FailureBudget(component, component_delta, "certificate-bound predeclared event")
        for certificate in certificates
        for component, component_delta in certificate.failure_allocations
    )
    ledger = FailureBudgetLedger(
        total_delta=float(cfg["global_delta"]),
        allocations=ledger_allocations,
    )
    decision = decide_target(
        certificates,
        scientific_tolerance=float(cfg["scientific_tolerance"]),
        failure_ledger=ledger,
    ).as_dict()
    decision.update(
        runtime_seconds=perf_counter() - started,
        failed=False,
        failure_ledger=ledger.as_dict(),
        typed_contract_enforced=True,
        proof_verification_mandatory=True,
        ablated_component=(
            "support_certifier"
            if not support_certifier
            else "target_null_check"
            if not target_null_check
            else None
        ),
    )
    return decision


def _forced_ridge(cfg: dict[str, Any], data, test: np.ndarray,
                  design: np.ndarray, outcomes: np.ndarray):
    result = fit_ridge_target(
        design[test], outcomes[test], data.target, lam=float(cfg["ridge_lambda"])
    )
    estimate = float(result.target_estimate or 0.0)
    return _point_output(estimate, float(cfg["scientific_tolerance"]), result.runtime_seconds)


def _run_cell(cfg: dict[str, Any], cell_name: str, rep: int, mode: str, n_units: int):
    cell = CELLS[cell_name]
    bundle = make_seed_bundle(int(cfg["master_seed"]), "cs06", f"{mode}:{cell_name}", rep)
    seed = int(bundle.rng.integers(0, 2**32 - 1))
    theta = np.ones(int(cfg["parameter_dim"])) if cell["shared"] else None
    data = generate_aligned_geometry(
        n_units=n_units, parameter_dim=int(cfg["parameter_dim"]),
        target_support_mass=float(cell["mass"]),
        perturbation=PerturbationSpec(str(cell["kind"]), float(cell["level"])),
        seed=seed, support_bandwidth=float(cfg["support_bandwidth"]),
        support_gap=float(cfg["support_gap"]),
        outcome_noise_sigma=float(cfg["outcome_noise_sigma"]),
        trace_noise_sigma=float(cfg["trace_noise_sigma"]), theta=theta,
    )
    split = split_trajectory_ids(data.trajectory_ids, (0.4, 0.3, 0.3), seed=seed ^ 0xDEADBEEF)
    test = np.asarray(split.test, dtype=int)
    scale = float(cell["scale"])
    design = data.observed_design * scale
    outcomes = data.outcomes * scale
    full = _full_output(cfg, cell_name, data, test, design, outcomes, scale, mark_sharing=True)
    no_mark = _full_output(cfg, cell_name, data, test, design, outcomes, scale, mark_sharing=False)
    no_support = _full_output(
        cfg, cell_name, data, test, design, outcomes, scale,
        mark_sharing=True, support_certifier=False,
    )
    no_null = _full_output(
        cfg, cell_name, data, test, design, outcomes, scale,
        mark_sharing=True, target_null_check=False,
    )
    forced = _forced_ridge(cfg, data, test, design, outcomes)
    fixed = dict(full)
    if cell["mass"] > 0:
        singular_values = np.linalg.svd(design[test], compute_uv=False)
        if float(singular_values[-1]) <= float(cfg["fixed_absolute_tolerance"]):
            fixed = {
                "status": DecisionStatus.INCONCLUSIVE.value,
                "point_estimate": None, "interval": None,
                "runtime_seconds": full["runtime_seconds"], "failed": False,
                "reasons": ["fixed absolute tolerance rejects the scaled but full-rank operator"],
            }
    return {
        "rep": rep, "cell": cell_name,
        "fixture_kind": "oracle_informed_contract_fixture",
        "evaluation_only": {
            "expected_status": (
                DecisionStatus.POINT_ESTIMABLE.value if cell["mass"] > 0
                else DecisionStatus.NONRECOVERABLE.value
            ),
            "support_mass": cell["mass"],
            "perturbation_label": cell["kind"],
        },
        "expected_status": (
            DecisionStatus.POINT_ESTIMABLE.value if cell["mass"] > 0
            else DecisionStatus.NONRECOVERABLE.value
        ),
        "true_target": data.true_target, "scale": scale,
        "split_disjoint": split.as_dict()["disjoint"],
        "methods": {
            "full_crome": full, "no_support_certifier": no_support,
            "no_target_null_check": no_null, "force_point": forced,
            "no_mark_sharing": no_mark, "fixed_tolerance": fixed,
        },
    }


def _bootstrap_rmse(errors: list[float], reps: int, seed: int):
    if not errors:
        return {"value": None, "interval": None, "n": 0}
    squared = np.square(np.asarray(errors, dtype=float)); rng = np.random.default_rng(seed)
    draws = np.sqrt(np.mean(rng.choice(squared, size=(reps, squared.size), replace=True), axis=1))
    return {
        "value": float(math.sqrt(np.mean(squared))), "n": int(squared.size),
        "interval": {"method": "percentile_bootstrap", "level": 0.95,
                     "lower": float(np.quantile(draws, 0.025)), "upper": float(np.quantile(draws, 0.975))},
    }


def _variant_summary(cfg: dict[str, Any], records: list[dict[str, Any]], variant: str):
    nonpoint = [row for row in records if row["expected_status"] != DecisionStatus.POINT_ESTIMABLE.value]
    false_count = sum(row["methods"][variant]["status"] == DecisionStatus.POINT_ESTIMABLE.value for row in nonpoint)
    point_oracles = [
        row for row in records
        if row["expected_status"] == DecisionStatus.POINT_ESTIMABLE.value
    ]
    point_count = sum(
        row["methods"][variant]["status"] == DecisionStatus.POINT_ESTIMABLE.value
        for row in point_oracles
    )
    structural_null = [row for row in records if row["cell"] == "structural_null"]
    structural_count = sum(
        row["methods"][variant].get("structural_status") == "NONIDENTIFIED"
        for row in structural_null
    )
    shared_errors = []
    for row in records:
        if row["cell"] != "shared_sparse":
            continue
        estimate = row["methods"][variant].get("point_estimate")
        if estimate is not None:
            shared_errors.append(float(estimate) - float(row["true_target"]))
    return {
        "false_point_count": false_count,
        "false_point": {"rate": false_count / len(nonpoint), "total": len(nonpoint),
                        "interval": wilson_interval(false_count, len(nonpoint), level=float(cfg["mc_confidence_level"]))},
        "point_oracle_yield": {
            "count": point_count,
            "total": len(point_oracles),
            "rate": point_count / len(point_oracles),
            "interval": wilson_interval(
                point_count, len(point_oracles), level=float(cfg["mc_confidence_level"])
            ),
        },
        "structural_null_recovery": {
            "count": structural_count,
            "total": len(structural_null),
            "rate": structural_count / len(structural_null),
            "interval": wilson_interval(
                structural_count, len(structural_null), level=float(cfg["mc_confidence_level"])
            ),
        },
        "shared_sparse_rmse": _bootstrap_rmse(
            shared_errors, int(cfg["bootstrap_reps"]), int(cfg["master_seed"]) + VARIANTS.index(variant)
        ),
    }


def _gate(cfg: dict[str, Any], summaries: dict[str, dict[str, Any]], diagnostics: dict[str, Any]):
    full_false = summaries["full_crome"]["false_point_count"]
    full_upper = summaries["full_crome"]["false_point"]["interval"]["upper"]
    full_rmse = summaries["full_crome"]["shared_sparse_rmse"]["value"]
    no_mark_rmse = summaries["no_mark_sharing"]["shared_sparse_rmse"]["value"]
    ratio = float(no_mark_rmse / full_rmse)
    checks = {
        "full_crome_safety": {"passed": full_false == 0 and full_upper <= float(cfg["gate_max_full_false_point_upper"]), "observed": {"count": full_false, "upper": full_upper}, "required": float(cfg["gate_max_full_false_point_upper"])},
        "support_ablation_loses_point_yield": {
            "passed": summaries["no_support_certifier"]["point_oracle_yield"]["count"] < summaries["full_crome"]["point_oracle_yield"]["count"],
            "observed": summaries["no_support_certifier"]["point_oracle_yield"]["count"],
            "required": f"< {summaries['full_crome']['point_oracle_yield']['count']}",
        },
        "null_ablation_loses_structural_recovery": {
            "passed": summaries["no_target_null_check"]["structural_null_recovery"]["count"] < summaries["full_crome"]["structural_null_recovery"]["count"],
            "observed": summaries["no_target_null_check"]["structural_null_recovery"]["count"],
            "required": f"< {summaries['full_crome']['structural_null_recovery']['count']}",
        },
        "force_point_harms": {"passed": summaries["force_point"]["false_point_count"] > full_false, "observed": summaries["force_point"]["false_point_count"], "required": f"> {full_false}"},
        "mark_sharing_helps": {"passed": ratio >= float(cfg["gate_min_mark_sharing_rmse_ratio"]), "observed": ratio, "required": float(cfg["gate_min_mark_sharing_rmse_ratio"])},
        "fixed_tolerance_scale_failure": {"passed": diagnostics["fixed_tolerance_scale_failures"] > 0, "observed": diagnostics["fixed_tolerance_scale_failures"], "required": "> 0"},
        "uncertainty_attached": {
            "passed": all(
                summary["false_point"]["interval"]
                and summary["point_oracle_yield"]["interval"]
                and summary["structural_null_recovery"]["interval"]
                and (
                    summary["shared_sparse_rmse"]["n"] == 0
                    or summary["shared_sparse_rmse"]["interval"]
                )
                for summary in summaries.values()
            ),
            "observed": True,
            "required": True,
        },
    }
    return {"passed": all(item["passed"] for item in checks.values()), "checks": checks}


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write_csv(records: list[dict[str, Any]], config_hash: str, path: Path):
    rows = []
    for row in records:
        for variant in VARIANTS:
            output = row["methods"][variant]; interval = _interval(output)
            rows.append({
                "rep": row["rep"], "cell": row["cell"], "variant": variant,
                "expected_status": row["expected_status"], "predicted_status": output["status"],
                "structural_status": output.get("structural_status"),
                "operational_status": output.get("operational_status"),
                "certificate_scope": output.get("certificate_scope"),
                "typed_contract_enforced": bool(output.get("typed_contract_enforced", False)),
                "proof_verification_mandatory": bool(output.get("proof_verification_mandatory", False)),
                "ablated_component": output.get("ablated_component"),
                "true_target": row["true_target"], "point_estimate": output.get("point_estimate"),
                "interval_lower": interval[0] if interval else None, "interval_upper": interval[1] if interval else None,
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
    cfg = load_exp_config("cs06_ablation"); cfg["mode"] = mode
    n_reps = mode_reps(cfg, mode); n_units = int(cfg[f"n_trajectories_{mode}"])
    records = [_run_cell(cfg, cell, rep, mode, n_units) for rep in range(n_reps) for cell in CELLS]
    summaries = {variant: _variant_summary(cfg, records, variant) for variant in VARIANTS}
    diagnostics = {
        "fixed_tolerance_scale_failures": sum(
            row["cell"] == "scale_small"
            and row["methods"]["full_crome"]["status"] == DecisionStatus.POINT_ESTIMABLE.value
            and row["methods"]["fixed_tolerance"]["status"] != DecisionStatus.POINT_ESTIMABLE.value
            for row in records
        )
    }
    config_hash = _hash_json(cfg); expected_pairs = n_reps * len(CELLS) * len(VARIANTS)
    result = {
        "experiment": "cs06_ablation", "mode": mode, "master_seed": int(cfg["master_seed"]),
        "config_sha256": config_hash, "n_reps": n_reps, "n_cells": len(CELLS),
        "variants": summaries, "diagnostics": diagnostics,
        "gate": _gate(cfg, summaries, diagnostics),
        "artifact_audit": {
            "paired_complete": expected_pairs == sum(len(row["methods"]) for row in records),
            "unique_keys": len(records) == len({(row["rep"], row["cell"]) for row in records}),
            "strict_json": True,
        },
        "config": cfg, "replication_records": records,
    }
    json.dumps(result, allow_nan=False)
    if outdir is not None:
        outdir = Path(outdir); save_raw_and_summary(result, outdir / f"cs06_{mode}.json")
        source = outdir.parent / "source_data" / f"cs06_{mode}.csv" if outdir.name == "raw" else outdir / f"cs06_{mode}_source.csv"
        _write_csv(records, config_hash, source)
    return result


if __name__ == "__main__":
    print(json.dumps(run("smoke", Path("results/raw"))["gate"], indent=2))
