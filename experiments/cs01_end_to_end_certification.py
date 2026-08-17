"""CS01: end-to-end four-state CROME certification benchmark."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from crome_identification.certification import (
    DecisionStatus,
    TargetInterval,
    certify_overlap_target,
    certify_temporal_conditional,
    certify_temporal_design_known,
    certify_temporal_gap_conditional,
    decide_target,
)
from crome_identification.identification.holder_sets import multi_lag_intersection
from crome_identification.seeding import make_seed_bundle

from ._common import load_exp_config, mode_reps, save_raw_and_summary
from ._proof_fixtures import exact_null_fixture_proof


_ORACLE_STATUSES = (
    DecisionStatus.POINT_ESTIMABLE,
    DecisionStatus.SET_ESTIMABLE,
    DecisionStatus.NONRECOVERABLE,
    DecisionStatus.INCONCLUSIVE,
)


def _four_regime_decisions(
    cfg: dict[str, Any], mode: str, rep: int
) -> list[dict[str, Any]]:
    rng = make_seed_bundle(int(cfg["master_seed"]), "cs01", mode, rep).rng
    tolerance = float(cfg["scientific_tolerance"])
    observed_point = 1.0 + float(rng.uniform(-0.005, 0.005))

    temporal_point = certify_temporal_design_known(
        True,
        lags=np.array([0.01]),
        responses=np.array([observed_point]),
        bandwidth=0.01,
        holder_L=0.0,
        holder_alpha=1.0,
        response_error_bound=float(cfg["point_error_radius"]),
        provenance="oracle-informed CS01 accumulation fixture",
    )
    overlap_point = certify_overlap_target(
        np.eye(2),
        np.array([observed_point, -0.5]),
        np.array([1.0, 0.0]),
        noise_radius=float(cfg["overlap_noise_radius"]),
        design_error_bound=0.0,
        theta_radius=3.0,
        exact_design=True,
        deterministic_uncertainty=True,
        target_id="scalar_target",
        provenance="oracle-informed CS01 identity-operator fixture",
    )
    point = decide_target(
        [temporal_point, overlap_point], scientific_tolerance=tolerance
    )

    holder_set = multi_lag_intersection(
        np.array([1.0, 1.0]),
        np.array([0.5, 1.0]),
        L=1.0,
        alpha=1.0,
    )
    temporal_set = certify_temporal_gap_conditional(
        np.array([0.5, 1.0]),
        np.array([1.0, 1.0]),
        holder_L=1.0,
        holder_alpha=1.0,
        response_error_bound=0.0,
        deterministic_uncertainty=True,
        provenance="oracle-informed CS01 anchored-set fixture",
    )
    bounded_set = decide_target([temporal_set], scientific_tolerance=tolerance)

    null_operator = np.array([[1.0, 1.0]])
    null_target = np.array([1.0, -1.0])
    overlap_null = certify_overlap_target(
        null_operator,
        np.array([0.5]),
        null_target,
        noise_radius=0.0,
        design_error_bound=0.0,
        theta_radius=3.0,
        exact_design=True,
        exact_null_proof=exact_null_fixture_proof(
            null_operator,
            null_target,
            np.array([1.0, -1.0]),
            fixture_name="CS01 structural-null regime",
        ),
        target_id="scalar_target",
        provenance="oracle-informed CS01 structural-null fixture",
    )
    nonrecoverable = decide_target([overlap_null], scientific_tolerance=tolerance)

    weak_support = certify_temporal_conditional(
        np.array([0.8, 0.9]),
        np.array([1.0, 1.0]),
        bandwidth=0.1,
        n_units=20,
        lower_mass_c=1.0,
        lower_mass_beta=1.0,
        holder_L=1.0,
        holder_alpha=1.0,
        noise_scale=0.1,
        delta=0.05,
        provenance="oracle-informed CS01 insufficient-mass fixture",
    )
    inconclusive = decide_target([weak_support], scientific_tolerance=tolerance)

    decisions = (point, bounded_set, nonrecoverable, inconclusive)
    records = []
    for oracle, decision in zip(_ORACLE_STATUSES, decisions, strict=True):
        records.append(
            {
                "oracle_status": oracle.value,
                "crome": decision.as_dict(),
                "naive": {
                    "status": DecisionStatus.POINT_ESTIMABLE.value,
                    "point_estimate": observed_point,
                    "reason": "always return a fitted scalar",
                },
            }
        )
    return records


def _method_summary(records: list[dict[str, Any]], method: str) -> dict[str, Any]:
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    correct = 0
    false_points = 0
    nonpoint_oracles = 0
    point_leakage = 0

    for record in records:
        oracle = record["oracle_status"]
        output = record[method]
        predicted = output["status"]
        confusion[oracle][predicted] += 1
        correct += int(predicted == oracle)
        if oracle != DecisionStatus.POINT_ESTIMABLE.value:
            nonpoint_oracles += 1
            false_points += int(predicted == DecisionStatus.POINT_ESTIMABLE.value)
        if method == "crome" and predicted != DecisionStatus.POINT_ESTIMABLE.value:
            point_leakage += int(output["point_estimate"] is not None)

    total = len(records)
    return {
        "status_accuracy": correct / total,
        "false_point_rate": false_points / nonpoint_oracles,
        "false_point_count": false_points,
        "nonpoint_oracle_count": nonpoint_oracles,
        "point_leakage_count": point_leakage if method == "crome" else None,
        "confusion": {
            oracle: dict(predictions) for oracle, predictions in confusion.items()
        },
    }


def _robustness_check(cfg: dict[str, Any]) -> dict[str, Any]:
    common = {
        "Ahat": np.eye(2),
        "y": np.array([1.0, 2.0]),
        "C": np.array([1.0, 0.0]),
        "noise_radius": float(cfg["overlap_noise_radius"]),
        "theta_radius": 3.0,
        "exact_design": False,
        "deterministic_uncertainty": True,
        "target_id": "scalar_target",
        "provenance": "CS01 deterministic bounded-error sensitivity fixture",
    }
    tight = certify_overlap_target(**common, design_error_bound=0.0)
    robust = certify_overlap_target(
        **common,
        design_error_bound=float(cfg["robust_design_error_bound"]),
    )
    tight_decision = decide_target(
        [tight], scientific_tolerance=float(cfg["scientific_tolerance"])
    )
    robust_decision = decide_target(
        [robust], scientific_tolerance=float(cfg["scientific_tolerance"])
    )
    return {
        "tight_status": tight_decision.status.value,
        "robust_status": robust_decision.status.value,
        "tight_radius": tight.error_radius,
        "robust_radius": robust.error_radius,
        "decision_changed": tight_decision.status is not robust_decision.status,
    }


def run(mode: str = "smoke", outdir: Path | None = None) -> dict[str, Any]:
    cfg = load_exp_config("cs_end_to_end")
    cfg["mode"] = mode
    n_reps = mode_reps(cfg, mode)
    replication_records = [
        {"rep": rep, "regimes": _four_regime_decisions(cfg, mode, rep)}
        for rep in range(n_reps)
    ]
    flat_records = [
        record
        for replication in replication_records
        for record in replication["regimes"]
    ]
    crome = _method_summary(flat_records, "crome")
    naive = _method_summary(flat_records, "naive")
    robustness = _robustness_check(cfg)
    summary = {
        "experiment": "cs01_end_to_end_certification",
        "fixture_kind": "oracle_informed_contract_fixture",
        "mode": mode,
        "master_seed": int(cfg["master_seed"]),
        "n_reps": n_reps,
        "n_regime_evaluations": len(flat_records),
        "oracle_statuses": [status.value for status in _ORACLE_STATUSES],
        "config": cfg,
        "crome": crome,
        "naive": naive,
        "robustness_check": robustness,
        "gate_metrics": {
            "all_four_states_exercised": len(set(summary_statuses(flat_records))) == 4,
            "no_point_leakage": crome["point_leakage_count"] == 0,
            "lower_false_point_rate_than_naive": (
                crome["false_point_rate"] < naive["false_point_rate"]
            ),
            "robust_budget_changes_decision": robustness["decision_changed"],
        },
        "replication_records": replication_records,
    }
    if outdir is not None:
        save_raw_and_summary(summary, Path(outdir) / f"cs01_{mode}.json")
    return summary


def summary_statuses(records: list[dict[str, Any]]) -> list[str]:
    return [record["oracle_status"] for record in records]


if __name__ == "__main__":
    print(run("smoke", Path("results/raw")))
