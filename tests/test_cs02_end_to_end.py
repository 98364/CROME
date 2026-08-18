import csv
import json

import numpy as np

from experiments import cs02_split_calibrated as cs02


def test_ridge_interval_handles_zero_lambda_structural_null_design():
    design = np.array([[1.0, 1.0], [2.0, 2.0]])

    interval = cs02._inverse_interval(
        design,
        np.array([1.0, 0.0]),
        estimate=0.5,
        noise_norm_bound=0.1,
        method="ridge",
        ridge_lambda=0.0,
        tsvd_rank=1,
    )

    assert interval is not None
    assert interval[0] < 0.5 < interval[1]


def test_cs02_records_disjoint_split_provenance_and_all_replications():
    out = cs02.run(mode="smoke", outdir=None)

    assert out["crome_weight_strategy"] == "certificate_optimal"
    assert out["n_reps"] == len(out["replication_records"])
    assert all(row["split_audit"]["disjoint"] for row in out["replication_records"])
    assert set(out["methods"]) == {"crome", "naive_boundary", "ridge", "tsvd"}
    assert all(len(row["regimes"]) == 4 for row in out["replication_records"])
    assert all(
        regime["methods"]["crome"]["weight_strategy"] == "certificate_optimal"
        for row in out["replication_records"]
        for regime in row["regimes"]
    )
    expected_rows = out["n_reps"] * 4
    assert all(summary["n_total"] == expected_rows for summary in out["methods"].values())


def test_cs02_gate_is_computed_from_coverage_and_false_point_checks():
    out = cs02.run(mode="smoke", outdir=None)
    gate = out["main_readiness_gate"]

    assert "coverage_lower_bound" in gate["checks"]
    assert "false_point_upper_bound" in gate["checks"]
    assert "split_integrity" in gate["checks"]
    assert gate["main_ready"] == all(item["passed"] for item in gate["checks"].values())


def test_cs02_typed_producers_reach_the_three_declared_public_states():
    out = cs02.run(mode="smoke", outdir=None)
    by_expected = {}
    for replication in out["replication_records"]:
        for regime in replication["regimes"]:
            by_expected.setdefault(regime["expected_status"], set()).add(
                regime["methods"]["crome"]["status"]
            )

    for expected in ("POINT_ESTIMABLE", "SET_ESTIMABLE", "NONRECOVERABLE"):
        assert by_expected[expected] == {expected}


def test_cs02_uses_same_target_and_calibration_before_test():
    out = cs02.run(mode="smoke", outdir=None)

    for replication in out["replication_records"]:
        for regime in replication["regimes"]:
            assert regime["target_definition"] == "C @ theta"
            assert regime["data_roles"]["hyperparameter_selection"] == "train"
            assert regime["data_roles"]["error_budget_calibration"] == "calibration"
            assert regime["data_roles"]["final_estimation"] == "test"


def test_cs02_saves_raw_summary_and_source_table(tmp_path):
    out = cs02.run(mode="smoke", outdir=tmp_path)

    raw_path = tmp_path / "cs02_smoke.json"
    summary_path = tmp_path / "cs02_smoke_summary.json"
    source_path = tmp_path / "cs02_smoke_source.csv"
    assert raw_path.exists()
    assert summary_path.exists()
    assert source_path.exists()
    assert len(json.loads(raw_path.read_text())["replication_records"]) == out["n_reps"]
    assert "replication_records" not in json.loads(summary_path.read_text())
    with source_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == out["n_reps"] * 4 * 4
    assert len({(row["rep"], row["regime"], row["method"]) for row in rows}) == len(rows)
