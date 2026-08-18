from pathlib import Path

import numpy as np

from experiments.cs04_online_retail import (
    _build_real_timing_geometry,
    _inject_semisynthetic,
    run,
)


DATA_PATH = Path("data/processed/online_retail_ii_events.npz")


def test_real_timing_geometry_is_built_before_outcome_injection():
    geometry = _build_real_timing_geometry(DATA_PATH, "fine_4h")
    assert not hasattr(geometry, "outcomes")
    assert geometry.n_customers == 2505
    assert geometry.target_near_zero_count > 0
    injected = _inject_semisynthetic(geometry, seed=7, outcome_sigma=0.03, trace_sigma=0.03)
    assert injected.outcomes.shape == (geometry.n_customers,)
    assert np.isclose(injected.true_target, 1.0)


def test_predeclared_real_timing_regimes_have_required_geometry():
    fine = _build_real_timing_geometry(DATA_PATH, "fine_4h")
    gap = _build_real_timing_geometry(DATA_PATH, "daily_gap")
    collapse = _build_real_timing_geometry(DATA_PATH, "target_mark_collapse")
    hidden = _build_real_timing_geometry(DATA_PATH, "rounded_hidden_time")
    assert fine.target_near_zero_count >= 100
    assert gap.target_near_zero_count == 0
    assert np.all(collapse.true_design[:, 0] == 0)
    assert not np.array_equal(hidden.true_design, hidden.observed_design)


def test_cs04_smoke_uses_all_customers_and_emits_four_statuses(tmp_path):
    result = run("smoke", tmp_path)
    assert result["crome_weight_strategy"] == "certificate_optimal"
    assert result["data_profile"]["eligible_customers"] == 2505
    assert all(row["n_customers"] == 2505 for row in result["replication_records"])
    assert all(
        row["methods"]["crome"]["weight_strategy"] == "certificate_optimal"
        for row in result["replication_records"]
    )
    statuses = {row["methods"]["crome"]["status"] for row in result["replication_records"]}
    assert statuses == {"POINT_ESTIMABLE", "NONRECOVERABLE", "INCONCLUSIVE"}
    product_statuses = {
        (
            row["methods"]["crome"]["structural_status"],
            row["methods"]["crome"]["operational_status"],
        )
        for row in result["replication_records"]
    }
    assert ("NONIDENTIFIED", "SET") in product_statuses
    assert result["artifact_audit"]["data_hashes_attached"]
    assert set(
        result["gate"]["checks"]["product_status_population"]["observed"]
    ) == {
        "UNKNOWN_x_POINT_AT_TAU",
        "NONIDENTIFIED_x_SET",
        "NONIDENTIFIED_x_INCONCLUSIVE",
        "UNKNOWN_x_INCONCLUSIVE",
    }
