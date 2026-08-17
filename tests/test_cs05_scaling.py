import math

from experiments.cs05_scaling import _scaling_specs, run


def test_scaling_specs_cover_trajectories_parameters_events_and_endpoints():
    cfg = {
        "n_trajectories_smoke": [200, 500],
        "parameter_dims_smoke": [3, 10],
        "events_per_trajectory_smoke": [1, 5],
        "endpoints_per_trajectory_smoke": [1, 4],
        "fixed_events": 5,
        "fixed_endpoints": 4,
        "fixed_n_trajectories": 500,
        "fixed_parameter_dim": 10,
    }
    specs = _scaling_specs(cfg, "smoke")
    assert len({spec.key for spec in specs}) == len(specs)
    assert {spec.n_trajectories for spec in specs} >= {200, 500}
    assert {spec.parameter_dim for spec in specs} >= {3, 10}
    assert {spec.events_per_trajectory for spec in specs} >= {1, 5}
    assert {spec.endpoints_per_trajectory for spec in specs} >= {1, 4}


def test_cs05_smoke_records_peak_memory_accuracy_and_slopes(tmp_path):
    result = run("smoke", tmp_path)
    assert result["artifact_audit"]["unique_keys"]
    assert result["gate"]["checks"]["cell_completeness"]["passed"]
    for row in result["replication_records"]:
        assert row["exact"]["peak_python_bytes"] > 0
        assert row["approximate"]["peak_python_bytes"] > 0
        assert row["input_matrix_bytes"] > 0
        assert math.isfinite(row["exact"]["runtime_seconds"])
        assert row["exact"]["status"] == row["approximate"]["status"]
        assert row["target_discrepancy"] <= result["config"]["approximation_budget"]
    assert math.isfinite(result["empirical_slopes"]["runtime_vs_rows"])
    assert math.isfinite(result["empirical_slopes"]["memory_vs_rows"])
