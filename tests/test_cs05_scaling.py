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
    assert result["crome_weight_strategy"] == "certificate_optimal"
    assert result["artifact_audit"]["unique_keys"]
    assert result["gate"]["checks"]["cell_completeness"]["passed"]
    for row in result["replication_records"]:
        assert row["exact"]["peak_python_bytes"] > 0
        assert row["approximate"]["peak_python_bytes"] > 0
        assert row["input_matrix_bytes"] > 0
        assert math.isfinite(row["exact"]["runtime_seconds"])
        assert row["exact"]["status"] == row["approximate"]["status"]
        assert row["exact"]["weight_strategy"] == "certificate_optimal"
        assert row["approximate"]["weight_strategy"] == "certificate_optimal"
        assert row["target_discrepancy"] <= result["config"]["approximation_budget"]
    assert math.isfinite(result["empirical_slopes"]["runtime_vs_rows"])
    assert math.isfinite(result["empirical_slopes"]["memory_vs_rows"])


def test_cs05_records_reproducible_runtime_environment(tmp_path):
    result = run("smoke", tmp_path)
    environment = result["runtime_environment"]
    assert environment["python"]
    assert environment["numpy"]
    assert environment["scipy"]
    assert environment["operating_system"]["system"]
    assert environment["operating_system"]["release"]
    assert environment["architecture"]
    assert environment["cpu_count"] > 0
    assert environment["physical_memory_bytes"] > 0
    assert environment["linear_algebra"]["blas"]
    assert environment["linear_algebra"]["lapack"]
    assert set(environment["thread_settings"]) == {
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    }
    assert environment["dependency_lock"]["path"] == "requirements-jss-lock.txt"
    assert len(environment["dependency_lock"]["sha256"]) == 64
