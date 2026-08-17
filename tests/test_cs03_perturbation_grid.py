import json

import numpy as np

from crome_identification.certification import calibrate_bounded_design_error
from experiments.cs03_perturbation_grid import (
    _cell_specs,
    _semantic_digest,
    run,
)


def test_bounded_design_calibration_uses_only_supplied_calibration_rows():
    reference = np.zeros((80, 3))
    observed = reference.copy()
    observed[:8, 0] = 0.2
    result = calibrate_bounded_design_error(
        reference,
        observed,
        n_test=40,
        row_error_bound=1.5,
        delta=0.05,
    )
    assert result.operator_error_bound > 0
    assert result.n_calibration == 80
    assert result.n_test == 40
    assert "bounded independent row errors" in result.assumptions


def test_cs03_grid_is_continuous_unique_and_contains_all_perturbations():
    cells = _cell_specs(
        {
            "support_masses": [0.0, 0.25, 0.8],
            "perturbation_levels": {
                "clean": [0.0],
                "timestamp_jitter": [0.01],
                "rounding": [0.1],
                "basis_error": [0.05],
                "mark_noise": [0.1],
                "baseline_error": [0.05],
            },
        }
    )
    keys = {(c.support_mass, c.perturbation.kind, c.perturbation.level) for c in cells}
    assert len(keys) == len(cells)
    assert {c.support_mass for c in cells} == {0.0, 0.25, 0.8}
    assert {c.perturbation.kind for c in cells} == {
        "clean",
        "timestamp_jitter",
        "rounding",
        "basis_error",
        "mark_noise",
        "baseline_error",
    }


def test_cs03_smoke_is_complete_reproducible_and_auditable(tmp_path):
    first = run("smoke", tmp_path / "first")
    second = run("smoke", tmp_path / "second")
    assert _semantic_digest(first) == _semantic_digest(second)
    keys = [
        (row["rep"], row["support_mass"], row["perturbation"], row["level"])
        for row in first["replication_records"]
    ]
    assert len(keys) == len(set(keys))
    assert first["artifact_audit"]["strict_json"]
    assert first["artifact_audit"]["unique_keys"]
    assert set(first["methods"]) == {
        "crome_optimal",
        "crome_current",
        "matched_uncertainty",
        "naive_boundary",
        "ridge",
        "tsvd",
    }
    statuses = {
        row["methods"]["crome_optimal"]["status"]
        for row in first["replication_records"]
    }
    assert {"POINT_ESTIMABLE", "NONRECOVERABLE", "INCONCLUSIVE"} <= statuses
    product_statuses = {
        (
            row["methods"]["crome_optimal"]["structural_status"],
            row["methods"]["crome_optimal"]["operational_status"],
        )
        for row in first["replication_records"]
    }
    assert ("NONIDENTIFIED", "SET") in product_statuses
    expected_product_statuses = {
        "_x_".join(
            (
                row["methods"]["crome_optimal"]["structural_status"],
                row["methods"]["crome_optimal"]["operational_status"],
                    row["methods"]["crome_optimal"]["certificate_scope"],
            )
        )
        for row in first["replication_records"]
    }
    assert set(
        first["gate"]["checks"]["product_status_transition"]["observed"]
    ) == expected_product_statuses
    for row in first["replication_records"]:
        optimal = row["methods"]["crome_optimal"].get("certificate_radius")
        current = row["methods"]["crome_current"].get("certificate_radius")
        if optimal is not None and current is not None:
            assert optimal <= current + 1e-8
    optimal_points = sum(
        row["methods"]["crome_optimal"]["status"] == "POINT_ESTIMABLE"
        for row in first["replication_records"]
    )
    matched_points = sum(
        row["methods"]["matched_uncertainty"]["status"] == "POINT_ESTIMABLE"
        for row in first["replication_records"]
    )
    assert matched_points == optimal_points
    assert first["story_metrics"]["rq1"]["radius_noninferiority"]
    assert (
        first["story_metrics"]["rq1"]["optimal"]["point_yield"]
        > first["story_metrics"]["rq1"]["current"]["point_yield"]
    )
    assert first["story_metrics"]["rq2"]["point_outputs_matched"]
    json.dumps(first, allow_nan=False)
