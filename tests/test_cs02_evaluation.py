import numpy as np
import pytest

from crome_identification.evaluation import (
    EvaluationRow,
    split_trajectory_ids,
    summarize_method_rows,
    wilson_interval,
)


def test_trajectory_split_is_disjoint_complete_and_reproducible():
    first = split_trajectory_ids(np.arange(100), (0.4, 0.3, 0.3), seed=17)
    second = split_trajectory_ids(np.arange(100), (0.4, 0.3, 0.3), seed=17)

    assert first == second
    assert len(first.train) == 40
    assert len(first.calibration) == 30
    assert len(first.test) == 30
    assert not (set(first.train) & set(first.calibration))
    assert not (set(first.train) & set(first.test))
    assert not (set(first.calibration) & set(first.test))
    assert set(first.train) | set(first.calibration) | set(first.test) == set(range(100))


def test_trajectory_split_rejects_duplicate_ids_and_invalid_ratios():
    with pytest.raises(ValueError, match="unique"):
        split_trajectory_ids(np.array([1, 1, 2]), (0.4, 0.3, 0.3), seed=1)
    with pytest.raises(ValueError, match="sum to one"):
        split_trajectory_ids(np.arange(10), (0.4, 0.4, 0.4), seed=1)


def test_method_summary_counts_failures_without_conditioning_on_success():
    rows = [
        EvaluationRow("POINT_ESTIMABLE", "POINT_ESTIMABLE", 1.0, (0.9, 1.1), 1.0),
        EvaluationRow("SET_ESTIMABLE", "POINT_ESTIMABLE", 1.0, None, 1.0),
        EvaluationRow("POINT_ESTIMABLE", "INCONCLUSIVE", None, None, 1.0),
    ]

    out = summarize_method_rows(rows, level=0.95, missing_point_penalty=2.0)

    assert out["n_total"] == 3
    assert out["false_point_count"] == 1
    assert out["abstention_count"] == 1
    assert out["point_coverage_count"] == 1
    assert out["point_oracle_count"] == 2
    assert out["point_coverage_rate"] == pytest.approx(0.5)
    assert out["target_rmse_unconditional"] == pytest.approx(np.sqrt(2.0))


def test_false_public_points_use_product_truth_not_legacy_expected_status():
    rows = [
        EvaluationRow(
            "POINT_ESTIMABLE",
            "POINT_ESTIMABLE",
            1.0,
            (0.9, 1.1),
            1.0,
            expected_structural_status="NONIDENTIFIED",
            expected_operational_status="POINT_AT_TAU",
        ),
        EvaluationRow(
            "NONRECOVERABLE",
            "POINT_ESTIMABLE",
            1.0,
            (0.9, 1.1),
            1.0,
            expected_structural_status="UNKNOWN",
            expected_operational_status="POINT_AT_TAU",
        ),
    ]

    out = summarize_method_rows(rows)

    assert out["false_point_count"] == 1
    assert out["nonpoint_oracle_count"] == 1
    assert out["point_oracle_count"] == 1


def test_wilson_interval_is_finite_at_boundaries():
    zero = wilson_interval(0, 20, level=0.95)
    all_success = wilson_interval(20, 20, level=0.95)

    assert zero["lower"] == pytest.approx(0.0)
    assert 0.0 < zero["upper"] < 1.0
    assert 0.0 < all_success["lower"] < 1.0
    assert all_success["upper"] == pytest.approx(1.0)
