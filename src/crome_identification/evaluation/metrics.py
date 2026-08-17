"""Unconditional paper-facing evaluation summaries."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from statistics import NormalDist
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EvaluationRow:
    expected_status: str
    predicted_status: str
    point_estimate: float | None
    interval: tuple[float, float] | None
    true_target: float
    runtime_seconds: float = 0.0
    failed: bool = False
    expected_structural_status: str | None = None
    expected_operational_status: str | None = None


def product_truth_allows_public_point(
    structural_status: str,
    operational_status: str,
) -> bool:
    """Return whether the product truth permits exposure of a public point."""
    return structural_status != "NONIDENTIFIED" and operational_status == "POINT_AT_TAU"


def _oracle_allows_public_point(row: EvaluationRow) -> bool:
    product_values = (
        row.expected_structural_status,
        row.expected_operational_status,
    )
    if product_values == (None, None):
        return row.expected_status == "POINT_ESTIMABLE"
    if None in product_values:
        raise ValueError("product truth requires both structural and operational statuses")
    return product_truth_allows_public_point(
        row.expected_structural_status,
        row.expected_operational_status,
    )


def wilson_interval(successes: int, total: int, *, level: float = 0.95) -> dict[str, Any]:
    if total <= 0:
        raise ValueError("total must be positive")
    if not 0 <= successes <= total:
        raise ValueError("successes must lie between zero and total")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between zero and one")
    probability = successes / total
    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    z2_over_n = z**2 / total
    denominator = 1.0 + z2_over_n
    center = (probability + z2_over_n / 2.0) / denominator
    half_width = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / total
            + z**2 / (4.0 * total**2)
        )
        / denominator
    )
    return {
        "method": "wilson",
        "level": level,
        "lower": max(0.0, center - half_width),
        "upper": min(1.0, center + half_width),
    }


def _rate_summary(successes: int, total: int, level: float) -> dict[str, Any]:
    return {
        "successes": successes,
        "total": total,
        "rate": successes / total if total else None,
        "interval": wilson_interval(successes, total, level=level) if total else None,
    }


def summarize_method_rows(
    rows: list[EvaluationRow],
    *,
    level: float = 0.95,
    missing_point_penalty: float = 2.0,
) -> dict[str, Any]:
    """Summarize all rows without dropping failures or abstentions."""

    if not rows:
        raise ValueError("rows must not be empty")
    if not math.isfinite(missing_point_penalty) or missing_point_penalty <= 0:
        raise ValueError("missing_point_penalty must be finite and positive")

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    false_points = 0
    nonpoint_oracles = 0
    abstentions = 0
    failures = 0
    point_oracles = 0
    point_coverage = 0
    conditional_squared_errors: list[float] = []
    unconditional_squared_errors: list[float] = []
    widths: list[float] = []
    runtimes: list[float] = []

    for row in rows:
        confusion[row.expected_status][row.predicted_status] += 1
        failures += int(row.failed)
        abstentions += int(row.predicted_status == "INCONCLUSIVE" or row.failed)
        runtimes.append(float(row.runtime_seconds))

        oracle_allows_point = _oracle_allows_public_point(row)
        if not oracle_allows_point:
            nonpoint_oracles += 1
            false_points += int(row.predicted_status == "POINT_ESTIMABLE")

        interval_valid = False
        if row.interval is not None:
            lower, upper = map(float, row.interval)
            interval_valid = (
                math.isfinite(lower) and math.isfinite(upper) and lower <= upper
            )
            if interval_valid:
                widths.append(upper - lower)

        if oracle_allows_point:
            point_oracles += 1
            if interval_valid:
                lower, upper = row.interval or (math.inf, -math.inf)
                point_coverage += int(lower <= row.true_target <= upper)
            if row.point_estimate is not None and math.isfinite(row.point_estimate):
                squared_error = (float(row.point_estimate) - row.true_target) ** 2
                conditional_squared_errors.append(squared_error)
                unconditional_squared_errors.append(squared_error)
            else:
                unconditional_squared_errors.append(missing_point_penalty**2)

    accuracy_count = sum(
        predictions.get(expected, 0) for expected, predictions in confusion.items()
    )
    false_point = _rate_summary(false_points, nonpoint_oracles, level)
    coverage = _rate_summary(point_coverage, point_oracles, level)
    abstention = _rate_summary(abstentions, len(rows), level)
    return {
        "n_total": len(rows),
        "status_accuracy": accuracy_count / len(rows),
        "confusion": {
            expected: dict(predictions) for expected, predictions in confusion.items()
        },
        "false_point_count": false_points,
        "false_public_point_count": false_points,
        "nonpoint_oracle_count": nonpoint_oracles,
        "false_point_rate": false_point["rate"],
        "false_point_interval": false_point["interval"],
        "abstention_count": abstentions,
        "abstention_rate": abstention["rate"],
        "abstention_interval": abstention["interval"],
        "failure_count": failures,
        "point_oracle_count": point_oracles,
        "point_coverage_count": point_coverage,
        "point_coverage_rate": coverage["rate"],
        "point_coverage_interval": coverage["interval"],
        "target_rmse_conditional": (
            float(math.sqrt(np.mean(conditional_squared_errors)))
            if conditional_squared_errors
            else None
        ),
        "target_rmse_unconditional": (
            float(math.sqrt(np.mean(unconditional_squared_errors)))
            if unconditional_squared_errors
            else None
        ),
        "mean_interval_width": float(np.mean(widths)) if widths else None,
        "runtime_seconds": {
            "mean": float(np.mean(runtimes)),
            "p95": float(np.quantile(runtimes, 0.95)),
        },
    }
