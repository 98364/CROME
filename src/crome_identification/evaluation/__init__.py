"""Evaluation contracts for split-calibrated CROME experiments."""

from .metrics import (
    EvaluationRow,
    product_truth_allows_public_point,
    summarize_method_rows,
    wilson_interval,
)
from .splits import TrajectorySplit, split_trajectory_ids

__all__ = [
    "EvaluationRow",
    "TrajectorySplit",
    "split_trajectory_ids",
    "product_truth_allows_public_point",
    "summarize_method_rows",
    "wilson_interval",
]
