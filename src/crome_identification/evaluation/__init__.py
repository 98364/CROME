"""Evaluation contracts for split-calibrated CROME experiments."""

from .metrics import EvaluationRow, summarize_method_rows, wilson_interval
from .splits import TrajectorySplit, split_trajectory_ids

__all__ = [
    "EvaluationRow",
    "TrajectorySplit",
    "split_trajectory_ids",
    "summarize_method_rows",
    "wilson_interval",
]
