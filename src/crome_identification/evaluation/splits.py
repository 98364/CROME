"""Deterministic trajectory-level data splitting."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class TrajectorySplit:
    train: tuple[int, ...]
    calibration: tuple[int, ...]
    test: tuple[int, ...]

    def as_dict(self) -> dict[str, list[int] | bool]:
        train = set(self.train)
        calibration = set(self.calibration)
        test = set(self.test)
        return {
            "train": list(self.train),
            "calibration": list(self.calibration),
            "test": list(self.test),
            "disjoint": not (
                train & calibration or train & test or calibration & test
            ),
        }


def split_trajectory_ids(
    trajectory_ids: np.ndarray,
    ratios: tuple[float, float, float],
    *,
    seed: int,
) -> TrajectorySplit:
    """Assign every unique trajectory exactly once to train/calibration/test."""

    ids = np.asarray(trajectory_ids)
    if ids.ndim != 1:
        raise ValueError("trajectory_ids must be one-dimensional")
    if ids.size < 3:
        raise ValueError("at least three trajectory IDs are required")
    if np.unique(ids).size != ids.size:
        raise ValueError("trajectory IDs must be unique")
    if len(ratios) != 3 or any(not math.isfinite(value) or value <= 0 for value in ratios):
        raise ValueError("split ratios must contain three finite positive values")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("split ratios must sum to one")

    raw_counts = np.asarray(ratios, dtype=float) * ids.size
    counts = np.floor(raw_counts).astype(int)
    remainder = int(ids.size - np.sum(counts))
    priority = np.argsort(-(raw_counts - counts), kind="stable")
    counts[priority[:remainder]] += 1
    if np.any(counts == 0):
        raise ValueError("each split must contain at least one trajectory")

    shuffled = np.random.default_rng(seed).permutation(ids)
    train_end = int(counts[0])
    calibration_end = train_end + int(counts[1])
    return TrajectorySplit(
        train=tuple(int(value) for value in shuffled[:train_end]),
        calibration=tuple(int(value) for value in shuffled[train_end:calibration_end]),
        test=tuple(int(value) for value in shuffled[calibration_end:]),
    )

