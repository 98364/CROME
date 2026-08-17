"""Parameter and functional estimators for the mark-aware design."""

from __future__ import annotations

import numpy as np


def estimate_theta(A: np.ndarray, Z: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Min-norm least squares via SVD pseudoinverse."""
    return np.linalg.lstsq(A, Z, rcond=tol)[0]


def ridge_theta(A: np.ndarray, Z: np.ndarray, lam: float = 1e-2) -> np.ndarray:
    """Ridge: (A'A + lam I)^{-1} A'Z — regularization-driven allocation when singular."""
    AtA = A.T @ A
    p = AtA.shape[0]
    return np.linalg.solve(AtA + lam * np.eye(p), A.T @ Z)


def prediction(A: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return A @ theta
