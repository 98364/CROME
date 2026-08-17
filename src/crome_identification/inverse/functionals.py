"""Null-space / row-space characterization of identifiable linear functionals."""

from __future__ import annotations

import numpy as np


def null_space_basis(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Orthonormal basis for Null(A) via SVD. Columns span the null space.
    """
    A = np.asarray(A, dtype=float)
    if A.size == 0:
        p = A.shape[1] if A.ndim == 2 else 0
        return np.eye(p, dtype=float)
    u, s, vt = np.linalg.svd(A, full_matrices=True)
    rank = int(np.sum(s > tol * (s[0] if s.size else 1.0)))
    return vt[rank:].T


def is_identifiable(A: np.ndarray, C: np.ndarray, tol: float = 1e-10) -> bool:
    """
    C theta identifiable from A theta iff Null(A) ⊆ Null(C)
    iff Row(C) ⊆ Row(A).
    """
    A = np.asarray(A, dtype=float)
    C = np.asarray(C, dtype=float)
    if C.ndim == 1:
        C = C.reshape(1, -1)
    residual = identification_residual(A, C, tol=tol)
    scale = max(1.0, float(np.linalg.norm(C, ord=2)))
    return bool(residual <= max(tol, 1e-8) * scale)


def identification_residual(A: np.ndarray, C: np.ndarray, tol: float = 1e-10) -> float:
    """Return ``||C(I-A^+A)||_op``, the target identification residual.

    The residual is an identification diagnostic.  It is zero exactly when the
    rows of ``C`` lie in the row space of ``A`` (up to the declared numerical
    tolerance).  It is not a condition-number or stability diagnostic.
    """
    A = np.asarray(A, dtype=float)
    C = np.asarray(C, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a two-dimensional matrix")
    if C.ndim == 1:
        C = C.reshape(1, -1)
    if C.ndim != 2 or C.shape[1] != A.shape[1]:
        raise ValueError("C must have the same parameter dimension as A")
    projector_row = np.linalg.pinv(A, rcond=tol) @ A
    residual = C @ (np.eye(A.shape[1]) - projector_row)
    return float(np.linalg.norm(residual, ord=2))


def target_noise_amplification(
    A: np.ndarray,
    C: np.ndarray,
    tol: float = 1e-10,
) -> float | None:
    """Return ``||C A^+||_op`` only when the target is identifiable.

    ``None`` is deliberate for exact non-identification.  A small norm produced
    by the Moore--Penrose minimum-norm selection must not be interpreted as
    stable recovery of a target that varies along ``Null(A)``.
    """
    A = np.asarray(A, dtype=float)
    C = np.asarray(C, dtype=float)
    if C.ndim == 1:
        C = C.reshape(1, -1)
    if not is_identifiable(A, C, tol=tol):
        return None
    return float(np.linalg.norm(C @ np.linalg.pinv(A, rcond=tol), ord=2))


def identifiable_functional(
    A: np.ndarray,
    Z: np.ndarray,
    C: np.ndarray,
    tol: float = 1e-10,
) -> tuple[np.ndarray, bool]:
    """
    Estimate vartheta = C theta via C A^+ Z when identifiable.
    """
    A = np.asarray(A, dtype=float)
    Z = np.asarray(Z, dtype=float)
    C = np.asarray(C, dtype=float)
    if C.ndim == 1:
        C = C.reshape(1, -1)
    ok = is_identifiable(A, C, tol=tol)
    theta_hat = np.linalg.lstsq(A, Z, rcond=tol)[0]
    return C @ theta_hat, ok
