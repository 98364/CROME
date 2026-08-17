"""Design diagnostics: rank, Gram spectrum, condition."""

from __future__ import annotations

import numpy as np


def gram_spectrum(A: np.ndarray) -> np.ndarray:
    """Eigenvalues of normalized Gram G = A'A / N_row."""
    A = np.asarray(A, dtype=float)
    if A.shape[0] == 0:
        return np.array([])
    G = (A.T @ A) / A.shape[0]
    return np.sort(np.linalg.eigvalsh(G))[::-1]


def matrix_diagnostics(A: np.ndarray, tol: float = 1e-10) -> dict[str, float]:
    A = np.asarray(A, dtype=float)
    n, p = A.shape
    if n == 0:
        return {
            "n_row": 0.0,
            "n_col": float(p),
            "rank": 0.0,
            "full_column_rank": 0.0,
            "lambda_min": 0.0,
            "cond": np.inf,
            "effective_rank": 0.0,
        }
    s = np.linalg.svd(A, compute_uv=False)
    rank = int(np.sum(s > tol * s[0])) if s.size else 0
    full = float(rank == p and n >= p)
    eigs = gram_spectrum(A)
    lam_min = float(eigs[-1]) if eigs.size else 0.0
    if rank < p or lam_min <= 0:
        cond = np.inf
    else:
        cond = float(eigs[0] / max(eigs[-1], 1e-300))
    # effective rank via entropy of normalized spectrum
    if eigs.size and eigs.sum() > 0:
        pspec = eigs / eigs.sum()
        pspec = pspec[pspec > 0]
        ent = -np.sum(pspec * np.log(pspec))
        erank = float(np.exp(ent))
    else:
        erank = 0.0
    return {
        "n_row": float(n),
        "n_col": float(p),
        "rank": float(rank),
        "full_column_rank": full,
        "lambda_min": lam_min,
        "cond": cond,
        "effective_rank": erank,
    }
