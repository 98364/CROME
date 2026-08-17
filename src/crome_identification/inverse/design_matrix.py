"""Mark-aware shared-parameter design matrix for overlap recovery."""

from __future__ import annotations

import numpy as np


def _same_time(values: np.ndarray, target: float) -> np.ndarray:
    """ULP-scale equality for timestamp conventions, not statistical closeness."""
    if values.size == 0:
        return np.zeros(0, dtype=bool)
    scale = max(1.0, abs(float(target)), float(np.max(np.abs(values))))
    atol = 4.0 * np.finfo(float).eps * scale
    return np.abs(values - target) <= atol


def mark_one_hot_features(mark: int, C: int) -> np.ndarray:
    phi = np.zeros(C, dtype=float)
    phi[int(mark)] = 1.0
    return phi


def lag_basis(q: float, L_basis: int, betas: np.ndarray | None = None) -> np.ndarray:
    """
    psi_0 = 1 (jump), psi_ell → 0 as q → 0 for ell >= 1.
    Default: 1, (e^{-b q}-1), (1-e^{-b q}), q e^{-b q} truncated to L_basis+1 terms.
    """
    q = float(q)
    if betas is None:
        betas = np.array([1.0, 0.3, 0.8])
    terms = [1.0]
    if L_basis >= 1:
        terms.append(np.exp(-betas[0] * q) - 1.0)
    if L_basis >= 2:
        terms.append(1.0 - np.exp(-betas[1] * q))
    if L_basis >= 3:
        terms.append(q * np.exp(-betas[2] * q))
    # pad / trim
    while len(terms) < L_basis + 1:
        terms.append(0.0)
    return np.asarray(terms[: L_basis + 1], dtype=float)


def build_design_matrix(
    obs_times: list[np.ndarray],
    event_times: list[np.ndarray],
    event_marks: list[np.ndarray],
    *,
    C: int,
    L_basis: int = 2,
    Qresp: float = 5.0,
    phi_dim: int | None = None,
    normalize: bool = True,
    drop_zero_lag: bool = True,
    return_column_scales: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]
    | tuple[np.ndarray, np.ndarray, list[tuple[int, int]], np.ndarray]
):
    """
    A_{ij,(a,ell)} = sum_{k: tau_ik < t_ij} phi_a(m_ik) psi_ell(t_ij - tau_ik)

    Default phi = mark one-hot (phi_dim = C).
    Drops rows with any zero lag (same timestamp).
    Returns A, row_index_map (traj_i, obs_j), and column is flattened (a, ell).
    """
    if phi_dim is None:
        phi_dim = C
    n_col = phi_dim * (L_basis + 1)
    rows: list[np.ndarray] = []
    index_map: list[tuple[int, int]] = []

    for i, (t_obs, taus, marks) in enumerate(
        zip(obs_times, event_times, event_marks, strict=True)
    ):
        t_obs = np.asarray(t_obs, dtype=float)
        taus = np.asarray(taus, dtype=float)
        marks = np.asarray(marks, dtype=int)
        for j, t in enumerate(t_obs):
            # skip if any event shares exact timestamp (zero lag)
            if drop_zero_lag and taus.size and np.any(_same_time(taus, float(t))):
                continue
            row = np.zeros(n_col, dtype=float)
            for tau, m in zip(taus, marks, strict=True):
                q = t - tau
                if q <= 0 or q > Qresp:
                    continue
                phi = mark_one_hot_features(int(m), C)
                psi = lag_basis(q, L_basis)
                # Kronecker-style: for each a, ell block
                for a in range(phi_dim):
                    for ell in range(L_basis + 1):
                        row[a * (L_basis + 1) + ell] += phi[a] * psi[ell]
            rows.append(row)
            index_map.append((i, j))

    if not rows:
        empty = (np.zeros((0, n_col)), np.zeros(0), [])
        if return_column_scales:
            return (*empty, np.ones(n_col, dtype=float))
        return empty

    A = np.vstack(rows)
    column_scales = np.ones(A.shape[1], dtype=float)
    if normalize and A.shape[0] > 0:
        # non-intercept-like columns: scale ||A.,j||_2 = sqrt(N_row)
        # treat first of each mark block (ell=0 jump columns) also scaled uniformly
        N = A.shape[0]
        target = np.sqrt(N)
        for j in range(A.shape[1]):
            nrm = np.linalg.norm(A[:, j])
            if nrm > 0:
                column_scales[j] = target / nrm
                A[:, j] *= column_scales[j]
    result = (A, np.arange(A.shape[0]), index_map)
    if return_column_scales:
        return (*result, column_scales)
    return result


def residualize_outcomes(
    Y_obs: list[np.ndarray],
    B_hat: list[np.ndarray],
    index_map: list[tuple[int, int]],
    obs_times: list[np.ndarray],
) -> np.ndarray:
    """Build Z vector aligned with design rows."""
    z = []
    # map (i,j) -> residual requires original obs index j before drop — index_map stores surviving j
    for i, j in index_map:
        z.append(float(Y_obs[i][j] - B_hat[i][j]))
    return np.asarray(z, dtype=float)
