"""One-sided local linear regression at the zero lag boundary."""

from __future__ import annotations

import numpy as np


def epanechnikov_one_sided(u: np.ndarray) -> np.ndarray:
    """Support u ∈ [0, 1], one-sided for q ≥ 0 near 0."""
    u = np.asarray(u, dtype=float)
    w = np.zeros_like(u)
    mask = (u >= 0) & (u <= 1)
    w[mask] = 0.75 * (1.0 - u[mask] ** 2)
    return w


def one_sided_local_linear(
    lags: np.ndarray,
    y: np.ndarray,
    h: float,
    *,
    x0: float = 0.0,
) -> dict[str, float]:
    """
    Estimate r(0+) by local linear on positive lags with one-sided kernel.
    Returns intercept estimate at x0 (default 0).
    """
    lags = np.asarray(lags, dtype=float)
    y = np.asarray(y, dtype=float)
    if h <= 0:
        raise ValueError("bandwidth h must be positive")
    mask = lags > 0
    lags, y = lags[mask], y[mask]
    if lags.size < 2:
        return {"estimate": float("nan"), "n_eff": 0.0}

    u = (lags - x0) / h
    w = epanechnikov_one_sided(u)
    # only right side
    w = np.where(lags >= x0, w, 0.0)
    if w.sum() <= 0:
        return {"estimate": float("nan"), "n_eff": 0.0}

    # weighted least squares: y ~ a + b (lag - x0)
    X = np.column_stack([np.ones_like(lags), lags - x0])
    # Weight rows directly.  Materializing diag(w) is quadratic in the number
    # of lags and makes the pre-specified n=2,000 scan unnecessarily expensive.
    XtWX = X.T @ (w[:, None] * X)
    XtWy = X.T @ (w * y)
    try:
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]
    return {"estimate": float(beta[0]), "n_eff": float((w > 0).sum()), "slope": float(beta[1])}


def select_bandwidth_loocv_trajectory(
    lags: np.ndarray,
    y: np.ndarray,
    traj_id: np.ndarray,
    grid: list[float],
) -> float:
    """Leave-one-trajectory-out MSE on positive lags (simple CV)."""
    lags = np.asarray(lags, dtype=float)
    y = np.asarray(y, dtype=float)
    traj_id = np.asarray(traj_id)
    best_h, best_mse = grid[0], np.inf
    for h in grid:
        errs = []
        for tid in np.unique(traj_id):
            train = traj_id != tid
            test = traj_id == tid
            if train.sum() < 2 or test.sum() == 0:
                continue
            fit = one_sided_local_linear(lags[train], y[train], h)
            if not np.isfinite(fit["estimate"]):
                continue
            # evaluate intercept as prediction near zero; use residuals at test lags via local fit
            # simplified: compare fitted boundary to mean of test y at smallest lags
            pred = fit["estimate"]
            errs.append((np.mean(y[test]) - pred) ** 2)
        if errs:
            mse = float(np.mean(errs))
            if mse < best_mse:
                best_mse, best_h = mse, h
    return float(best_h)


def select_bandwidth_group_kfold(
    lags: np.ndarray,
    y: np.ndarray,
    traj_id: np.ndarray,
    grid: list[float],
    *,
    n_splits: int = 5,
) -> float:
    """Select a boundary bandwidth by deterministic trajectory-level K-fold CV.

    Complete trajectories, rather than individual lags, are assigned to folds.
    The routine evaluates the fitted local line only on positive validation
    lags within the candidate bandwidth.  It is intended as a computationally
    bounded empirical diagnostic, not as a coverage procedure.
    """
    lags = np.asarray(lags, dtype=float)
    y = np.asarray(y, dtype=float)
    traj_id = np.asarray(traj_id)
    if not grid:
        raise ValueError("bandwidth grid must be non-empty")
    groups = np.unique(traj_id)
    if groups.size < 2:
        return float(grid[0])
    n_splits = max(2, min(int(n_splits), int(groups.size)))
    fold_of_group = {group: idx % n_splits for idx, group in enumerate(groups)}
    row_folds = np.asarray([fold_of_group[group] for group in traj_id])

    best_h, best_mse = float(grid[0]), np.inf
    for h in grid:
        squared_errors: list[float] = []
        for fold in range(n_splits):
            train = row_folds != fold
            test = (row_folds == fold) & (lags > 0) & (lags <= h)
            if train.sum() < 2 or test.sum() == 0:
                continue
            fit = one_sided_local_linear(lags[train], y[train], float(h))
            if not np.isfinite(fit["estimate"]):
                continue
            prediction = fit["estimate"] + fit.get("slope", 0.0) * lags[test]
            squared_errors.extend(np.square(y[test] - prediction).tolist())
        if squared_errors:
            mse = float(np.mean(squared_errors))
            if mse < best_mse:
                best_h, best_mse = float(h), mse
    return best_h
