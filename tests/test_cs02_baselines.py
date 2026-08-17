import numpy as np
import pytest

from crome_identification.baselines import (
    fit_naive_boundary,
    fit_ridge_target,
    fit_tsvd_target,
    select_inverse_hyperparameters,
)


def test_naive_boundary_estimates_same_scalar_target():
    q = np.linspace(0.01, 0.5, 100)

    out = fit_naive_boundary(q, 1.0 + 0.2 * q, bandwidth=0.5)

    assert out.success
    assert out.target_estimate == pytest.approx(1.0, abs=1e-8)
    assert out.hyperparameters == {"bandwidth": 0.5}


def test_ridge_and_tsvd_return_c_theta_not_full_parameter():
    A = np.eye(3)
    y = np.array([1.0, 2.0, 3.0])
    C = np.array([1.0, 0.0, 0.0])

    ridge = fit_ridge_target(A, y, C, lam=0.0)
    tsvd = fit_tsvd_target(A, y, C, rank=3)

    assert ridge.target_estimate == pytest.approx(1.0)
    assert tsvd.target_estimate == pytest.approx(1.0)
    assert np.isscalar(ridge.target_estimate)
    assert np.isscalar(tsvd.target_estimate)


def test_inverse_hyperparameters_are_selected_deterministically_from_supplied_rows():
    rng = np.random.default_rng(5)
    A = rng.normal(size=(60, 3))
    y = A @ np.array([1.0, -0.5, 0.25]) + rng.normal(0.0, 0.01, size=60)
    ridge_grid = [0.0, 0.01, 0.1]
    rank_grid = [1, 2, 3]

    first = select_inverse_hyperparameters(A, y, ridge_grid, rank_grid, n_splits=5)
    second = select_inverse_hyperparameters(A, y, ridge_grid, rank_grid, n_splits=5)

    assert first == second
    assert first.ridge_lambda in ridge_grid
    assert first.tsvd_rank in rank_grid


def test_baselines_return_explicit_failures_for_unusable_inputs():
    boundary = fit_naive_boundary(np.array([0.8, 0.9]), np.array([1.0, 1.0]), bandwidth=0.2)

    assert not boundary.success
    assert boundary.target_estimate is None
    with pytest.raises(ValueError, match="rank"):
        fit_tsvd_target(np.eye(2), np.ones(2), np.array([1.0, 0.0]), rank=3)
