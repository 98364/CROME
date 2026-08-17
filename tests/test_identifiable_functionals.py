import numpy as np
import pytest

from crome_identification.inverse.functionals import (
    identification_residual,
    is_identifiable,
    null_space_basis,
    target_noise_amplification,
)


def test_null_space_detects_duplicate_columns():
    A = np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 1.0], [3.0, 3.0, 0.0]])
    N = null_space_basis(A, tol=1e-10)
    assert N.shape[1] >= 1
    assert not is_identifiable(A, np.array([1.0, -1.0, 0.0]))
    assert is_identifiable(A, np.array([1.0, 1.0, 0.0]))


def test_full_rank_all_identifiable():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(20, 3))
    assert is_identifiable(A, np.eye(3))


def test_empty_design_has_full_parameter_null_space():
    A = np.zeros((0, 3))
    N = null_space_basis(A)

    assert N.shape == (3, 3)
    assert np.allclose(N, np.eye(3))
    assert not is_identifiable(A, np.array([1.0, 0.0, 0.0]))


def test_target_diagnostics_reject_pseudoinverse_amplification_for_nonidentified_target():
    A = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    single = np.array([1.0, 0.0])
    compound = np.array([1.0, 1.0])

    assert identification_residual(A, single) > 0.0
    assert target_noise_amplification(A, single) is None
    assert identification_residual(A, compound) == pytest.approx(0.0, abs=1e-10)
    assert target_noise_amplification(A, compound) is not None


def test_near_collinearity_can_amplify_contrast_more_than_compound():
    eps = 1e-3
    A = np.array([[1.0, 1.0], [0.0, eps], [1.0, 1.0 + eps]])
    contrast = np.array([1.0, -1.0])
    compound = np.array([1.0, 1.0])

    assert is_identifiable(A, contrast)
    assert is_identifiable(A, compound)
    assert target_noise_amplification(A, contrast) > target_noise_amplification(A, compound)
