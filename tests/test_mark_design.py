import numpy as np
import pytest

from crome_identification.inverse.design_matrix import build_design_matrix
from crome_identification.inverse.diagnostics import matrix_diagnostics


def test_parameter_dim_independent_of_event_count():
    obs = [np.array([1.0, 2.0, 3.0])]
    # many events
    taus = [np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])]
    marks = [np.array([0, 1, 2, 0, 1, 2])]
    A, _, _ = build_design_matrix(obs, taus, marks, C=3, L_basis=2, Qresp=5.0, normalize=False)
    assert A.shape[1] == 3 * 3  # C * (L+1)


def test_mark_changes_columns():
    obs = [np.array([2.0])]
    A0, _, _ = build_design_matrix(
        obs, [np.array([1.0])], [np.array([0])], C=2, L_basis=1, normalize=False
    )
    A1, _, _ = build_design_matrix(
        obs, [np.array([1.0])], [np.array([1])], C=2, L_basis=1, normalize=False
    )
    assert not np.allclose(A0, A1)


def test_wide_matrix_not_full_rank():
    obs = [np.array([1.0, 2.0])]
    A, _, _ = build_design_matrix(
        obs,
        [np.array([0.5])],
        [np.array([0])],
        C=5,
        L_basis=3,
        normalize=False,
    )
    d = matrix_diagnostics(A)
    if A.shape[1] > A.shape[0]:
        assert d["full_column_rank"] == 0.0


def test_normalized_design_returns_scales_for_physical_parameters():
    obs = [np.array([1.0, 2.0, 3.0])]
    taus = [np.array([0.2, 1.2])]
    marks = [np.array([0, 1])]
    raw, _, _ = build_design_matrix(obs, taus, marks, C=2, L_basis=0, normalize=False)
    normalized, _, _, scales = build_design_matrix(
        obs,
        taus,
        marks,
        C=2,
        L_basis=0,
        normalize=True,
        return_column_scales=True,
    )
    theta_true = np.array([2.0, 5.0])
    z = raw @ theta_true

    theta_normalized = np.linalg.lstsq(normalized, z, rcond=None)[0]
    theta_physical = scales * theta_normalized

    assert theta_physical == pytest.approx(theta_true)


def test_strictly_positive_near_zero_lag_is_not_dropped():
    A, _, index_map = build_design_matrix(
        [np.array([1.0])],
        [np.array([0.999999])],
        [np.array([0])],
        C=1,
        L_basis=0,
        normalize=False,
    )

    assert A.shape == (1, 1)
    assert index_map == [(0, 0)]
