import numpy as np

from crome_identification.identification.holder_sets import (
    multi_lag_intersection,
    single_lag_set,
)
from crome_identification.responses.equivalence import response_with_jump


def test_true_jump_in_oracle_set():
    J = 1.0
    lags = np.array([0.25, 0.5, 1.0])
    r = response_with_jump(lags, J)
    s = multi_lag_intersection(r, lags, L=1.0, alpha=1.0, eta=1.0)
    assert s.contains(J)
    assert not s.is_empty()


def test_multi_lag_not_wider_than_single():
    J = 1.0
    lags = np.array([0.25, 0.5, 1.0])
    r = response_with_jump(lags, J)
    multi = multi_lag_intersection(r, lags, L=1.0, alpha=1.0)
    single = single_lag_set(float(r[-1]), float(lags[-1]), L=1.0, alpha=1.0)
    assert multi.width <= single.width + 1e-12


def test_empty_intersection_inconsistency():
    # contradictory observations under small L
    lags = np.array([0.5, 1.0])
    r = np.array([10.0, -10.0])
    s = multi_lag_intersection(r, lags, L=0.1, alpha=1.0)
    assert s.is_empty()
