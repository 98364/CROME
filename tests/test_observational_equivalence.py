import numpy as np
import pytest

from crome_identification.identification.support_gap import assert_observational_equivalence
from crome_identification.observation.endpoint import sample_on_lags
from crome_identification.responses.equivalence import (
    response_with_jump,
    response_without_jump_matching,
)


def test_gate1_equivalence_on_positive_lags():
    q_star = 1.0
    lags = np.array([1.0, 2.0, 3.0, 4.0])
    out = assert_observational_equivalence(q_star, lags, J_a=1.0)
    assert out["allclose_on_lags"]
    assert out["jumps_differ"]
    assert out["lower_bound"] == 0.5


def test_q0_forbidden_in_endpoint_sample():
    with pytest.raises(ValueError):
        sample_on_lags(lambda q: response_with_jump(q, 1.0), np.array([0.0, 1.0]))


def test_jumps_differ_but_match_after_qstar():
    q = np.array([1.0, 2.0, 3.0])
    ya = response_with_jump(q, 1.0)
    yb = response_without_jump_matching(q, 1.0, q_star=1.0)
    assert np.allclose(ya, yb)
    assert abs(response_with_jump(1e-12, 1.0) - 0.0) > 0.5
