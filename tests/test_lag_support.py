import numpy as np
import pytest

from crome_identification.observation.lag_support import (
    empirical_support_gap,
    observed_positive_lags,
)


def test_aligned_gap_equals_delta():
    lags = observed_positive_lags(np.array([1.0, 2.0, 3.0]), np.array([0.0]))
    assert empirical_support_gap(lags) == 1.0


def test_async_can_be_smaller_than_delta():
    lags = observed_positive_lags(np.array([1.0, 2.0]), np.array([0.7]))
    assert empirical_support_gap(lags) == pytest.approx(0.3)
