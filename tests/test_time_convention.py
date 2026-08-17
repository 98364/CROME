import numpy as np
import pytest

from crome_identification.observation.lag_support import (
    first_forward_recurrence,
    observed_positive_lags,
)
from crome_identification.processes.marked_events import Trajectory, simulate_trajectory
from crome_identification.processes.simulator import coarsen_event_times, sample_endpoint
from crome_identification.responses.equivalence import response_with_jump
from crome_identification.responses.kernels import SharedKernelParams


def test_lags_strictly_positive():
    lags = observed_positive_lags(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]))
    assert np.all(lags > 0)


def test_forward_recurrence_on_grid_is_delta():
    # event exactly on grid → R = Delta, not 0
    assert first_forward_recurrence(2.0, 1.0) == 1.0


def test_forward_recurrence_decimal_grid_hit_is_delta():
    assert first_forward_recurrence(0.3, 0.1) == pytest.approx(0.1)


def test_coarsening_decimal_grid_hit_stays_in_its_bin():
    out = coarsen_event_times(np.array([0.3, 0.6, 0.7]), 0.1)
    assert out == pytest.approx(np.array([0.3, 0.6, 0.7]))


def test_jump_right_limit():
    J = 1.5
    q = np.array([1e-8, 1e-6, 1e-4])
    r = response_with_jump(q, J)
    assert np.allclose(r, J, atol=1e-3)


def test_endpoint_sampling_does_not_interpolate_across_jump():
    kernel = SharedKernelParams.from_lists([1.0], [0.0], [0.0], [0.0], [1.0, 1.0, 1.0])
    traj = Trajectory(
        times_grid=np.array([0.0, 1.0]),
        X=np.zeros(2),
        V=np.zeros(2),
        U=np.zeros(2),
        B=np.zeros(2),
        Y_latent=np.array([0.0, 1.0]),
        event_times=np.array([0.5]),
        event_marks=np.array([0]),
        intensity_path=np.zeros((1, 1)),
        rho_path=np.ones((1, 1)),
        meta={"kernel": kernel, "process_noise": None},
    )

    _, sampled = sample_endpoint(traj, np.array([0.25, 0.5, 0.75]), 0.0, np.random.default_rng(0))

    assert sampled == pytest.approx(np.array([0.0, 1.0, 1.0]))


def test_simulation_rejects_non_integral_time_grid():
    kernel = SharedKernelParams.from_lists([1.0], [0.0], [0.0], [0.0], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="integer multiple"):
        simulate_trajectory(
            T=1.0,
            delta=0.6,
            rng=np.random.default_rng(0),
            kernel=kernel,
            lambda0=np.array([0.1]),
            alpha_x=np.array([0.0]),
            alpha_v=np.array([0.0]),
        )
