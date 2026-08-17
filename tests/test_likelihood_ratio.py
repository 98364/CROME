import numpy as np

from crome_identification.interventions.likelihood_ratio import likelihood_ratio_path


def test_lr_nonnegative_and_one_when_rho_one():
    times = np.linspace(0, 1, 101)
    C = 2
    n_steps = 100
    intensity = np.ones((n_steps, C)) * 0.1
    rho = np.ones((n_steps, C))
    L = likelihood_ratio_path(
        event_times=np.array([0.2, 0.5]),
        event_marks=np.array([0, 1]),
        times_grid=times,
        intensity_obs=intensity,
        rho_path=rho,
        t0=0.0,
        t1=1.0,
    )
    assert L >= 0
    assert abs(L - 1.0) < 1e-10


def test_lr_positive_under_tilt():
    times = np.linspace(0, 1, 101)
    C = 1
    n_steps = 100
    intensity = np.ones((n_steps, C)) * 0.2
    rho = np.ones((n_steps, C)) * 1.5
    L = likelihood_ratio_path(
        event_times=np.array([0.3]),
        event_marks=np.array([0]),
        times_grid=times,
        intensity_obs=intensity,
        rho_path=rho,
        t0=0.0,
        t1=1.0,
    )
    assert L > 0
