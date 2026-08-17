import numpy as np

from crome_identification.benchmarks.aligned_geometry import (
    PerturbationSpec,
    generate_aligned_geometry,
)


def test_zero_target_support_creates_target_null_direction():
    data = generate_aligned_geometry(
        n_units=400,
        parameter_dim=4,
        target_support_mass=0.0,
        seed=12,
    )
    assert np.all(data.true_design[:, 0] == 0.0)
    assert np.linalg.matrix_rank(data.true_design) == 3
    assert np.allclose(data.true_design @ np.array([1.0, 0.0, 0.0, 0.0]), 0.0)
    assert data.truth_status == "NONRECOVERABLE"


def test_positive_target_support_supplies_target_information():
    data = generate_aligned_geometry(
        n_units=400,
        parameter_dim=4,
        target_support_mass=0.4,
        seed=15,
    )
    assert np.count_nonzero(data.true_design[:, 0]) > 0
    assert np.linalg.matrix_rank(data.true_design) == 4
    assert data.truth_status == "POINT_ESTIMABLE"
    assert np.isclose(data.true_target, data.target @ data.theta)


def test_perturbation_changes_observed_not_latent_operator():
    clean = generate_aligned_geometry(
        n_units=300,
        parameter_dim=3,
        target_support_mass=0.3,
        seed=19,
    )
    perturbed = generate_aligned_geometry(
        n_units=300,
        parameter_dim=3,
        target_support_mass=0.3,
        perturbation=PerturbationSpec(kind="basis_error", level=0.2),
        seed=19,
    )
    assert np.array_equal(clean.true_design, perturbed.true_design)
    assert np.array_equal(clean.lags, perturbed.lags)
    assert not np.array_equal(perturbed.true_design, perturbed.observed_design)


def test_seed_reproduces_all_arrays():
    kwargs = dict(
        n_units=250,
        parameter_dim=5,
        target_support_mass=0.2,
        perturbation=PerturbationSpec(kind="mark_noise", level=0.1),
        seed=23,
    )
    first = generate_aligned_geometry(**kwargs)
    second = generate_aligned_geometry(**kwargs)
    for field in (
        "lags",
        "marks",
        "true_design",
        "observed_design",
        "outcomes",
        "trace_responses",
    ):
        assert np.array_equal(getattr(first, field), getattr(second, field))
