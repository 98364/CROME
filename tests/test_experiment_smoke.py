"""Fast smoke: import and run deterministic / tiny experiments."""

import numpy as np

import pytest

from experiments import (
    exp02_support_gap,
    exp03_async_phase,
    exp04_partial_id,
    exp05_overlap_functionals,
    exp06_regularization_nullspace,
    exp07_perturbations,
)


@pytest.mark.smoke
def test_exp02_smoke():
    out = exp02_support_gap.run(mode="smoke", outdir=None)
    assert out["responses_equal_on_lags"]
    assert out["q0_in_observation_rejected"]


@pytest.mark.smoke
def test_exp06_smoke():
    out = exp06_regularization_nullspace.run(mode="smoke", outdir=None)
    assert out["null_dim"] >= 1
    assert out["regularization_moves_unidentifiable_contrast"]


@pytest.mark.smoke
def test_exp03_suppresses_point_estimates_for_every_gap_design():
    out = exp03_async_phase.run(mode="smoke", outdir=None)

    for design in ["aligned", "discrete", "gap_near_zero"]:
        assert out["designs"][design]["estimate"]["point_estimate"] is None
    assert out["designs"]["continuous_uniform"]["estimate"]["point_estimate"] is not None


def test_exp03_reports_first_lags_for_plot_data():
    out = exp03_async_phase.run(mode="smoke", outdir=None)

    for design in out["designs"].values():
        first_lags = np.asarray(design["first_lags"])
        assert first_lags.shape == (50,)
        assert np.all(first_lags > 0.0)
        assert np.all(first_lags <= out["Delta"])


def test_exp03_main_snapshot_records_requested_mode(monkeypatch):
    cfg = exp03_async_phase.load_exp_config("lag_support")
    cfg.update({"mode": "smoke", "n_traj_main": 20, "n_grid_main": [20], "n_reps_main": 1})
    monkeypatch.setattr(exp03_async_phase, "load_exp_config", lambda _: cfg)

    out = exp03_async_phase.run(mode="main", outdir=None)

    assert out["config"]["mode"] == "main"


def test_exp04_honors_trajectory_count_and_confidence_level(monkeypatch):
    cfg = {
        "L": 1.0,
        "alpha": 1.0,
        "eta": 1.0,
        "level": 0.8,
        "fixed_lags": [0.1, 0.5, 1.0],
        "master_seed": 123,
        "n_traj_smoke": 3,
        "n_reps_smoke": 1,
        "n_bootstrap": 19,
    }
    monkeypatch.setattr(exp04_partial_id, "load_exp_config", lambda _: cfg)

    out = exp04_partial_id.run(mode="smoke", outdir=None)

    assert out["n_traj"] == 3
    assert out["level"] == pytest.approx(0.8)
    assert out["n_boot"] == 19


def test_exp04_reports_interior_and_boundary_coverage(monkeypatch):
    cfg = {
        "L": 1.0,
        "alpha": 1.0,
        "eta": 1.0,
        "level": 0.95,
        "fixed_lags": [0.1, 0.5, 1.0],
        "master_seed": 123,
        "n_traj_smoke": 8,
        "n_reps_smoke": 3,
        "n_bootstrap": 19,
        "mc_confidence_level": 0.95,
        "L_grid": [0.5, 1.0],
        "alpha_grid": [1.0],
    }
    monkeypatch.setattr(exp04_partial_id, "load_exp_config", lambda _: cfg)

    out = exp04_partial_id.run(mode="smoke", outdir=None)

    assert set(out["scenarios"]) == {"interior", "least_favorable_boundary"}
    boundary = out["scenarios"]["least_favorable_boundary"]
    assert boundary["oracle_set"] == pytest.approx(
        {"lower": 1.0, "upper": 1.0, "width": 0.0}
    )
    for scenario in out["scenarios"].values():
        assert 0.0 <= scenario["point_coverage"] <= 1.0
        assert 0.0 <= scenario["oracle_set_coverage"] <= 1.0
        assert scenario["point_coverage_mc_se"] >= 0.0
        assert scenario["oracle_set_coverage_mc_se"] >= 0.0
        assert 0 <= scenario["point_coverage_successes"] <= out["n_reps"]
        assert 0 <= scenario["oracle_set_coverage_successes"] <= out["n_reps"]
        assert scenario["point_coverage_ci"]["method"] == "wilson"
        assert scenario["oracle_set_coverage_ci"]["method"] == "wilson"
        assert scenario["mean_cr_width"] >= 0.0
    assert out["coverage"] == out["scenarios"]["interior"]["point_coverage"]

    sensitivity = out["sensitivity"]
    assert sensitivity["dgp"] == "interior"
    assert [(item["L"], item["alpha"]) for item in sensitivity["grid"]] == [
        (0.5, 1.0),
        (1.0, 1.0),
    ]
    assert not sensitivity["grid"][0]["oracle_contains_J"]
    assert sensitivity["grid"][1]["oracle_set"] == pytest.approx(
        out["scenarios"]["interior"]["oracle_set"]
    )


@pytest.mark.smoke
def test_exp05_persistent_response_recovery_matches_oracle_noise_floor():
    out = exp05_overlap_functionals.run(mode="smoke", outdir=None)

    assert out["full_parameter_error"] < 0.2
    assert abs(out["compound_jump_error"]) < 0.1
    assert out["pred_mse"] < 0.05


def test_exp05_reports_spectrum_and_coefficients_for_plot_data():
    out = exp05_overlap_functionals.run(mode="smoke", outdir=None)

    assert len(out["singular_values"]) == int(out["diagnostics"]["n_col"])
    assert len(out["gram_eigenvalues"]) == int(out["diagnostics"]["n_col"])
    assert len(out["theta_true"]) == len(out["theta_hat"]) == len(out["parameter_labels"])
    assert np.all(np.diff(out["singular_values"]) <= 0.0)


def test_exp05_main_snapshot_records_requested_mode(monkeypatch):
    cfg = exp05_overlap_functionals.load_exp_config("overlap")
    cfg.update({"mode": "smoke", "n_traj_main": 5, "n_reps_main": 1})
    monkeypatch.setattr(exp05_overlap_functionals, "load_exp_config", lambda _: cfg)

    out = exp05_overlap_functionals.run(mode="main", outdir=None)

    assert out["config"]["mode"] == "main"


def test_exp06_reports_null_and_ridge_paths_for_plot_data():
    out = exp06_regularization_nullspace.run(mode="smoke", outdir=None)

    null_path = out["null_shift_path"]
    assert len(null_path["shift"]) == len(null_path["bad_contrast"])
    assert len(null_path["shift"]) == len(null_path["good_contrast"])
    assert len(null_path["shift"]) == len(null_path["pred_mse"])
    assert np.ptp(null_path["bad_contrast"]) > 1.0
    assert np.ptp(null_path["good_contrast"]) < 1e-8
    assert np.ptp(null_path["pred_mse"]) < 1e-10

    ridge_path = out["ridge_path"]
    assert len(ridge_path["lambda"]) == len(ridge_path["good_contrast"])
    assert len(ridge_path["lambda"]) == len(ridge_path["pred_mse"])


@pytest.mark.smoke
def test_exp07_exact_and_zero_jitter_share_the_same_observations():
    out = exp07_perturbations.run(mode="smoke", outdir=None)

    for baseline in ["oracle", "correct_mean", "omit_V", "wrong_trend"]:
        exact = out["grid"][f"{baseline}|exact"]
        zero_jitter = out["grid"][f"{baseline}|jitter_0.0"]
        assert zero_jitter["pred_mse"] == pytest.approx(exact["pred_mse"])
        assert "theta_error" in exact
        assert "compound_jump_error" in exact

    oracle = out["grid"]["oracle|exact"]
    assert oracle["theta_error"] < 0.2
    assert abs(oracle["compound_jump_error"]) < 0.1
    assert oracle["pred_mse"] < 0.05


def test_exp07_main_snapshot_records_requested_mode(monkeypatch):
    cfg = exp07_perturbations.load_exp_config("perturbation")
    cfg.update({"mode": "smoke", "n_traj_main": 5, "n_reps_main": 1})
    monkeypatch.setattr(exp07_perturbations, "load_exp_config", lambda _: cfg)

    out = exp07_perturbations.run(mode="main", outdir=None)

    assert out["config"]["mode"] == "main"
