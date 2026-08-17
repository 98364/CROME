"""Acceptance contracts for the frozen-priority experiments.

These tests are intentionally small; main-mode Monte Carlo runs are checked
separately before publication.
"""

from crome_identification.inverse.functionals import is_identifiable
from experiments import (
    exp03_async_phase,
    exp05b_mark_information,
)


def test_exp03_reports_sample_size_scan(monkeypatch):
    cfg = {
        "Delta": 1.0,
        "J_a": 1.0,
        "master_seed": 123,
        "phase_designs": ["aligned", "discrete", "continuous_uniform", "gap_near_zero"],
        "bandwidth_grid": [0.2, 0.4],
        "n_grid_smoke": [20, 40],
        "n_reps_smoke": 2,
        "near_zero_threshold": 0.2,
    }
    monkeypatch.setattr(exp03_async_phase, "load_exp_config", lambda _: cfg)

    out = exp03_async_phase.run(mode="smoke", outdir=None)

    assert out["sample_sizes"] == [20, 40]
    assert out["n_reps"] == 2
    assert set(out["convergence"]) == {"continuous_uniform", "gap_near_zero"}
    for design in out["convergence"].values():
        for item in design:
            assert "q_min" in item and "median" in item["q_min"]
            assert "near_zero_count" in item
            assert "failure_frequency" in item


def test_exp05b_reports_all_mark_controls(monkeypatch):
    cfg = exp05b_mark_information.default_config()
    cfg.update({"n_traj_smoke": 8, "n_reps_smoke": 3, "scenarios_smoke": ["simultaneous"]})
    monkeypatch.setattr(exp05b_mark_information, "load_exp_config", lambda _: cfg)

    out = exp05b_mark_information.run(mode="smoke", outdir=None)
    scenario = out["scenarios"]["simultaneous"]

    assert set(scenario["variants"]) == {
        "collapsed_mark",
        "observed_shared_mark",
        "shuffled_mark",
        "free_trajectory",
    }
    for variant in scenario["variants"].values():
        assert "mark_target" in variant["targets"]
        assert "compound_target" in variant["targets"]
        assert "identification_residual" in variant["targets"]["mark_target"]
    assert "incremental_rank" in scenario["mark_information"]


def test_exp05b_strong_excitation_identifies_the_shared_mark_target(monkeypatch):
    cfg = exp05b_mark_information.default_config()
    cfg.update({"n_traj_smoke": 30, "scenarios_smoke": ["strong_excitation"]})
    monkeypatch.setattr(exp05b_mark_information, "load_exp_config", lambda _: cfg)

    resolved = exp05b_mark_information._config("smoke")
    variants, _ = exp05b_mark_information._build_variants(
        resolved, "strong_excitation"
    )
    observed = variants["observed_shared_mark"]
    design, scales = exp05b_mark_information._normalize(observed["A_raw"])
    mark_target = observed["targets"]["mark_target"] * scales

    assert is_identifiable(design, mark_target, tol=float(resolved["svd_tol"]))
    assert exp05b_mark_information._partial_information(
        observed["A_raw"], float(resolved["svd_tol"])
    )["incremental_rank"] == 3


def test_exp05b_simultaneous_events_add_no_mark_information(monkeypatch):
    cfg = exp05b_mark_information.default_config()
    cfg.update({"n_traj_smoke": 30, "scenarios_smoke": ["simultaneous"]})
    monkeypatch.setattr(exp05b_mark_information, "load_exp_config", lambda _: cfg)

    resolved = exp05b_mark_information._config("smoke")
    variants, _ = exp05b_mark_information._build_variants(resolved, "simultaneous")
    observed = variants["observed_shared_mark"]

    assert exp05b_mark_information._partial_information(
        observed["A_raw"], float(resolved["svd_tol"])
    )["incremental_rank"] == 0


def test_exp05b_config_snapshot_records_requested_mode(monkeypatch):
    cfg = exp05b_mark_information.default_config()
    cfg["mode"] = "smoke"
    monkeypatch.setattr(exp05b_mark_information, "load_exp_config", lambda _: cfg)

    assert exp05b_mark_information._config("main")["mode"] == "main"
