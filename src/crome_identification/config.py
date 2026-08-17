"""YAML config loading and default experiment fixtures."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG_DIR = PACKAGE_ROOT / "configs"
INSTALLED_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
PREFIX_CONFIG_DIR = Path(sys.prefix) / "configs"


def _default_config_dir() -> Path:
    for candidate in (SOURCE_CONFIG_DIR, INSTALLED_CONFIG_DIR, PREFIX_CONFIG_DIR):
        if candidate.exists():
            return candidate
    return SOURCE_CONFIG_DIR


DEFAULT_CONFIG_DIR = _default_config_dir()


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping: {path}")
    return data


def resolve_config(name_or_path: str, config_dir: Path | None = None) -> dict[str, Any]:
    """Load by absolute path, relative path, or stem under configs/."""
    p = Path(name_or_path)
    if p.exists():
        return load_yaml(p)
    base = config_dir if config_dir is not None else _default_config_dir()
    candidate = base / name_or_path
    if candidate.exists():
        return load_yaml(candidate)
    if not name_or_path.endswith(".yaml"):
        candidate = base / f"{name_or_path}.yaml"
        if candidate.exists():
            return load_yaml(candidate)
    raise FileNotFoundError(f"Config not found: {name_or_path}")


def default_dgp_params() -> dict[str, Any]:
    """Default synthetic DGP from the execution plan (section 9.8)."""
    return {
        "n": 500,
        "T": 20.0,
        "delta": 0.01,
        "t0": 5.0,
        "Q": 5.0,
        # Main kernels contain permanent components, so the default inverse
        # response window covers the complete default observation horizon.
        "Qresp": 20.0,
        "Delta": 1.0,
        "C": 3,
        "kappa_X": 0.5,
        "mu_X": 0.0,
        "sigma_X": 0.4,
        "kappa_V": 0.8,
        "mu_V": 0.0,
        "sigma_V": 0.25,
        "beta_X": 0.5,
        "beta_V": 0.3,
        "kappa_U": 1.0,
        "sigma_U": 0.2,
        "process_noise_sigma": 0.0,
        "sigma_meas": 0.1,
        "lambda0": [0.10, 0.08, 0.06],
        "alpha_x": [0.35, -0.25, 0.20],
        "alpha_v": [0.20, 0.30, -0.20],
        "alpha_y": [0.0, 0.0, 0.0],
        "alpha_n": [0.0, 0.0, 0.0],
        "rho_window": [5.0, 10.0],
        "rho_type": 1,
        "rho": 1.5,
        "J": [1.0, 0.5, -0.7],
        "a1": [0.6, 0.3, -0.4],
        "a2": [0.2, 0.1, -0.1],
        "a3": [0.0, 0.0, 0.0],
        "beta_kernel": [1.0, 0.3, 0.8],
        "L_holder": 1.0,
        "alpha_holder": 1.0,
        "eta": 1.0,
        "master_seed": 20260806,
        "n_bootstrap": 999,
        "n_reps_smoke": 20,
        "n_reps_dev": 100,
        "n_reps_main": 500,
        "svd_tol": 1.0e-10,
    }
