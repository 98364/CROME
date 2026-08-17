"""Shared helpers for experiment scripts."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

from crome_identification.config import default_dgp_params, load_yaml, resolve_config
from crome_identification.responses.kernels import SharedKernelParams


def mode_reps(cfg: dict[str, Any], mode: str, key_prefix: str = "n_reps") -> int:
    table = {
        "smoke": cfg.get(f"{key_prefix}_smoke", 20),
        "dev": cfg.get(f"{key_prefix}_dev", 100),
        "main": cfg.get(f"{key_prefix}_main", 500),
    }
    return int(table[mode])


def merge_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    base = default_dgp_params()
    base.update(cfg)
    return base


def kernel_from_cfg(cfg: dict[str, Any]) -> SharedKernelParams:
    return SharedKernelParams.from_lists(
        cfg["J"],
        cfg["a1"],
        cfg["a2"],
        cfg["a3"],
        cfg["beta_kernel"],
    )


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, default=_json_default, allow_nan=False),
        encoding="utf-8",
    )


def save_raw_and_summary(obj: dict[str, Any], raw_path: Path) -> tuple[Path, Path]:
    """Save full replication output and a compact manuscript-facing summary.

    Under the standard ``results/raw`` layout, the compact file is written to
    ``results/summaries`` with the same name.  For a custom output directory it
    is written beside the raw file with a ``_summary`` suffix.
    """
    raw_path = Path(raw_path)
    obj.setdefault(
        "runtime_versions",
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    )
    save_json(obj, raw_path)
    compact = dict(obj)
    compact.pop("replication_records", None)
    if raw_path.parent.name == "raw":
        summary_path = raw_path.parent.parent / "summaries" / raw_path.name
    else:
        summary_path = raw_path.with_name(f"{raw_path.stem}_summary{raw_path.suffix}")
    save_json(compact, summary_path)
    return raw_path, summary_path


def _json_default(o: Any):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def load_exp_config(stem: str) -> dict[str, Any]:
    try:
        return merge_defaults(resolve_config(stem))
    except FileNotFoundError:
        return merge_defaults({})
