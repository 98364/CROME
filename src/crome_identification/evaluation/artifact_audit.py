"""Integrity audit for CS02 raw, compact, and source-table artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any


def _reject_nonfinite(token: str):
    raise ValueError(f"non-finite JSON token: {token}")


def _load_strict_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _check(passed: bool, observed: Any, required: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "observed": observed, "required": required}


def audit_cs02_artifacts(
    raw_path: str | Path,
    summary_path: str | Path,
    source_path: str | Path,
) -> dict[str, Any]:
    raw_path = Path(raw_path)
    summary_path = Path(summary_path)
    source_path = Path(source_path)
    raw = _load_strict_json(raw_path)
    compact = _load_strict_json(summary_path)
    with source_path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))

    n_reps = int(raw["n_reps"])
    n_regimes = int(raw["n_regimes"])
    methods = tuple(raw["methods"])
    expected_method_rows = n_reps * n_regimes
    expected_source_rows = expected_method_rows * len(methods)
    unique_keys = {
        (row["rep"], row["regime"], row["method"]) for row in source_rows
    }
    method_denominators = {
        method: int(raw["methods"][method]["n_total"]) for method in methods
    }
    source_config_hashes = {row["config_sha256"] for row in source_rows}

    recomputed: dict[str, dict[str, int]] = {}
    for method in methods:
        rows = [row for row in source_rows if row["method"] == method]
        false_points = sum(int(row["false_point"]) for row in rows)
        point_rows = [row for row in rows if row["expected_status"] == "POINT_ESTIMABLE"]
        covered = 0
        for row in point_rows:
            if row["interval_lower"] == "" or row["interval_upper"] == "":
                continue
            covered += int(
                float(row["interval_lower"])
                <= float(row["true_target"])
                <= float(row["interval_upper"])
            )
        recomputed[method] = {
            "n_total": len(rows),
            "false_point_count": false_points,
            "point_coverage_count": covered,
        }

    source_matches_summary = all(
        recomputed[method]["n_total"] == raw["methods"][method]["n_total"]
        and recomputed[method]["false_point_count"]
        == raw["methods"][method]["false_point_count"]
        and recomputed[method]["point_coverage_count"]
        == raw["methods"][method]["point_coverage_count"]
        for method in methods
    )
    gate = raw["main_readiness_gate"]
    gate_recomputed = all(item["passed"] for item in gate["checks"].values())
    checks = {
        "raw_replication_count": _check(
            len(raw["replication_records"]) == n_reps,
            len(raw["replication_records"]),
            n_reps,
        ),
        "compact_omits_replications": _check(
            "replication_records" not in compact,
            "replication_records" in compact,
            False,
        ),
        "method_denominators": _check(
            all(value == expected_method_rows for value in method_denominators.values()),
            method_denominators,
            expected_method_rows,
        ),
        "source_row_count": _check(
            len(source_rows) == expected_source_rows,
            len(source_rows),
            expected_source_rows,
        ),
        "source_unique_keys": _check(
            len(unique_keys) == expected_source_rows,
            len(unique_keys),
            expected_source_rows,
        ),
        "source_config_hash": _check(
            source_config_hashes == {raw["config_sha256"]},
            sorted(source_config_hashes),
            [raw["config_sha256"]],
        ),
        "source_recomputes_summary": _check(
            source_matches_summary,
            recomputed,
            "match raw method counts",
        ),
        "gate_recomputes": _check(
            bool(gate["main_ready"]) == gate_recomputed,
            gate["main_ready"],
            gate_recomputed,
        ),
    }
    return {
        "passed": all(item["passed"] for item in checks.values()),
        "raw_path": str(raw_path),
        "summary_path": str(summary_path),
        "source_path": str(source_path),
        "checks": checks,
    }


_VALID_STATUSES = {
    "POINT_ESTIMABLE", "SET_ESTIMABLE", "NONRECOVERABLE", "INCONCLUSIVE"
}
_SOURCE_KEYS = {
    "cs03": ("rep", "support_mass", "perturbation", "level", "method"),
    "cs04": ("rep", "regime", "method"),
    "cs05": (
        "rep", "n_trajectories", "parameter_dim",
        "events_per_trajectory", "endpoints_per_trajectory",
    ),
    "cs06": ("rep", "cell", "variant"),
}
_METHOD_COUNTS = {"cs03": 6, "cs04": 4, "cs05": 1, "cs06": 6}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(
        config, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_is_finite(rows: list[dict[str, str]]) -> bool:
    forbidden = {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity"}
    return all(
        value.strip().lower() not in forbidden
        for row in rows for value in row.values() if value is not None
    )


def audit_cs_experiment_artifacts(
    experiment: str,
    raw_path: str | Path,
    summary_path: str | Path,
    source_path: str | Path,
) -> dict[str, Any]:
    """Audit an E-CS3--E-CS6 raw/compact/source artifact triple."""

    if experiment not in _SOURCE_KEYS:
        raise ValueError(f"unsupported experiment audit: {experiment}")
    raw_path, summary_path, source_path = map(Path, (raw_path, summary_path, source_path))
    raw = _load_strict_json(raw_path)
    compact = _load_strict_json(summary_path)
    with source_path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    n_reps = int(raw["n_reps"])
    n_cells = int(raw.get("n_cells", raw.get("n_regimes", 0)))
    expected_raw = n_reps * n_cells
    expected_source = expected_raw * _METHOD_COUNTS[experiment]
    key_columns = _SOURCE_KEYS[experiment]
    unique_keys = {tuple(row[column] for column in key_columns) for row in source_rows}
    expected_compact = dict(raw)
    expected_compact.pop("replication_records", None)
    status_values = {
        row.get("predicted_status") for row in source_rows
        if row.get("predicted_status") not in (None, "")
    }
    source_hashes = {row.get("config_sha256") for row in source_rows}
    gate = raw["gate"]
    gate_recomputed = all(item["passed"] for item in gate["checks"].values())
    checks = {
        "raw_row_count": _check(
            len(raw["replication_records"]) == expected_raw,
            len(raw["replication_records"]), expected_raw,
        ),
        "compact_exact_copy": _check(compact == expected_compact, compact == expected_compact, True),
        "source_row_count": _check(len(source_rows) == expected_source, len(source_rows), expected_source),
        "source_unique_keys": _check(len(unique_keys) == expected_source, len(unique_keys), expected_source),
        "valid_statuses": _check(status_values <= _VALID_STATUSES, sorted(status_values), sorted(_VALID_STATUSES)),
        "finite_source_values": _check(_source_is_finite(source_rows), _source_is_finite(source_rows), True),
        "config_hash_recomputes": _check(_config_hash(raw["config"]) == raw["config_sha256"], _config_hash(raw["config"]), raw["config_sha256"]),
        "source_config_hash": _check(source_hashes == {raw["config_sha256"]}, sorted(source_hashes), [raw["config_sha256"]]),
        "gate_recomputes": _check(bool(gate["passed"]) == gate_recomputed, gate["passed"], gate_recomputed),
    }
    return {
        "passed": all(item["passed"] for item in checks.values()),
        "raw_path": str(raw_path), "summary_path": str(summary_path),
        "source_path": str(source_path), "checks": checks,
    }


def audit_compact_cs_experiment_artifacts(
    experiment: str,
    summary_path: str | Path,
    source_path: str | Path,
) -> dict[str, Any]:
    """Audit a release summary and its tidy source-data table without raw runs."""

    if experiment not in {"cs02", *_SOURCE_KEYS}:
        raise ValueError(f"unsupported compact experiment audit: {experiment}")
    summary_path, source_path = map(Path, (summary_path, source_path))
    compact = _load_strict_json(summary_path)
    with source_path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))

    n_reps = int(compact["n_reps"])
    if experiment == "cs02":
        n_cells = int(compact["n_regimes"])
        methods = tuple(compact["methods"])
        key_columns = ("rep", "regime", "method")
        gate = compact["main_readiness_gate"]
        gate_passed = bool(gate["main_ready"])
    else:
        n_cells = int(compact.get("n_cells", compact.get("n_regimes", 0)))
        methods = tuple(compact.get("methods", ()))
        method_count = _METHOD_COUNTS[experiment]
        if not methods:
            methods = tuple(str(index) for index in range(method_count))
        key_columns = _SOURCE_KEYS[experiment]
        gate = compact["gate"]
        gate_passed = bool(gate["passed"])

    expected_source = n_reps * n_cells * len(methods)
    unique_keys = {tuple(row[column] for column in key_columns) for row in source_rows}
    status_values = {
        row.get("predicted_status") for row in source_rows
        if row.get("predicted_status") not in (None, "")
    }
    source_hashes = {row.get("config_sha256") for row in source_rows}
    gate_recomputed = all(item["passed"] for item in gate["checks"].values())
    checks = {
        "compact_omits_replications": _check(
            "replication_records" not in compact,
            "replication_records" in compact,
            False,
        ),
        "source_row_count": _check(
            len(source_rows) == expected_source,
            len(source_rows),
            expected_source,
        ),
        "source_unique_keys": _check(
            len(unique_keys) == expected_source,
            len(unique_keys),
            expected_source,
        ),
        "valid_statuses": _check(
            status_values <= _VALID_STATUSES,
            sorted(status_values),
            sorted(_VALID_STATUSES),
        ),
        "finite_source_values": _check(
            _source_is_finite(source_rows),
            _source_is_finite(source_rows),
            True,
        ),
        "config_hash_recomputes": _check(
            _config_hash(compact["config"]) == compact["config_sha256"],
            _config_hash(compact["config"]),
            compact["config_sha256"],
        ),
        "source_config_hash": _check(
            source_hashes == {compact["config_sha256"]},
            sorted(source_hashes),
            [compact["config_sha256"]],
        ),
        "gate_recomputes": _check(
            gate_passed == gate_recomputed,
            gate_passed,
            gate_recomputed,
        ),
    }
    return {
        "passed": all(item["passed"] for item in checks.values()),
        "summary_path": str(summary_path),
        "source_path": str(source_path),
        "checks": checks,
    }


def _audit_manifest(root: Path, manifest_path: Path) -> dict[str, Any]:
    rows = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        path = root / relative
        observed = _sha256(path) if path.exists() else None
        rows.append({"path": relative, "expected": expected, "observed": observed, "passed": observed == expected})
    return {"passed": bool(rows) and all(row["passed"] for row in rows), "files": rows}


def _portable_repository_paths(value: Any, root: Path) -> Any:
    """Serialize repository-local absolute paths relative to the repository root."""

    if isinstance(value, dict):
        return {key: _portable_repository_paths(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_repository_paths(item, root) for item in value]
    if isinstance(value, str):
        path = Path(value)
        if path.is_absolute():
            try:
                return path.relative_to(root).as_posix()
            except ValueError:
                return value
    return value


def audit_public_cs_artifacts(
    root: str | Path,
    revision_dir: str | Path = "results/revision_20260812",
) -> dict[str, Any]:
    """Audit the compact evidence intentionally shipped in the public repository."""

    root = Path(root).resolve()
    revision_root = (root / revision_dir).resolve()
    experiments = {
        "cs02": audit_compact_cs_experiment_artifacts(
            "cs02",
            root / "results/summaries/cs02_main.json",
            root / "results/source_data/cs02_main.csv",
        ),
        **{
            experiment: audit_compact_cs_experiment_artifacts(
                experiment,
                revision_root / f"summaries/{experiment}_main.json",
                revision_root / f"source_data/{experiment}_main.csv",
            )
            for experiment in ("cs03", "cs04", "cs05", "cs06")
        },
    }
    profile_path = root / "data/processed/online_retail_ii_profile.json"
    profile = _load_strict_json(profile_path)
    processed_path = root / "data/processed/online_retail_ii_events.npz"
    data_checks = {
        "processed_npz_sha256": _check(
            processed_path.exists()
            and _sha256(processed_path) == profile["processed_npz_sha256"],
            _sha256(processed_path) if processed_path.exists() else None,
            profile["processed_npz_sha256"],
        )
    }
    result = {
        "passed": all(item["passed"] for item in experiments.values())
        and all(item["passed"] for item in data_checks.values()),
        "experiments": experiments,
        "data_hashes": {
            "passed": all(item["passed"] for item in data_checks.values()),
            "checks": data_checks,
        },
    }
    return _portable_repository_paths(result, root)


def audit_revision_cs_artifacts(
    root: str | Path,
    revision_dir: str | Path = "results/revision_20260812",
) -> dict[str, Any]:
    """Audit the exact CS03--CS06 artifact tree used by the reported results."""

    root = Path(root).resolve()
    revision_root = (root / revision_dir).resolve()
    experiments = {
        experiment: audit_compact_cs_experiment_artifacts(
            experiment,
            revision_root / f"summaries/{experiment}_main.json",
            revision_root / f"source_data/{experiment}_main.csv",
        )
        for experiment in ("cs03", "cs04", "cs05", "cs06")
    }
    manifest = _audit_manifest(revision_root, revision_root / "manifest.sha256")
    result = {
        "passed": all(item["passed"] for item in experiments.values()) and manifest["passed"],
        "revision_root": str(revision_root),
        "experiments": experiments,
        "manifest": manifest,
    }
    return _portable_repository_paths(result, root)


def _strip_measurement_noise(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_measurement_noise(item)
            for key, item in value.items()
            if "runtime" not in key
            and "peak_python" not in key
            and key != "empirical_slopes"
        }
    if isinstance(value, list):
        return [_strip_measurement_noise(item) for item in value]
    return value


def audit_smoke_reproducibility() -> dict[str, Any]:
    from experiments import (
        cs03_perturbation_grid,
        cs04_online_retail,
        cs05_scaling,
        cs06_ablation,
    )

    runners = {
        "cs03": cs03_perturbation_grid.run,
        "cs04": cs04_online_retail.run,
        "cs05": cs05_scaling.run,
        "cs06": cs06_ablation.run,
    }
    results = {}
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        for name, runner in runners.items():
            first = _strip_measurement_noise(runner("smoke", base / f"{name}_first"))
            second = _strip_measurement_noise(runner("smoke", base / f"{name}_second"))
            first_digest = hashlib.sha256(
                json.dumps(first, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest()
            second_digest = hashlib.sha256(
                json.dumps(second, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest()
            results[name] = {
                "passed": first_digest == second_digest,
                "first_sha256": first_digest, "second_sha256": second_digest,
            }
    return {"passed": all(item["passed"] for item in results.values()), "experiments": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit CS02 experiment artifacts")
    parser.add_argument("raw", type=Path)
    parser.add_argument("summary", type=Path)
    parser.add_argument("source", type=Path)
    args = parser.parse_args(argv)
    result = audit_cs02_artifacts(args.raw, args.summary, args.source)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
