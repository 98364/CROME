"""Conjunctive paper-build Gate for the complete CROME CS evidence package."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
import scipy

from crome_identification.evaluation.artifact_audit import (
    audit_public_cs_artifacts,
    audit_revision_cs_artifacts,
    audit_smoke_reproducibility,
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(passed: bool, observed: Any, required: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "observed": observed, "required": required}


_REPOSITORY_MANIFEST = Path("results/repository_manifest.sha256")


def _repository_release_paths(root: Path) -> tuple[Path, ...]:
    """Return the compact, non-self-referential public release surface."""

    patterns = (
        ".gitignore",
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "src/crome_identification/**/*.py",
        "experiments/**/*.py",
        "figures/__init__.py",
        "figures/cs_paper_figures.py",
        "scripts/*.py",
        "tests/**/*.py",
        "configs/*.yaml",
        "data/README.md",
        "data/processed/*",
        "results/summaries/cs02_main.json",
        "results/source_data/cs02_main.csv",
        "results/revision_20260812/summaries/*.json",
        "results/revision_20260812/source_data/*.csv",
        "results/revision_20260812/manifest.sha256",
        "results/figures/cs/manifest.json",
        "results/figures/cs/source_data.csv",
        "results/figures/cs/*.pdf",
        "results/figures/cs/*.png",
    )
    paths = {
        path.resolve()
        for pattern in patterns
        for path in root.glob(pattern)
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.resolve() != (root / _REPOSITORY_MANIFEST).resolve()
        and path.resolve() != (root / "results/summaries/cs_final_gate.json").resolve()
    }
    return tuple(sorted(paths, key=lambda path: path.relative_to(root).as_posix()))


def _repository_manifest_paths(root: Path) -> tuple[Path, ...]:
    return _repository_release_paths(root)


def write_repository_manifest(root: str | Path = ".") -> Path:
    """Write a portable manifest relative to this repository root."""

    root = Path(root).resolve()
    manifest_path = root / _REPOSITORY_MANIFEST
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        f"{_sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in _repository_manifest_paths(root)
    ]
    manifest_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return manifest_path


def _audit_repository_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / _REPOSITORY_MANIFEST
    if not manifest_path.exists():
        return {"passed": False, "reason": "repository release manifest is missing"}
    expected: dict[str, str] = {}
    malformed: list[str] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            malformed.append(line)
            continue
        expected[parts[1]] = parts[0]
    live = {
        path.relative_to(root).as_posix(): path
        for path in _repository_manifest_paths(root)
    }
    rows = []
    for relative in sorted(set(expected) | set(live)):
        path = live.get(relative)
        observed = _sha256(path) if path is not None and path.exists() else None
        rows.append(
            {
                "path": relative,
                "expected": expected.get(relative),
                "observed": observed,
                "passed": observed is not None and observed == expected.get(relative),
            }
        )
    return {
        "passed": not malformed and bool(rows) and all(row["passed"] for row in rows),
        "manifest": _REPOSITORY_MANIFEST.as_posix(),
        "file_count": len(rows),
        "malformed_rows": malformed,
        "files": rows,
    }


def _figure_contract(root: Path, revision_root: Path) -> dict[str, Any]:
    manifest_path = root / "results/figures/cs/manifest.json"
    if not manifest_path.exists():
        return {"passed": False, "reason": "figure manifest is missing"}
    manifest = _load(manifest_path)
    source_hashes = {
        name: _sha256(revision_root / f"summaries/{name}")
        for name in ("cs03_main.json", "cs06_main.json")
    }
    source_ok = manifest.get("source_summaries") == source_hashes
    file_rows = []
    for figure in manifest.get("figures", {}).values():
        for suffix, metadata in figure.items():
            result_path = root / "results/figures/cs" / metadata["name"]
            observed = _sha256(result_path) if result_path.exists() else None
            row = {
                "path": str(result_path.relative_to(root)),
                "expected": metadata["sha256"],
                "observed": observed,
                "passed": observed == metadata["sha256"],
            }
            file_rows.append(row)
    passed = source_ok and bool(file_rows) and all(row["passed"] for row in file_rows)
    return {
        "passed": passed,
        "source_summaries": {"passed": source_ok, "observed": manifest.get("source_summaries"), "required": source_hashes},
        "files": file_rows,
    }


def _single_component_ablation_contract(source_path: str | Path) -> dict[str, Any]:
    source_path = Path(source_path)
    with source_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    paired: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        paired.setdefault((row["rep"], row["cell"]), {})[row["variant"]] = row
    checks = []
    for methods in paired.values():
        support = methods.get("no_support_certifier", {})
        null = methods.get("no_target_null_check", {})
        forced = methods.get("force_point", {})
        checks.append(
            support.get("ablated_component") == "support_certifier"
            and null.get("ablated_component") == "target_null_check"
            and all(
                output.get("typed_contract_enforced", "").lower() == "true"
                and output.get("proof_verification_mandatory", "").lower() == "true"
                for output in (support, null)
            )
            and forced.get("typed_contract_enforced", "").lower() == "false"
        )
    return {
        "passed": bool(checks) and all(checks),
        "observed": f"{sum(checks)}/{len(checks)} paired rows isolate one verified component",
        "required": f"{len(checks)}/{len(checks)}",
    }


def build_final_gate(
    root: str | Path = ".",
    *,
    rerun_smoke: bool = True,
    revision_dir: str | Path = "results/revision_20260812",
) -> dict[str, Any]:
    root = Path(root).resolve()
    revision_root = (root / revision_dir).resolve()
    summaries = {"cs02": _load(root / "results/summaries/cs02_main.json")}
    summaries.update({
        name: _load(revision_root / f"summaries/{name}_main.json")
        for name in ("cs03", "cs04", "cs05", "cs06")
    })
    experiment_passes = {
        "cs02": bool(summaries["cs02"]["main_readiness_gate"]["main_ready"]),
        **{name: bool(summaries[name]["gate"]["passed"]) for name in ("cs03", "cs04", "cs05", "cs06")},
    }
    support_audit = audit_public_cs_artifacts(root, revision_dir)
    revision_audit = audit_revision_cs_artifacts(root, revision_dir)
    ablation_contract = _single_component_ablation_contract(
        revision_root / "source_data/cs06_main.csv"
    )
    figure_contract = _figure_contract(root, revision_root)
    repository_manifest = _audit_repository_manifest(root)
    headline_contract = {
        "utility_records": len(summaries["cs03"].get("utility_records", [])),
        "rq2_point_outputs_matched": bool(
            summaries["cs03"].get("story_metrics", {}).get("rq2", {}).get("point_outputs_matched")
        ),
    }
    headline_passed = (
        headline_contract["utility_records"] > 0
        and headline_contract["rq2_point_outputs_matched"]
    )
    reproducibility = audit_smoke_reproducibility() if rerun_smoke else {
        "passed": False,
        "experiments": {},
        "reason": "smoke rerun was explicitly disabled",
    }
    checks = {
        "all_experiment_gates": _check(all(experiment_passes.values()), experiment_passes, True),
        "artifact_integrity": _check(
            support_audit["passed"] and revision_audit["passed"],
            {"public_release": support_audit["passed"], "revision_evidence": revision_audit["passed"]},
            True,
        ),
        "smoke_reproducibility": _check(reproducibility["passed"], reproducibility["passed"], True),
        "processed_timing_data_hash": _check(
            support_audit["data_hashes"]["passed"], support_audit["data_hashes"]["passed"], True,
        ),
        "headline_evidence_contract": _check(headline_passed, headline_contract, {"utility_records": "> 0", "rq2_point_outputs_matched": True}),
        "single_component_ablation_contract": _check(ablation_contract["passed"], ablation_contract["observed"], ablation_contract["required"]),
        "figure_output_hashes": _check(figure_contract["passed"], figure_contract["passed"], True),
        "repository_release_manifest": _check(
            repository_manifest["passed"],
            {
                "manifest": repository_manifest.get("manifest"),
                "file_count": repository_manifest.get("file_count", 0),
            },
            "portable local repository manifest hash-matches every release file",
        ),
    }
    artifacts = {}
    for name in ("cs02", "cs03", "cs04", "cs05", "cs06"):
        base = root / "results" if name == "cs02" else revision_root
        paths: dict[str, Path] = {
            "summary": base / f"summaries/{name}_main.json",
            "source_data": base / f"source_data/{name}_main.csv",
        }
        artifacts[name] = {
            kind: {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for kind, path in paths.items()
            if path.exists()
        }
    ready = all(item["passed"] for item in checks.values())
    return {
        "PAPER_BUILD_READY": ready,
        "decision": (
            "The code, evidence, and figures pass the conjunctive release Gate."
            if ready else
            "Paper construction remains blocked by at least one failed evidence Gate."
        ),
        "checks": checks,
        "experiment_gate_details": {
            "cs02": summaries["cs02"]["main_readiness_gate"],
            **{name: summaries[name]["gate"] for name in ("cs03", "cs04", "cs05", "cs06")},
        },
        "smoke_reproducibility": reproducibility,
        "artifact_audit": {
            "public_release": support_audit,
            "revision_evidence": revision_audit,
            "figure_contract": figure_contract,
            "repository_manifest": repository_manifest,
        },
        "artifact_ledger": artifacts,
        "runtime_versions": {
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        },
        "claim_boundary": (
            "Evidence supports identifiability-aware target recovery on aligned synthetic geometry and "
            "real Online Retail II event timing with controlled coarsening and injected outcomes; it does "
            "not establish a real-world causal effect or a production systems claim."
        ),
    }


def main() -> int:
    root = Path.cwd()
    result = build_final_gate(root, rerun_smoke=True)
    output = root / "results/summaries/cs_final_gate.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"PAPER_BUILD_READY": result["PAPER_BUILD_READY"], "checks": result["checks"]}, indent=2))
    return 0 if result["PAPER_BUILD_READY"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
