import hashlib
import json
from pathlib import Path

from experiments import cs03_perturbation_grid as cs03
from crome_identification.evaluation.artifact_audit import (
    audit_cs_experiment_artifacts,
    audit_public_cs_artifacts,
    audit_smoke_reproducibility,
    audit_revision_cs_artifacts,
)
from experiments.cs_final_gate import (
    _figure_contract,
    _repository_manifest_paths,
    _single_component_ablation_contract,
    build_final_gate,
)


def test_repository_release_manifest_is_self_contained():
    root = Path(__file__).resolve().parents[1]
    paths = _repository_manifest_paths(root)
    relatives = {path.relative_to(root).as_posix() for path in paths}

    assert "README.md" in relatives
    assert "requirements-jss-lock.txt" in relatives
    assert all(not relative.startswith("GitHub/") for relative in relatives)
    assert all(not relative.startswith("results/raw/") for relative in relatives)
    assert all(not relative.startswith("data/raw/") for relative in relatives)
    assert all(not relative.startswith("paper/") for relative in relatives)
    assert all(not relative.startswith("JSS/") for relative in relatives)
    assert all(Path(relative).suffix not in {".svg", ".tiff"} for relative in relatives)


def test_compact_figure_contract_checks_only_distributed_formats(tmp_path):
    revision_root = tmp_path / "results/revision_20260812"
    summary_root = revision_root / "summaries"
    figure_root = tmp_path / "results/figures/cs"
    summary_root.mkdir(parents=True)
    figure_root.mkdir(parents=True)

    source_hashes = {}
    for name in ("cs03_main.json", "cs06_main.json"):
        path = summary_root / name
        path.write_text("{}", encoding="utf-8")
        source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    formats = {}
    for suffix in ("pdf", "png"):
        path = figure_root / f"figure.{suffix}"
        path.write_bytes(suffix.encode("ascii"))
        formats[suffix] = {
            "name": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    for suffix in ("svg", "tiff"):
        formats[suffix] = {
            "name": f"figure.{suffix}",
            "sha256": "0" * 64,
        }
    (figure_root / "manifest.json").write_text(
        json.dumps({"figures": {"rq": formats}, "source_summaries": source_hashes}),
        encoding="utf-8",
    )

    audit = _figure_contract(tmp_path, revision_root)

    assert audit["passed"]
    assert {Path(row["path"]).suffix for row in audit["files"]} == {".pdf", ".png"}


def test_public_source_tables_use_portable_lf_line_endings():
    root = Path(__file__).resolve().parents[1]
    source_tables = (
        root / "results/source_data/cs02_main.csv",
        *sorted((root / "results/revision_20260812/source_data").glob("*.csv")),
        root / "results/figures/cs/source_data.csv",
    )

    assert all(b"\r\n" not in path.read_bytes() for path in source_tables)


def test_public_cs06_table_preserves_the_single_component_ablation_contract():
    root = Path(__file__).resolve().parents[1]
    audit = _single_component_ablation_contract(
        root / "results/revision_20260812/source_data/cs06_main.csv"
    )

    assert audit["passed"]


def test_generic_audit_checks_strict_json_hashes_counts_and_compact_copy(tmp_path):
    cs03.run("smoke", tmp_path)
    audit = audit_cs_experiment_artifacts(
        "cs03",
        tmp_path / "cs03_smoke.json",
        tmp_path / "cs03_smoke_summary.json",
        tmp_path / "cs03_smoke_source.csv",
    )
    assert audit["passed"]
    assert all(check["passed"] for check in audit["checks"].values())


def test_public_artifacts_are_auditable_without_generated_raw_runs():
    audit = audit_public_cs_artifacts(".")
    assert audit["passed"]
    assert all(item["passed"] for item in audit["experiments"].values())
    assert audit["data_hashes"]["passed"]


def test_all_smoke_scientific_outputs_reproduce_ignoring_measurement_noise():
    audit = audit_smoke_reproducibility()
    assert audit["passed"]
    assert all(item["passed"] for item in audit["experiments"].values())


def test_revision_evidence_passes_its_own_manifest_and_audit():
    audit = audit_revision_cs_artifacts(".")
    assert audit["passed"]
    assert audit["manifest"]["passed"]
    assert all(item["passed"] for item in audit["experiments"].values())


def test_final_gate_is_conjunctive_and_authorizes_paper_build():
    gate = build_final_gate(".", rerun_smoke=True)
    assert gate["PAPER_BUILD_READY"]
    assert all(item["passed"] for item in gate["checks"].values())
    assert gate["checks"]["certificate_weight_strategy"]["passed"]
    assert gate["artifact_ledger"]["cs03"]["summary"]["path"].startswith(
        "results/revision_20260812/"
    )
    assert "raw" not in gate["artifact_ledger"]["cs03"]
    assert gate["checks"]["single_component_ablation_contract"]["passed"]
    assert gate["checks"]["repository_release_manifest"]["passed"]
    repository_files = {
        item["path"]
        for item in gate["artifact_audit"]["repository_manifest"]["files"]
    }
    assert "figures/cs_paper_figures.py" in repository_files


def test_final_gate_serializes_only_portable_repository_paths():
    root = Path(__file__).resolve().parents[1]
    gate = build_final_gate(root, rerun_smoke=False)

    def strings(value):
        if isinstance(value, dict):
            for item in value.values():
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)
        elif isinstance(value, str):
            yield value

    absolute_paths = [value for value in strings(gate) if Path(value).is_absolute()]
    assert absolute_paths == []
