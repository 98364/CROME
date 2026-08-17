from experiments import cs02_split_calibrated as cs02
from crome_identification.evaluation.artifact_audit import audit_cs02_artifacts


def _strip_runtime(value):
    if isinstance(value, dict):
        return {
            key: _strip_runtime(item)
            for key, item in value.items()
            if key not in {"runtime_seconds", "runtime_versions"}
        }
    if isinstance(value, list):
        return [_strip_runtime(item) for item in value]
    return value


def test_cs02_artifact_audit_recomputes_counts_from_raw_and_source(tmp_path):
    cs02.run(mode="smoke", outdir=tmp_path)

    audit = audit_cs02_artifacts(
        tmp_path / "cs02_smoke.json",
        tmp_path / "cs02_smoke_summary.json",
        tmp_path / "cs02_smoke_source.csv",
    )

    assert audit["passed"]
    assert all(check["passed"] for check in audit["checks"].values())


def test_cs02_scientific_content_is_seed_reproducible_ignoring_runtime():
    first = cs02.run(mode="smoke", outdir=None)
    second = cs02.run(mode="smoke", outdir=None)

    assert _strip_runtime(first) == _strip_runtime(second)
