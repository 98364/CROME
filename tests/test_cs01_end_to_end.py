import json

import pytest

from experiments import cs01_end_to_end_certification as cs01


def test_cs01_covers_all_four_oracle_states_without_point_leakage():
    out = cs01.run(mode="smoke", outdir=None)

    assert set(out["oracle_statuses"]) == {
        "POINT_ESTIMABLE",
        "SET_ESTIMABLE",
        "NONRECOVERABLE",
        "INCONCLUSIVE",
    }
    assert out["crome"]["status_accuracy"] == pytest.approx(1.0)
    assert out["crome"]["point_leakage_count"] == 0


def test_cs01_reduces_false_point_certification_against_naive():
    out = cs01.run(mode="smoke", outdir=None)

    assert out["crome"]["false_point_rate"] < out["naive"]["false_point_rate"]


def test_cs01_writes_raw_and_compact_outputs(tmp_path):
    out = cs01.run(mode="smoke", outdir=tmp_path)

    raw_path = tmp_path / "cs01_smoke.json"
    summary_path = tmp_path / "cs01_smoke_summary.json"
    assert raw_path.exists()
    assert summary_path.exists()
    assert len(json.loads(raw_path.read_text())["replication_records"]) == out["n_reps"]
    assert "replication_records" not in json.loads(summary_path.read_text())
