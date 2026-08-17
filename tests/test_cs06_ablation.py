from experiments.cs06_ablation import CELLS, VARIANTS, run


def test_ablation_contract_contains_all_required_variants_and_hard_cells():
    assert set(VARIANTS) == {
        "full_crome",
        "no_support_certifier",
        "no_target_null_check",
        "force_point",
        "no_mark_sharing",
        "fixed_tolerance",
    }
    assert {"unsupported_corrupted", "structural_null", "shared_sparse", "scale_small"} <= set(CELLS)


def test_cs06_smoke_is_paired_and_exposes_expected_failure_modes(tmp_path):
    result = run("smoke", tmp_path)
    assert result["artifact_audit"]["paired_complete"]
    summaries = result["variants"]
    assert summaries["full_crome"]["false_point_count"] == 0
    assert summaries["no_support_certifier"]["false_point_count"] == 0
    assert summaries["no_target_null_check"]["false_point_count"] == 0
    assert summaries["force_point"]["false_point_count"] > 0
    assert (
        summaries["no_support_certifier"]["point_oracle_yield"]["count"]
        < summaries["full_crome"]["point_oracle_yield"]["count"]
    )
    assert (
        summaries["no_target_null_check"]["structural_null_recovery"]["count"]
        < summaries["full_crome"]["structural_null_recovery"]["count"]
    )
    assert result["diagnostics"]["fixed_tolerance_scale_failures"] > 0
    assert "interval" in summaries["full_crome"]["false_point"]
    assert "interval" in summaries["no_mark_sharing"]["shared_sparse_rmse"]
    full_outputs = [
        row["methods"]["full_crome"] for row in result["replication_records"]
    ]
    assert all("failure_ledger" in output for output in full_outputs)
    assert all(output.get("typed_contract_enforced") for output in full_outputs)
    for variant in ("no_support_certifier", "no_target_null_check"):
        outputs = [row["methods"][variant] for row in result["replication_records"]]
        assert all(output.get("typed_contract_enforced") for output in outputs)
        assert all(output.get("proof_verification_mandatory") for output in outputs)
    assert result["gate"]["checks"]["support_ablation_loses_point_yield"]["passed"]
    assert result["gate"]["checks"]["null_ablation_loses_structural_recovery"]["passed"]
