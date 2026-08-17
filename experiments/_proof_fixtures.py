"""Auditable proof payloads used only by controlled simulation fixtures.

These helpers deliberately live under ``experiments``.  They use frozen
simulation design information and therefore must not be imported by a
deployment-facing certificate producer.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np

from crome_identification.certification import ExactNullProof, SupportGapProof


def exact_null_fixture_proof(
    operator: np.ndarray,
    target: np.ndarray,
    witness: np.ndarray,
    *,
    fixture_name: str,
    target_id: str = "scalar_target",
) -> ExactNullProof:
    design = np.asarray(operator, dtype=float)
    target_row = np.asarray(target, dtype=float)
    direction = np.asarray(witness, dtype=float)
    if design.ndim != 2 or target_row.shape != (design.shape[1],):
        raise ValueError("fixture operator and target dimensions do not agree")
    if direction.shape != target_row.shape:
        raise ValueError("fixture witness dimension does not agree with the target")
    if not np.all(design @ direction == 0.0):
        raise ValueError("fixture witness is not exactly null in binary-float arithmetic")
    if float(target_row @ direction) == 0.0:
        raise ValueError("fixture witness does not change the target")
    digest = hashlib.sha256(design.tobytes() + target_row.tobytes()).hexdigest()
    return ExactNullProof(
        operator=tuple(tuple(repr(float(value)) for value in row) for row in design),
        target=tuple(repr(float(value)) for value in target_row),
        witness=tuple(repr(float(value)) for value in direction),
        parameter_space="unrestricted",
        operator_id=f"sha256:{digest}",
        target_id=target_id,
        units="declared scalar-target units",
        provenance=f"oracle-informed controlled fixture: {fixture_name}",
    )


def coordinate_null_fixture_proof(
    operator: np.ndarray,
    target: np.ndarray,
    *,
    fixture_name: str,
    target_id: str = "scalar_target",
) -> ExactNullProof | None:
    """Return an exact coordinate witness when the frozen fixture exposes one."""

    design = np.asarray(operator, dtype=float)
    target_row = np.asarray(target, dtype=float)
    for index in range(design.shape[1]):
        direction = np.zeros(design.shape[1], dtype=float)
        direction[index] = 1.0
        if np.all(design @ direction == 0.0) and float(target_row @ direction) != 0.0:
            return exact_null_fixture_proof(
                design,
                target_row,
                direction,
                fixture_name=fixture_name,
                target_id=target_id,
            )
    return None


def support_gap_fixture_proof(
    observed_positive_lags: np.ndarray,
    *,
    fixture_name: str,
    target_id: str = "scalar_target",
) -> SupportGapProof:
    support = np.asarray(observed_positive_lags, dtype=float)
    support = support[np.isfinite(support) & (support > 0.0)]
    if support.size == 0:
        raise ValueError("support-gap fixture requires at least one positive observed lag")
    minimum = float(np.min(support))
    sampled_support = tuple(repr(float(value)) for value in np.unique(support))
    zero_signature = tuple("0" for _ in sampled_support)
    proof = SupportGapProof(
        observed_support=sampled_support,
        gap=("0", repr(minimum / 2.0)),
        observed_signature_left=zero_signature,
        observed_signature_right=zero_signature,
        target_value_left="0",
        target_value_right="1",
        response_class_premise="fixture response class is closed under the supplied local bump",
        support_specification="frozen oracle schedule used only by the controlled evaluator",
        units="declared scalar-target units",
        provenance=f"oracle-informed controlled fixture: {fixture_name}",
        baseline_level="0",
        bump_amplitude="1",
        bump_power=2,
        schema_version="crome.support-gap.v1",
        producer_version="crome-identification/0.1.0",
        artifact_hash="sha256:" + "0" * 64,
        target_id=target_id,
    )
    return replace(proof, artifact_hash=proof.recompute_artifact_hash())
