"""Experiment 2: observational equivalence under support gap (Regime A)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from crome_identification.identification.support_gap import (
    assert_observational_equivalence,
    two_point_lower_bound,
)
from crome_identification.observation.lag_support import observed_positive_lags
from crome_identification.responses.equivalence import observational_equivalent_pair

from ._common import load_exp_config, save_json


def run(mode: str = "smoke", outdir: Path | None = None) -> dict[str, Any]:
    cfg = load_exp_config("lag_support")
    Delta = float(cfg.get("Delta", 1.0))
    q_star = float(cfg.get("q_star", Delta))
    J_a = float(cfg.get("J_a", 1.0))

    # Aligned schedule: event at 0, observations at Delta, 2Delta, ...
    event_times = np.array([0.0])
    obs_times = np.arange(Delta, 5.0 * Delta + 1e-12, Delta)
    lags = observed_positive_lags(obs_times, event_times)

    check = assert_observational_equivalence(q_star, lags, J_a=J_a)
    pair = observational_equivalent_pair(q_star, J_a=J_a)

    # sampling on observed lags only
    from crome_identification.responses.equivalence import (
        response_with_jump,
        response_without_jump_matching,
        sample_on_lags,
    )

    ya = sample_on_lags(lambda q: response_with_jump(q, J_a), lags)
    yb = sample_on_lags(lambda q: response_without_jump_matching(q, J_a, q_star), lags)

    # prove that including q=0 breaks the gate test intentionally
    bad_lags = np.concatenate([[0.0], lags])
    zero_included_allclose = False
    try:
        sample_on_lags(lambda q: response_with_jump(q, J_a), bad_lags)
        zero_included_rejected = False
    except ValueError:
        zero_included_rejected = True

    summary = {
        "experiment": "exp02_support_gap",
        "mode": mode,
        "Delta": Delta,
        "q_star": q_star,
        "lags": lags.tolist(),
        "all_lags_positive": bool(np.all(lags > 0)),
        "responses_equal_on_lags": bool(np.allclose(ya, yb)),
        "jump_a": pair.J_a,
        "jump_b": pair.J_b,
        "two_point_lower_bound": two_point_lower_bound(pair.J_a, pair.J_b),
        "gate1_check": check,
        "q0_in_observation_rejected": zero_included_rejected,
        "note": "Deterministic Gate-1 fixture; no Monte Carlo required.",
    }
    if outdir is not None:
        save_json(summary, Path(outdir) / f"exp02_{mode}.json")
    return summary


if __name__ == "__main__":
    print(run("smoke", Path("results/raw")))
