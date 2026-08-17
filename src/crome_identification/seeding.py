"""Four-level seed hierarchy: experiment → config → MC rep → bootstrap."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeedBundle:
    """Spawned RNGs for one Monte Carlo repetition."""

    master_seed: int
    experiment: str
    config_id: str
    rep: int
    rng: np.random.Generator
    bootstrap_ss: np.random.SeedSequence

    def bootstrap_rng(self, b: int) -> np.random.Generator:
        if b < 0:
            raise ValueError("bootstrap index must be non-negative")
        child = _indexed_spawn(self.bootstrap_ss, b)
        return np.random.default_rng(child)


def make_seed_bundle(
    master_seed: int,
    experiment: str,
    config_id: str,
    rep: int,
) -> SeedBundle:
    """
    Derive independent generators without seed + worker_id collisions.

    Layout:
      master → experiment hash → config hash → rep → (main_rng, bootstrap_ss)
    """
    root = np.random.SeedSequence(master_seed)
    exp_ss = _named_spawn(root, experiment)
    cfg_ss = _named_spawn(exp_ss, config_id)
    if rep < 0:
        raise ValueError("rep must be non-negative")
    rep_ss = _indexed_spawn(cfg_ss, rep)
    main_ss = _indexed_spawn(rep_ss, 0)
    bootstrap_ss = _indexed_spawn(rep_ss, 1)
    return SeedBundle(
        master_seed=master_seed,
        experiment=experiment,
        config_id=config_id,
        rep=rep,
        rng=np.random.default_rng(main_ss),
        bootstrap_ss=bootstrap_ss,
    )


def _named_spawn(ss: np.random.SeedSequence, name: str) -> np.random.SeedSequence:
    # Stable 32-bit mix from name bytes (not cryptographic).
    h = 2166136261
    for b in name.encode("utf-8"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return _indexed_spawn(ss, h)


def _indexed_spawn(ss: np.random.SeedSequence, index: int) -> np.random.SeedSequence:
    """Derive a child stream without mutating the parent SeedSequence."""
    return np.random.SeedSequence(
        entropy=ss.entropy,
        spawn_key=(*ss.spawn_key, int(index)),
        pool_size=ss.pool_size,
    )
