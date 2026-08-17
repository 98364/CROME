import numpy as np

from crome_identification.seeding import make_seed_bundle


def _bootstrap_draws(order):
    bundle = make_seed_bundle(7, "experiment", "config", 0)
    return {
        b: bundle.bootstrap_rng(b).integers(0, 2**32, size=4, dtype=np.uint32)
        for b in order
    }


def test_bootstrap_streams_are_indexed_independently_of_request_order():
    forward = _bootstrap_draws([0, 1])
    reverse = _bootstrap_draws([1, 0])

    assert np.array_equal(forward[0], reverse[0])
    assert np.array_equal(forward[1], reverse[1])


def test_repeated_bootstrap_index_returns_the_same_stream():
    bundle = make_seed_bundle(7, "experiment", "config", 0)

    first = bundle.bootstrap_rng(3).integers(0, 2**32, size=4, dtype=np.uint32)
    second = bundle.bootstrap_rng(3).integers(0, 2**32, size=4, dtype=np.uint32)

    assert np.array_equal(first, second)
