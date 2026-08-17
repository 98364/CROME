import numpy as np

from crome_identification.processes.simulator import jitter_event_times


class _SequenceRng:
    def __init__(self, values):
        self._values = iter(values)

    def uniform(self, low, high, size=None):
        return next(self._values)


def test_jitter_preserves_event_identity_order_when_events_cross():
    event_times = np.array([0.4, 0.5])
    rng = _SequenceRng([0.15, -0.15])

    jittered = jitter_event_times(event_times, 0.2, rng, T=1.0)

    # The first output still belongs to the first event; record sorting must
    # reorder timestamps and marks together at the caller.
    assert np.allclose(jittered, np.array([0.55, 0.35]))
