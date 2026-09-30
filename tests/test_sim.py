import numpy as np
import pytest

from qec_certify.codes import code_capacity_circuit, rotated_surface_code
from qec_certify.sim import SyndromeData, sample_syndromes


@pytest.fixture
def circuit():
    return code_capacity_circuit(rotated_surface_code(3), 0.1)


def test_shapes_and_dtypes(circuit):
    data = sample_syndromes(circuit, 1000, seed=0)
    assert data.shots == 1000
    assert data.detection_events.shape == (1000, circuit.num_detectors)
    assert data.observable_flips.shape == (1000, 1)
    assert data.detection_events.dtype == bool
    assert data.observable_flips.dtype == bool


def test_seeded_sampling_is_reproducible(circuit):
    a = sample_syndromes(circuit, 500, seed=7)
    b = sample_syndromes(circuit, 500, seed=7)
    c = sample_syndromes(circuit, 500, seed=8)
    np.testing.assert_array_equal(a.detection_events, b.detection_events)
    np.testing.assert_array_equal(a.observable_flips, b.observable_flips)
    assert not np.array_equal(a.detection_events, c.detection_events)


def test_invalid_inputs(circuit):
    with pytest.raises(ValueError):
        sample_syndromes(circuit, -1, seed=0)
    with pytest.raises(ValueError):
        SyndromeData(np.zeros((3, 2), bool), np.zeros((4, 1), bool))
    with pytest.raises(ValueError):
        SyndromeData(np.zeros(3, bool), np.zeros((3, 1), bool))
