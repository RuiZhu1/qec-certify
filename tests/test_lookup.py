import numpy as np
import pytest

from qec_certify.codes import code_capacity_circuit, rotated_surface_code
from qec_certify.decoders import (
    Decoder,
    ExactMLDecoder,
    LookupTableDecoder,
    MWPMDecoder,
    TableDecoder,
)
from qec_certify.eval import exact_logical_error_rate
from qec_certify.sim import SyndromeData, sample_syndromes
from qec_certify.utils import unpack_bits


def data(dets, obs):
    return SyndromeData(np.array(dets, dtype=bool), np.array(obs, dtype=bool))


class _Constant(Decoder):
    """Always predicts the same flips."""

    def __init__(self, value, num_detectors, num_observables=1):
        self.value = value
        self.num_detectors = num_detectors
        self.num_observables = num_observables

    def predict(self, detection_events):
        dets = self._check_detection_events(detection_events)
        return np.full((len(dets), self.num_observables), self.value, dtype=bool)


def test_majority_vote_ties_and_fallback():
    train = data(
        [[0, 1], [0, 1], [0, 1], [1, 0], [1, 0]],
        [[1], [1], [0], [0], [1]],
    )
    queries = np.array([[0, 1], [1, 0], [1, 1]], dtype=bool)  # majority, tie, unseen
    no_fallback = LookupTableDecoder(2, 1).fit(train)
    np.testing.assert_array_equal(no_fallback.predict(queries)[:, 0], [True, False, False])
    with_fallback = LookupTableDecoder(2, 1, fallback=_Constant(True, 2)).fit(train)
    np.testing.assert_array_equal(with_fallback.predict(queries)[:, 0], [True, True, True])


def test_multiple_observables():
    train = data([[1], [1], [1]], [[1, 0], [1, 0], [0, 1]])
    decoder = LookupTableDecoder(1, 2).fit(train)
    np.testing.assert_array_equal(
        decoder.predict(np.array([[1], [0]], dtype=bool)), [[True, False], [False, False]]
    )


def test_fit_requires_data_and_unfitted_uses_fallback():
    decoder = LookupTableDecoder(2, 1, fallback=_Constant(True, 2))
    with pytest.raises(ValueError, match="requires training data"):
        decoder.fit()
    assert decoder.predict(np.zeros((3, 2), dtype=bool)).all()


def test_converges_to_exact_ml_on_d3():
    circuit = code_capacity_circuit(rotated_surface_code(3), 0.1)
    dem = circuit.detector_error_model()
    optimum = ExactMLDecoder(dem).optimal_logical_error_rate
    fallback = MWPMDecoder.from_circuit(circuit)
    decoder = LookupTableDecoder(dem.num_detectors, 1, fallback)
    decoder.fit(sample_syndromes(circuit, 200_000, seed=4))
    ler = exact_logical_error_rate(decoder, dem)
    assert optimum - 1e-12 <= ler <= optimum * 1.01


def test_table_decoder_freezes_a_decoder():
    circuit = code_capacity_circuit(rotated_surface_code(3), 0.1)
    mwpm = MWPMDecoder.from_circuit(circuit)
    table = TableDecoder.from_decoder(mwpm, chunk_size=7)
    syndromes = unpack_bits(np.arange(2**mwpm.num_detectors), mwpm.num_detectors)
    np.testing.assert_array_equal(table.predict(syndromes), mwpm.predict(syndromes))
    with pytest.raises(ValueError, match="max_detectors"):
        TableDecoder.from_decoder(mwpm, max_detectors=4)
    with pytest.raises(ValueError, match="shape"):
        TableDecoder(np.zeros(3, dtype=np.int64), 2, 1)
