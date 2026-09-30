import numpy as np
import pymatching

from qec_certify.codes import code_capacity_circuit, repetition_code, rotated_surface_code
from qec_certify.decoders import MWPMDecoder
from qec_certify.sim import sample_syndromes


def test_matches_raw_pymatching_on_decomposed_dem():
    circuit = code_capacity_circuit(rotated_surface_code(5), 0.08)
    dem = circuit.detector_error_model(decompose_errors=True)
    data = sample_syndromes(circuit, 5000, seed=11)
    expected = pymatching.Matching.from_detector_error_model(dem).decode_batch(
        data.detection_events
    )
    np.testing.assert_array_equal(
        MWPMDecoder.from_circuit(circuit).predict(data.detection_events), expected.astype(bool)
    )


def test_repetition_code_is_minimum_weight():
    d = 5
    decoder = MWPMDecoder.from_circuit(code_capacity_circuit(repetition_code(d), 0.1))
    rng = np.random.default_rng(3)
    flips = rng.random((2000, d)) < 0.3
    dets = flips[:, :-1] ^ flips[:, 1:]
    # The two bit-flip patterns consistent with a syndrome are c (with c_0 = 0) and its
    # complement. Minimum weight picks the complement, i.e. flips qubit 0 (the logical),
    # iff c has weight above d // 2.
    c = np.cumsum(np.concatenate([np.zeros((len(dets), 1), bool), dets], axis=1), axis=1) % 2
    np.testing.assert_array_equal(decoder.predict(dets)[:, 0], c.sum(axis=1) > d // 2)
