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


def test_correlated_matches_raw_pymatching():
    circuit = code_capacity_circuit(rotated_surface_code(5), 0.08)
    dem = circuit.detector_error_model(decompose_errors=True)
    data = sample_syndromes(circuit, 5000, seed=11)
    expected = pymatching.Matching.from_detector_error_model(
        dem, enable_correlations=True
    ).decode_batch(data.detection_events, enable_correlations=True)
    decoder = MWPMDecoder.from_circuit(circuit, correlated=True)
    assert decoder.correlated
    np.testing.assert_array_equal(decoder.predict(data.detection_events), expected.astype(bool))


def test_correlated_matching_beats_plain_matching_on_d5():
    """Using the Y-error correlation helps MWPM at d=5: paired comparison on the same shots."""
    circuit = code_capacity_circuit(rotated_surface_code(5), 0.1)
    data = sample_syndromes(circuit, 50_000, seed=12)

    def wrong(decoder):
        return np.any(decoder.predict(data.detection_events) != data.observable_flips, axis=1)

    plain = wrong(MWPMDecoder.from_circuit(circuit))
    correlated = wrong(MWPMDecoder.from_circuit(circuit, correlated=True))
    discordant = int(np.sum(plain != correlated))
    assert plain.sum() - correlated.sum() > 3 * np.sqrt(discordant)
