import numpy as np
import pytest
import stim

from qec_certify.codes import circuit_level_circuit, code_capacity_circuit, rotated_surface_code
from qec_certify.decoders import BPOSDDecoder, UnionFindDecoder
from qec_certify.eval import evaluate_decoder
from qec_certify.sim import sample_syndromes


@pytest.mark.parametrize("decoder_class", [BPOSDDecoder, UnionFindDecoder])
def test_dem_decoders_recover_single_logical_mechanism(decoder_class):
    dem = stim.DetectorErrorModel("error(0.1) D0 D1 L0")
    decoder = decoder_class(dem)
    predictions = decoder.predict(np.array([[0, 0], [1, 1]], dtype=bool))
    assert predictions.shape == (2, 1)
    assert predictions.dtype == bool
    assert not predictions[0, 0]
    assert predictions[1, 0]


@pytest.mark.parametrize("decoder_class", [BPOSDDecoder, UnionFindDecoder])
def test_dem_decoders_handle_no_visible_mechanisms(decoder_class):
    dem = stim.DetectorErrorModel("error(1) L0")
    decoder = decoder_class(dem)
    assert decoder.num_mechanisms == 0
    assert decoder.predict(np.zeros((3, 0), dtype=bool)).tolist() == [[True]] * 3


@pytest.mark.parametrize("decoder_class", [BPOSDDecoder, UnionFindDecoder])
def test_dem_decoders_on_circuit_level_samples(decoder_class):
    circuit = circuit_level_circuit(rotated_surface_code(3), 0.01, rounds=2)
    data = sample_syndromes(circuit, 10, 7)
    result = evaluate_decoder(decoder_class.from_circuit(circuit), data)
    assert result.shots == 10
    assert 0 <= result.num_errors <= 10


def test_dem_decoders_validate_detector_count():
    decoder = BPOSDDecoder(stim.DetectorErrorModel("error(0.1) D0"))
    with pytest.raises(ValueError, match="expected detection events"):
        decoder.predict(np.zeros((1, 2), dtype=bool))


def test_code_capacity_dem_decoders_are_reproducible():
    circuit = code_capacity_circuit(rotated_surface_code(3), 0.05)
    data = sample_syndromes(circuit, 20, 11)
    for decoder_class in (BPOSDDecoder, UnionFindDecoder):
        decoder = decoder_class.from_circuit(circuit)
        first = decoder.predict(data.detection_events)
        second = decoder.predict(data.detection_events)
        np.testing.assert_array_equal(first, second)
