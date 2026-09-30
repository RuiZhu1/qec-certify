"""Behaviour shared by every decoder implementing the common interface."""

import numpy as np
import pytest
import stim

from qec_certify.codes import code_capacity_circuit, repetition_code, rotated_surface_code
from qec_certify.decoders import Decoder, ExactMLDecoder, MWPMDecoder

DECODER_CLASSES = [MWPMDecoder, ExactMLDecoder]


def single_qubit_error_cases(code, basis):
    """(detection events, observable flip) of every weight-1 Pauli error."""
    n = code.num_data_qubits
    stabs = [s.pauli_string(n) for s in code.stabilizers]
    logical = code.logical_pauli_string(basis)
    dets, obs = [], []
    for q in range(n):
        for pauli in "XYZ":
            error = stim.PauliString(n)
            error[q] = pauli
            dets.append([not error.commutes(s) for s in stabs])
            obs.append([not error.commutes(logical)])
    return np.array(dets, dtype=bool), np.array(obs, dtype=bool)


@pytest.mark.parametrize("decoder_cls", DECODER_CLASSES)
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_corrects_every_single_qubit_error_d3_surface(decoder_cls, basis):
    code = rotated_surface_code(3)
    decoder = decoder_cls.from_circuit(code_capacity_circuit(code, 0.05, basis))
    dets, obs = single_qubit_error_cases(code, basis)
    np.testing.assert_array_equal(decoder.predict(dets), obs)


@pytest.mark.parametrize("decoder_cls", DECODER_CLASSES)
def test_corrects_every_single_qubit_error_repetition(decoder_cls):
    code = repetition_code(5)
    decoder = decoder_cls.from_circuit(code_capacity_circuit(code, 0.05))
    dets, obs = single_qubit_error_cases(code, "Z")
    np.testing.assert_array_equal(decoder.predict(dets), obs)


@pytest.mark.parametrize("decoder_cls", DECODER_CLASSES)
def test_interface(decoder_cls):
    circuit = code_capacity_circuit(rotated_surface_code(3), 0.05)
    decoder = decoder_cls.from_circuit(circuit)
    assert isinstance(decoder, Decoder)
    assert decoder.fit() is decoder
    assert decoder.num_detectors == circuit.num_detectors
    assert decoder.num_observables == 1

    predictions = decoder.predict(np.zeros((4, circuit.num_detectors), dtype=np.uint8))
    assert predictions.shape == (4, 1)
    assert predictions.dtype == bool
    assert not predictions.any()

    assert decoder.predict(np.zeros((0, circuit.num_detectors), dtype=bool)).shape == (0, 1)
    with pytest.raises(ValueError):
        decoder.predict(np.zeros((4, circuit.num_detectors + 1), dtype=bool))
    with pytest.raises(ValueError):
        decoder.predict(np.zeros(circuit.num_detectors, dtype=bool))
