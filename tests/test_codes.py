import itertools

import numpy as np
import pytest

from qec_certify.codes import (
    Stabilizer,
    code_capacity_circuit,
    make_code,
    repetition_code,
    rotated_surface_code,
)
from qec_certify.sim import sample_syndromes

CODES = [
    repetition_code(3),
    repetition_code(5),
    rotated_surface_code(3),
    rotated_surface_code(5),
    rotated_surface_code(7),
]


def code_id(code):
    return f"{code.name}-d{code.distance}"


def gf2_rank(matrix: np.ndarray) -> int:
    m = matrix.astype(np.uint8) % 2
    rank = 0
    for col in range(m.shape[1]):
        pivots = np.nonzero(m[rank:, col])[0]
        if len(pivots) == 0:
            continue
        pivot = rank + pivots[0]
        m[[rank, pivot]] = m[[pivot, rank]]
        for row in np.nonzero(m[:, col])[0]:
            if row != rank:
                m[row] ^= m[rank]
        rank += 1
        if rank == m.shape[0]:
            break
    return rank


@pytest.mark.parametrize("code", CODES, ids=code_id)
def test_stabilizers_commute_and_are_independent(code):
    n = code.num_data_qubits
    paulis = [s.pauli_string(n) for s in code.stabilizers]
    for a, b in itertools.combinations(paulis, 2):
        assert a.commutes(b)
    symplectic = np.array([np.concatenate(p.to_numpy()) for p in paulis])
    assert gf2_rank(symplectic) == len(code.stabilizers) == n - 1


@pytest.mark.parametrize("code", CODES, ids=code_id)
def test_logical_operators(code):
    lx = code.logical_pauli_string("X")
    lz = code.logical_pauli_string("Z")
    assert not lx.commutes(lz)
    for stab in code.stabilizers:
        s = stab.pauli_string(code.num_data_qubits)
        assert lx.commutes(s)
        assert lz.commutes(s)


@pytest.mark.parametrize("d", [3, 5, 7])
def test_rotated_surface_code_structure(d):
    code = rotated_surface_code(d)
    assert code.num_data_qubits == d * d
    for basis in "XZ":
        stabs = [s for s in code.stabilizers if s.basis == basis]
        assert len(stabs) == (d * d - 1) // 2
        assert sorted({s.weight for s in stabs}) == [2, 4]
        assert sum(s.weight == 2 for s in stabs) == d - 1
    assert len(code.logical_x) == len(code.logical_z) == d


def test_repetition_code_structure():
    code = repetition_code(5)
    assert [s.qubits for s in code.stabilizers] == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert all(s.basis == "Z" for s in code.stabilizers)
    assert code.logical_z == (0,)


@pytest.mark.parametrize(
    ("code", "basis"),
    [(c, b) for c in CODES for b in "XZ" if c.name == "rotated_surface" or b == "Z"],
    ids=lambda x: x if isinstance(x, str) else code_id(x),
)
def test_circuit_distance_equals_code_distance(code, basis):
    circuit = code_capacity_circuit(code, 0.1, basis)
    assert len(circuit.shortest_graphlike_error()) == code.distance


@pytest.mark.parametrize("code", CODES, ids=code_id)
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_circuit_detectors_and_observable(code, basis):
    circuit = code_capacity_circuit(code, 0.05, basis)
    assert circuit.num_detectors == len(code.stabilizers)
    assert circuit.num_observables == 1
    # Raises if any detector or observable were non-deterministic without noise.
    circuit.detector_error_model()
    coords = circuit.get_detector_coordinates()
    for k, stab in enumerate(code.stabilizers):
        assert coords[k] == [*stab.coords, 0.0]


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_noiseless_circuit_never_fires(basis):
    data = sample_syndromes(code_capacity_circuit(rotated_surface_code(3), 0.0, basis), 200, 1)
    assert not data.detection_events.any()
    assert not data.observable_flips.any()


def test_make_code():
    assert make_code("repetition", 3) == repetition_code(3)
    assert make_code("rotated_surface", 3) == rotated_surface_code(3)
    with pytest.raises(ValueError, match="unknown code family"):
        make_code("toric", 3)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        repetition_code(1)
    with pytest.raises(ValueError):
        rotated_surface_code(4)
    with pytest.raises(ValueError):
        Stabilizer("Y", (0,), (0.0, 0.0))
    with pytest.raises(ValueError):
        code_capacity_circuit(repetition_code(3), 0.1, basis="Y")
    with pytest.raises(ValueError):
        code_capacity_circuit(repetition_code(3), 0.8)
