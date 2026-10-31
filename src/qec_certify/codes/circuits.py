"""Stim circuits for memory experiments."""

from __future__ import annotations

import stim

from qec_certify.codes.base import StabilizerCode
from qec_certify.noise.models import NoiseModel, as_noise_model


def code_capacity_circuit(
    code: StabilizerCode, noise: float | NoiseModel, basis: str = "Z"
) -> stim.Circuit:
    """One-shot memory experiment under code-capacity noise.

    ``noise`` is a :class:`~qec_certify.noise.NoiseModel`, or a probability ``p`` for
    i.i.d. depolarizing noise. Every operation except the noise layer is ideal:

    1. Prepare each data qubit in the +1 eigenstate of ``basis``.
    2. Measure every stabilizer with ``MPP`` (projects onto the code space).
    3. Apply the noise layer to the data qubits (``DEPOLARIZE1(p)`` for a probability).
    4. Measure every stabilizer again. Detector ``k`` compares the two outcomes of
       ``code.stabilizers[k]``.
    5. Measure every data qubit in ``basis``. Observable 0 is the logical operator of
       ``basis``.

    Both X- and Z-type detectors are kept, so decoders can exploit the X/Z correlation
    introduced by Y errors.
    """
    if basis not in ("X", "Z"):
        raise ValueError(f"basis must be 'X' or 'Z', got {basis!r}")
    noise = as_noise_model(noise)

    n = code.num_data_qubits
    m = len(code.stabilizers)
    data = list(range(n))
    stabilizer_targets = [t for s in code.stabilizers for t in s.mpp_targets()]

    circuit = stim.Circuit()
    for q, coords in enumerate(code.data_coords):
        circuit.append("QUBIT_COORDS", [q], list(coords))
    circuit.append("R" if basis == "Z" else "RX", data)
    circuit.append("MPP", stabilizer_targets)
    circuit.append("TICK")
    noise.append_to(circuit, code)
    circuit.append("TICK")
    circuit.append("MPP", stabilizer_targets)
    for k, stab in enumerate(code.stabilizers):
        circuit.append(
            "DETECTOR",
            [stim.target_rec(-m + k), stim.target_rec(-2 * m + k)],
            [*stab.coords, 0.0],
        )
    circuit.append("M" if basis == "Z" else "MX", data)
    circuit.append(
        "OBSERVABLE_INCLUDE",
        [stim.target_rec(-n + q) for q in code.logical_support(basis)],
        0,
    )
    return circuit
