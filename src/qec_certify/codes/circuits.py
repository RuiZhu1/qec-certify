"""Stim circuits for memory experiments."""

from __future__ import annotations

import stim

from qec_certify.codes.base import StabilizerCode


def code_capacity_circuit(code: StabilizerCode, p: float, basis: str = "Z") -> stim.Circuit:
    """One-shot memory experiment under code-capacity i.i.d. depolarizing noise.

    Every operation except the noise layer is ideal:

    1. Prepare each data qubit in the +1 eigenstate of ``basis``.
    2. Measure every stabilizer with ``MPP`` (projects onto the code space).
    3. Apply ``DEPOLARIZE1(p)`` to every data qubit.
    4. Measure every stabilizer again. Detector ``k`` compares the two outcomes of
       ``code.stabilizers[k]``.
    5. Measure every data qubit in ``basis``. Observable 0 is the logical operator of
       ``basis``.

    Both X- and Z-type detectors are kept, so decoders can exploit the X/Z correlation
    introduced by Y errors.
    """
    if basis not in ("X", "Z"):
        raise ValueError(f"basis must be 'X' or 'Z', got {basis!r}")
    if not 0.0 <= p <= 0.75:
        raise ValueError(f"depolarizing probability must be in [0, 3/4], got {p}")

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
    circuit.append("DEPOLARIZE1", data, p)
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
