"""Stim circuits for memory experiments."""

from __future__ import annotations

import stim

from qec_certify.codes.base import StabilizerCode
from qec_certify.noise.circuit_level import CircuitLevelNoise
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


def circuit_level_circuit(
    code: StabilizerCode,
    noise: float | CircuitLevelNoise,
    rounds: int,
    basis: str = "Z",
) -> stim.Circuit:
    """Rotated-surface-code memory using Stim's standard syndrome-extraction schedule.

    Base noise strength ``p`` is applied after Cliffords, before measurement, after reset
    and to data qubits at the start of each round, with optional channel overrides. The noise
    object can also apply seeded qubit heterogeneity and drift between rounds. The generated
    circuit includes detectors and one logical observable. Unlike
    :func:`code_capacity_circuit`, this circuit contains explicit ancillas and noisy multi-round
    stabilizer extraction.
    """
    if code.name != "rotated_surface":
        raise ValueError(f"circuit-level experiments require rotated_surface, got {code.name!r}")
    if basis not in ("X", "Z"):
        raise ValueError(f"basis must be 'X' or 'Z', got {basis!r}")
    if not isinstance(rounds, int) or rounds < 1:
        raise ValueError(f"rounds must be a positive integer, got {rounds!r}")
    noise = noise if isinstance(noise, CircuitLevelNoise) else CircuitLevelNoise(float(noise))
    circuit = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis.lower()}",
        distance=code.distance,
        rounds=rounds,
        **noise.stim_arguments(),
    )
    return noise.apply_to(circuit, rounds)
