"""Noise-structure descriptors, computed exactly from a model's Pauli channels.

Definitions are in ``docs/noise_descriptors.md``. All of them are functions of the Pauli
error on one data qubit, or on a pair of neighbouring data qubits, after marginalising the
rest of the noise: the channels touching those qubits are composed (Paulis multiply as XOR
of their ``x + 2 z`` codes, phases ignored) and every other qubit is summed out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qec_certify.noise.models import Channel, NoiseModel, nearest_neighbour_pairs

if TYPE_CHECKING:
    from qec_certify.codes.base import StabilizerCode

# Pauli indices, x + 2 z.
IDENTITY, X, Z, Y = 0, 1, 2, 3


def marginal(channels: list[Channel], qubits: tuple[int, ...]) -> np.ndarray:
    """Distribution of the Pauli error on ``qubits``, indexed ``e_0 + 4 e_1 + ...``."""
    k = len(qubits)
    out = np.zeros(4**k)
    out[0] = 1.0
    for channel in channels:
        shared = [q for q in channel.qubits if q in qubits]
        if shared:
            out = xor_convolve(out, _embed(channel, qubits))
    return out


def xor_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``out[i] = sum_j a[j] b[i XOR j]``: the distribution of the product of two Paulis."""
    index = np.arange(len(a))
    return np.array([np.dot(a, b[i ^ index]) for i in index])


def bias(qubit_marginal: np.ndarray) -> float:
    """``eta = p_Z / (p_X + p_Y)``; ``inf`` without X and Y errors, ``nan`` without any."""
    xy = qubit_marginal[X] + qubit_marginal[Y]
    if xy == 0:
        return float("nan") if qubit_marginal[Z] == 0 else float("inf")
    return float(qubit_marginal[Z] / xy)


def entropy_bits(distribution: np.ndarray) -> float:
    """Shannon entropy in bits."""
    positive = distribution[distribution > 0]
    return float(-(positive * np.log2(positive)).sum())


def mutual_information_bits(pair_marginal: np.ndarray) -> float:
    """``I(E_a; E_b)`` in bits for a two-qubit marginal indexed ``e_a + 4 e_b``."""
    joint = pair_marginal.reshape(4, 4)  # [e_b, e_a]
    return max(
        0.0,
        entropy_bits(joint.sum(axis=0)) + entropy_bits(joint.sum(axis=1)) - entropy_bits(joint),
    )


def bulk_pair(code: StabilizerCode) -> tuple[int, int]:
    """A data qubit with the most neighbours (nearest the centre on ties) and a neighbour."""
    pairs = nearest_neighbour_pairs(code)
    if not pairs:
        raise ValueError("the code has no neighbouring data qubits")
    degree = np.zeros(len(code.data_coords), dtype=int)
    for a, b in pairs:
        degree[a] += 1
        degree[b] += 1
    coords = np.array(code.data_coords, dtype=float)
    distance = np.linalg.norm(coords - coords.mean(axis=0), axis=1)
    qubit = min(range(len(coords)), key=lambda q: (-degree[q], distance[q], q))
    neighbour = min(b if a == qubit else a for a, b in pairs if qubit in (a, b))
    return qubit, neighbour


def describe(noise: NoiseModel, code: StabilizerCode) -> dict[str, float]:
    """Descriptors of ``noise`` on ``code``, for a bulk qubit and one of its neighbours.

    Keys: ``error_rate`` (probability of a non-identity Pauli on the qubit), ``bias``,
    ``entropy_bits`` (of the qubit's Pauli error) and ``neighbour_mutual_information_bits``.
    """
    channels = noise.channels(code)
    a, b = bulk_pair(code)
    single = marginal(channels, (a,))
    return {
        "error_rate": float(1 - single[IDENTITY]),
        "bias": bias(single),
        "entropy_bits": entropy_bits(single),
        "neighbour_mutual_information_bits": mutual_information_bits(marginal(channels, (a, b))),
    }


def _embed(channel: Channel, qubits: tuple[int, ...]) -> np.ndarray:
    """``channel`` marginalised onto the qubits it shares with ``qubits``, as a distribution
    over all of ``qubits`` (identity on the others)."""
    k = len(channel.qubits)
    # Axis i of the reshaped table is channel.qubits[i] (the table index is e_0 + 4 e_1 + ...).
    table = channel.probabilities.reshape((4,) * k).transpose(range(k - 1, -1, -1))
    keep = [i for i, q in enumerate(channel.qubits) if q in qubits]
    summed = table.sum(axis=tuple(i for i in range(k) if i not in keep))
    full = np.zeros((4,) * len(qubits))
    position = [qubits.index(channel.qubits[i]) for i in keep]
    index: list[slice | int] = [0] * len(qubits)
    for axis in position:
        index[axis] = slice(None)
    # ``summed`` has axes in channel order; reorder them to the order of ``qubits``.
    full[tuple(index)] = summed.transpose(np.argsort(position))
    return full.transpose(range(len(qubits) - 1, -1, -1)).reshape(-1)
