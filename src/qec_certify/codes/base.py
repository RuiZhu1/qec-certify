"""Code descriptions shared by all code families."""

from __future__ import annotations

from dataclasses import dataclass

import stim

_BASES = ("X", "Z")


@dataclass(frozen=True)
class Stabilizer:
    """A CSS stabilizer generator: a product of a single Pauli type on data qubits.

    Attributes:
        basis: ``"X"`` or ``"Z"``.
        qubits: Indices of the data qubits in the support.
        coords: Spatial ``(x, y)`` position of the check, reused as detector coordinates.
    """

    basis: str
    qubits: tuple[int, ...]
    coords: tuple[float, float]

    def __post_init__(self) -> None:
        if self.basis not in _BASES:
            raise ValueError(f"basis must be 'X' or 'Z', got {self.basis!r}")
        if not self.qubits:
            raise ValueError("a stabilizer needs a non-empty support")

    @property
    def weight(self) -> int:
        return len(self.qubits)

    def pauli_string(self, num_qubits: int) -> stim.PauliString:
        return _pauli_string(self.basis, self.qubits, num_qubits)

    def mpp_targets(self) -> list[stim.GateTarget]:
        """Targets for measuring this stabilizer with Stim's ``MPP`` instruction."""
        target = stim.target_x if self.basis == "X" else stim.target_z
        targets: list[stim.GateTarget] = []
        for k, q in enumerate(self.qubits):
            if k:
                targets.append(stim.target_combiner())
            targets.append(target(q))
        return targets


@dataclass(frozen=True)
class StabilizerCode:
    """A CSS code with one logical qubit, described by its generators and logicals.

    Attributes:
        name: Code family name.
        distance: Code distance.
        data_coords: ``(x, y)`` coordinates of each data qubit; qubit ``q`` is at index ``q``.
        stabilizers: Independent stabilizer generators. Their order fixes the detector order.
        logical_x: Support of the logical X operator.
        logical_z: Support of the logical Z operator.
    """

    name: str
    distance: int
    data_coords: tuple[tuple[float, float], ...]
    stabilizers: tuple[Stabilizer, ...]
    logical_x: tuple[int, ...]
    logical_z: tuple[int, ...]

    @property
    def num_data_qubits(self) -> int:
        return len(self.data_coords)

    def logical_support(self, basis: str) -> tuple[int, ...]:
        if basis not in _BASES:
            raise ValueError(f"basis must be 'X' or 'Z', got {basis!r}")
        return self.logical_x if basis == "X" else self.logical_z

    def logical_pauli_string(self, basis: str) -> stim.PauliString:
        return _pauli_string(basis, self.logical_support(basis), self.num_data_qubits)


def _pauli_string(basis: str, qubits: tuple[int, ...], num_qubits: int) -> stim.PauliString:
    ps = stim.PauliString(num_qubits)
    for q in qubits:
        ps[q] = basis
    return ps
