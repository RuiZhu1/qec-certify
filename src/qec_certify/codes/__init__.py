"""Code construction: repetition and rotated surface codes, plus their Stim circuits."""

from __future__ import annotations

from qec_certify.codes.base import Stabilizer, StabilizerCode
from qec_certify.codes.circuits import circuit_level_circuit, code_capacity_circuit
from qec_certify.codes.reference import REFERENCE_CIRCUITS, load_reference_circuit
from qec_certify.codes.repetition import repetition_code
from qec_certify.codes.rotated_surface import rotated_surface_code

CODE_FAMILIES = {
    "repetition": repetition_code,
    "rotated_surface": rotated_surface_code,
}


def make_code(family: str, distance: int) -> StabilizerCode:
    """Build a code by family name (see ``CODE_FAMILIES``)."""
    try:
        builder = CODE_FAMILIES[family]
    except KeyError:
        raise ValueError(
            f"unknown code family {family!r}; choose from {sorted(CODE_FAMILIES)}"
        ) from None
    return builder(distance)


__all__ = [
    "CODE_FAMILIES",
    "REFERENCE_CIRCUITS",
    "Stabilizer",
    "StabilizerCode",
    "circuit_level_circuit",
    "code_capacity_circuit",
    "load_reference_circuit",
    "make_code",
    "repetition_code",
    "rotated_surface_code",
]
