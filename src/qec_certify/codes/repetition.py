"""Bit-flip repetition code."""

from __future__ import annotations

from qec_certify.codes.base import Stabilizer, StabilizerCode


def repetition_code(distance: int) -> StabilizerCode:
    """Distance-``d`` bit-flip repetition code on a line of ``d`` data qubits.

    Stabilizers are ``Z_i Z_{i+1}``; logical Z is ``Z_0`` and logical X is ``X^{⊗d}``.
    Only X/Y errors are detected, so the code protects Z-basis memory only.
    """
    if distance < 2:
        raise ValueError(f"distance must be >= 2, got {distance}")
    data_coords = tuple((float(i), 0.0) for i in range(distance))
    stabilizers = tuple(Stabilizer("Z", (i, i + 1), (i + 0.5, 0.0)) for i in range(distance - 1))
    return StabilizerCode(
        name="repetition",
        distance=distance,
        data_coords=data_coords,
        stabilizers=stabilizers,
        logical_x=tuple(range(distance)),
        logical_z=(0,),
    )
