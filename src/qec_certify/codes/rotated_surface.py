"""Rotated surface code."""

from __future__ import annotations

from qec_certify.codes.base import Stabilizer, StabilizerCode


def rotated_surface_code(distance: int) -> StabilizerCode:
    """Distance-``d`` rotated surface code, ``[[d^2, 1, d]]``, for odd ``d >= 3``.

    Data qubit ``(row, col)`` has index ``row * d + col`` and coordinates ``(col, row)``.
    Plaquette ``(i, j)`` with ``0 <= i, j <= d`` touches data qubits ``(i-1..i, j-1..j)`` and
    is X-type when ``i + j`` is even, Z-type otherwise. Weight-2 X plaquettes sit on the
    top and bottom boundaries, weight-2 Z plaquettes on the left and right boundaries.
    Logical Z runs along row 0 and logical X along column 0.

    Stabilizers are ordered X-type first, then Z-type, each in row-major plaquette order.
    """
    if distance < 3 or distance % 2 == 0:
        raise ValueError(f"distance must be odd and >= 3, got {distance}")
    d = distance

    def qubit(row: int, col: int) -> int:
        return row * d + col

    x_stabs: list[Stabilizer] = []
    z_stabs: list[Stabilizer] = []
    for i in range(d + 1):
        for j in range(d + 1):
            basis = "X" if (i + j) % 2 == 0 else "Z"
            on_top_or_bottom = i in (0, d)
            on_left_or_right = j in (0, d)
            if on_top_or_bottom and on_left_or_right:
                continue
            if on_top_or_bottom and basis != "X":
                continue
            if on_left_or_right and basis != "Z":
                continue
            support = tuple(
                qubit(r, c) for r in (i - 1, i) for c in (j - 1, j) if 0 <= r < d and 0 <= c < d
            )
            stab = Stabilizer(basis, support, (j - 0.5, i - 0.5))
            (x_stabs if basis == "X" else z_stabs).append(stab)

    return StabilizerCode(
        name="rotated_surface",
        distance=d,
        data_coords=tuple((float(c), float(r)) for r in range(d) for c in range(d)),
        stabilizers=tuple(x_stabs + z_stabs),
        logical_x=tuple(qubit(r, 0) for r in range(d)),
        logical_z=tuple(qubit(0, c) for c in range(d)),
    )
