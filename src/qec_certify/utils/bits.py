"""Little-endian packing of boolean rows into integer indices."""

from __future__ import annotations

import numpy as np

MAX_PACKED_BITS = 62


def pack_bits(bits: np.ndarray) -> np.ndarray:
    """Pack each row of a ``(rows, k)`` boolean array into an int64; column ``j`` is bit ``j``."""
    bits = np.asarray(bits)
    if bits.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {bits.shape}")
    k = bits.shape[1]
    if k > MAX_PACKED_BITS:
        raise ValueError(f"cannot pack more than {MAX_PACKED_BITS} bits, got {k}")
    weights = np.left_shift(np.int64(1), np.arange(k, dtype=np.int64))
    return bits.astype(np.int64) @ weights


def unpack_bits(values: np.ndarray, k: int) -> np.ndarray:
    """Inverse of :func:`pack_bits`: a ``(rows, k)`` boolean array."""
    values = np.asarray(values, dtype=np.int64)
    return ((values[:, None] >> np.arange(k, dtype=np.int64)) & 1).astype(bool)
