"""Exact (not sampled) logical error rates for small codes."""

from __future__ import annotations

import numpy as np
import stim

from qec_certify.decoders.base import Decoder
from qec_certify.decoders.ml_exact import syndrome_class_distribution
from qec_certify.utils.bits import pack_bits, unpack_bits


def exact_logical_error_rate(
    decoder: Decoder,
    dem: stim.DetectorErrorModel,
    *,
    max_bits: int = 22,
    joint: np.ndarray | None = None,
    chunk_size: int = 1 << 20,
) -> float:
    """Exact LER of ``decoder`` under ``dem``, by decoding every one of the ``2^D`` syndromes.

    ``LER = sum_s sum_{l != decoder(s)} P(s, l)`` with ``P`` from
    :func:`~qec_certify.decoders.ml_exact.syndrome_class_distribution`. Pass ``joint`` to reuse
    an already computed distribution when scoring several decoders under the same DEM;
    syndromes are decoded ``chunk_size`` at a time to bound memory.
    """
    if joint is None:
        joint = syndrome_class_distribution(dem, max_bits=max_bits)
    num_syndromes = joint.shape[1]
    correct = np.empty(num_syndromes)
    for start in range(0, num_syndromes, chunk_size):
        syndromes = np.arange(start, min(start + chunk_size, num_syndromes))
        predicted = pack_bits(decoder.predict(unpack_bits(syndromes, dem.num_detectors)))
        correct[syndromes] = joint[predicted, syndromes]
    return float((joint.sum(axis=0) - correct).sum())
