"""Exact (not sampled) logical error rates for small codes."""

from __future__ import annotations

import numpy as np
import stim

from qec_certify.decoders.base import Decoder
from qec_certify.decoders.ml_exact import syndrome_class_distribution
from qec_certify.utils.bits import pack_bits, unpack_bits


def exact_logical_error_rate(
    decoder: Decoder, dem: stim.DetectorErrorModel, *, max_bits: int = 22
) -> float:
    """Exact LER of ``decoder`` under ``dem``, by decoding every one of the ``2^D`` syndromes.

    ``LER = sum_s sum_{l != decoder(s)} P(s, l)`` with ``P`` from
    :func:`~qec_certify.decoders.ml_exact.syndrome_class_distribution`.
    """
    joint = syndrome_class_distribution(dem, max_bits=max_bits)
    syndromes = np.arange(joint.shape[1])
    predicted = pack_bits(decoder.predict(unpack_bits(syndromes, dem.num_detectors)))
    return float((joint.sum(axis=0) - joint[predicted, syndromes]).sum())
