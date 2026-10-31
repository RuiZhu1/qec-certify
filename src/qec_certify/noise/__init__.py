"""Noise models and noise-structure descriptors (M3).

Code-capacity noise models are Pauli channels on the data qubits; each writes its noise
layer into :func:`qec_certify.codes.code_capacity_circuit`. Descriptors are defined in
``docs/noise_descriptors.md``.
"""

from qec_certify.noise.descriptors import describe
from qec_certify.noise.models import (
    NOISE_MODELS,
    BiasedPauli,
    Channel,
    CorrelatedPairs,
    Depolarizing,
    NoiseModel,
    as_noise_model,
    make_noise,
    nearest_neighbour_pairs,
    noise_grid,
)

__all__ = [
    "NOISE_MODELS",
    "BiasedPauli",
    "Channel",
    "CorrelatedPairs",
    "Depolarizing",
    "NoiseModel",
    "as_noise_model",
    "describe",
    "make_noise",
    "nearest_neighbour_pairs",
    "noise_grid",
]
