"""Code-capacity and circuit-level noise models, plus noise-structure descriptors.

Code-capacity noise models are Pauli channels on the data qubits; each writes its noise
layer into :func:`qec_certify.codes.code_capacity_circuit`. Descriptors are defined in
``docs/noise_descriptors.md`` and do not apply to ``CircuitLevelNoise``.
"""

from qec_certify.noise.circuit_level import CircuitLevelNoise
from qec_certify.noise.descriptors import describe
from qec_certify.noise.grid import ExperimentNoise, noise_grid
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
)

__all__ = [
    "NOISE_MODELS",
    "BiasedPauli",
    "Channel",
    "CircuitLevelNoise",
    "CorrelatedPairs",
    "Depolarizing",
    "ExperimentNoise",
    "NoiseModel",
    "as_noise_model",
    "describe",
    "make_noise",
    "nearest_neighbour_pairs",
    "noise_grid",
]
