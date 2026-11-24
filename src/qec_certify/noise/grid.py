"""Experiment noise grids, including circuit-level configurations."""

from __future__ import annotations

from typing import TypeAlias

from qec_certify.noise.circuit_level import CircuitLevelNoise
from qec_certify.noise.models import NoiseModel
from qec_certify.noise.models import noise_grid as code_capacity_noise_grid

ExperimentNoise: TypeAlias = NoiseModel | CircuitLevelNoise


def noise_grid(config: dict) -> list[ExperimentNoise]:
    """Expand a noise config, using ``circuit_level`` for uniform Stim circuit noise."""
    if config["model"] != CircuitLevelNoise.name:
        return code_capacity_noise_grid(config)
    probabilities = _as_list(config["p"])
    heterogeneity = _as_list(config.get("heterogeneity", 0.0))
    drift = _as_list(config.get("drift", 0.0))
    channels = {
        key: float(config[key])
        for key in (
            "after_clifford_depolarization",
            "before_round_data_depolarization",
            "before_measure_flip_probability",
            "after_reset_flip_probability",
        )
        if key in config
    }
    return [
        CircuitLevelNoise(
            float(p),
            **channels,
            heterogeneity=float(h),
            drift=float(d),
            noise_seed=int(config.get("noise_seed", 0)),
        )
        for p in probabilities
        for h in heterogeneity
        for d in drift
    ]


def _as_list(value):
    return value if isinstance(value, list) else [value]
