"""Seeded sampling of detection events from Stim circuits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim


@dataclass(frozen=True)
class SyndromeData:
    """Sampled shots.

    Attributes:
        detection_events: Boolean array of shape ``(shots, num_detectors)``.
        observable_flips: Boolean array of shape ``(shots, num_observables)``.
    """

    detection_events: np.ndarray
    observable_flips: np.ndarray

    def __post_init__(self) -> None:
        if self.detection_events.ndim != 2 or self.observable_flips.ndim != 2:
            raise ValueError("detection_events and observable_flips must be 2-D")
        if len(self.detection_events) != len(self.observable_flips):
            raise ValueError("detection_events and observable_flips differ in shot count")

    @property
    def shots(self) -> int:
        return len(self.detection_events)


def sample_syndromes(circuit: stim.Circuit, shots: int, seed: int) -> SyndromeData:
    """Sample ``shots`` shots of ``circuit`` with Stim's detector sampler.

    Results are reproducible for a fixed ``seed``, Stim version and machine architecture.
    """
    if shots < 0:
        raise ValueError(f"shots must be non-negative, got {shots}")
    sampler = circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots, separate_observables=True)
    return SyndromeData(detection_events=dets, observable_flips=obs)
