"""Causal streaming interfaces and a reference replay adapter."""

from __future__ import annotations

import abc

import numpy as np

from qec_certify.decoders.base import Decoder
from qec_certify.decoders.dem import UnionFindDecoder


class StreamingDecoder(abc.ABC):
    """Consume one detector round at a time without access to future rounds."""

    num_detectors_per_round: int
    num_observables: int
    rounds: int
    mode: str

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear all shot state."""

    @abc.abstractmethod
    def decode_round(self, detection_events: np.ndarray) -> np.ndarray:
        """Return the current observable estimate for one new detector round."""


class CausalReplayDecoder(StreamingDecoder):
    """Replay a full-history offline decoder after every newly arrived round.

    Future detectors are fixed to zero. Intermediate outputs are causal current estimates,
    not corrections to apply; the final estimate exactly matches the offline decoder. This is
    a correctness and latency reference with quadratic total replay work over a shot, not an
    optimized incremental matching implementation.
    """

    mode = "causal_replay_batch1"

    def __init__(self, decoder: Decoder, *, rounds: int) -> None:
        if isinstance(decoder, UnionFindDecoder):
            raise ValueError("UnionFindDecoder does not support inconsistent causal prefixes")
        if not isinstance(rounds, int) or rounds < 1:
            raise ValueError(f"rounds must be a positive integer, got {rounds!r}")
        if decoder.num_detectors % rounds:
            raise ValueError(
                f"{decoder.num_detectors} detectors are not divisible into {rounds} rounds"
            )
        self.decoder = decoder
        self.rounds = rounds
        self.num_detectors_per_round = decoder.num_detectors // rounds
        self.num_observables = decoder.num_observables
        self._history = np.zeros(decoder.num_detectors, dtype=bool)
        self._latest = np.zeros(self.num_observables, dtype=bool)
        self.rounds_processed = 0

    def reset(self) -> None:
        self._history.fill(False)
        self._latest.fill(False)
        self.rounds_processed = 0

    def decode_round(self, detection_events: np.ndarray) -> np.ndarray:
        if self.rounds_processed >= self.rounds:
            raise ValueError(f"shot already contains {self.rounds} rounds; call reset()")
        events = np.asarray(detection_events)
        if events.shape != (self.num_detectors_per_round,):
            raise ValueError(
                f"expected one round of shape ({self.num_detectors_per_round},), got {events.shape}"
            )
        start = self.rounds_processed * self.num_detectors_per_round
        self._history[start : start + self.num_detectors_per_round] = events.astype(
            bool, copy=False
        )
        self._latest = self.decoder.predict(self._history[None, :])[0]
        self.rounds_processed += 1
        return self._latest.copy()

    def finalize(self) -> np.ndarray:
        if self.rounds_processed != self.rounds:
            raise ValueError(f"expected {self.rounds} rounds, got {self.rounds_processed}")
        return self._latest.copy()
