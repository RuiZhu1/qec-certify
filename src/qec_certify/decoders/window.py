"""Fixed-window causal MWPM on projected detector error models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim

from qec_certify.decoders.mwpm import MWPMDecoder
from qec_certify.decoders.streaming import StreamingDecoder


@dataclass(frozen=True)
class _Component:
    detectors: tuple[int, ...]
    observables: tuple[int, ...]


@dataclass(frozen=True)
class _Mechanism:
    probability: float
    components: tuple[_Component, ...]


@dataclass(frozen=True)
class _MatchInfo:
    probability: float
    observables: tuple[int, ...]
    latest_round: int
    internal: bool


@dataclass(frozen=True)
class _WindowDecoder:
    decoder: MWPMDecoder
    matches: dict[tuple[int, ...], _MatchInfo]


class SlidingWindowMWPMDecoder(StreamingDecoder):
    """Decode only the latest ``window_rounds`` with a precomputed local MWPM per round.

    Mechanisms crossing a window boundary are projected onto their in-window detectors and
    become boundary mechanisms. This is an approximation when the window is shorter than the
    shot; observable targets are retained. With a full-shot window there is no projection and
    the final prediction matches the offline decoder exactly.
    """

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        *,
        rounds: int,
        window_rounds: int,
        correlated: bool = False,
    ) -> None:
        if not isinstance(rounds, int) or rounds < 1:
            raise ValueError(f"rounds must be a positive integer, got {rounds!r}")
        if not isinstance(window_rounds, int) or window_rounds < 1:
            raise ValueError(f"window_rounds must be a positive integer, got {window_rounds!r}")
        if dem.num_detectors % rounds:
            raise ValueError(
                f"{dem.num_detectors} detectors are not divisible into {rounds} rounds"
            )
        self.rounds = rounds
        self.window_rounds = min(window_rounds, rounds)
        self.num_detectors_per_round = dem.num_detectors // rounds
        self.num_observables = dem.num_observables
        self.correlated = correlated
        self.mode = f"sliding_window_mwpm_w{self.window_rounds}"
        self.approximation = (
            "exact_full_history" if self.window_rounds == rounds else "projected_dem_boundary"
        )
        mechanisms = _mechanisms(dem)
        coordinates = dem.get_detector_coordinates()
        self._decoders = []
        for current_round in range(rounds):
            first_round = max(0, current_round - self.window_rounds + 1)
            start = first_round * self.num_detectors_per_round
            end = (current_round + 1) * self.num_detectors_per_round
            local_dem, matches = _project_dem(
                mechanisms,
                start,
                end,
                self.num_observables,
                self.num_detectors_per_round,
                coordinates,
            )
            decoder = MWPMDecoder(local_dem, correlated=correlated)
            self._decoders.append(_WindowDecoder(decoder, matches))
        self._buffer = np.zeros((self.window_rounds, self.num_detectors_per_round), dtype=bool)
        self._latest = np.zeros(self.num_observables, dtype=bool)
        self.rounds_processed = 0

    @classmethod
    def from_circuit(
        cls,
        circuit: stim.Circuit,
        *,
        rounds: int,
        window_rounds: int,
        correlated: bool = False,
    ) -> SlidingWindowMWPMDecoder:
        return cls(
            circuit.detector_error_model(decompose_errors=True),
            rounds=rounds,
            window_rounds=window_rounds,
            correlated=correlated,
        )

    def reset(self) -> None:
        self._buffer.fill(False)
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
        self._buffer[self.rounds_processed % self.window_rounds] = events.astype(bool, copy=False)
        first_round = max(0, self.rounds_processed - self.window_rounds + 1)
        indices = np.arange(first_round, self.rounds_processed + 1) % self.window_rounds
        window = self._buffer[indices].reshape(1, -1)
        self._latest = self._decoders[self.rounds_processed].decoder.predict(window)[0]
        self.rounds_processed += 1
        return self._latest.copy()

    def finalize(self) -> np.ndarray:
        if self.rounds_processed != self.rounds:
            raise ValueError(f"expected {self.rounds} rounds, got {self.rounds_processed}")
        return self._latest.copy()


class DelayedCommitMWPMDecoder(SlidingWindowMWPMDecoder):
    """Overlapping windows that commit only fully internal errors from the oldest round.

    Newer and boundary-crossing components stay uncommitted until a later window. The final
    estimate combines committed logical flips with the remaining tail. This is a causal
    boundary-anchoring heuristic, not an exact incremental MWPM proof.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mode = f"delayed_commit_mwpm_w{self.window_rounds}"
        self.approximation = (
            "exact_full_history" if self.window_rounds == self.rounds else "delayed_commit_boundary"
        )
        self._committed = np.zeros(self.num_observables, dtype=bool)

    def reset(self) -> None:
        super().reset()
        self._committed.fill(False)

    def decode_round(self, detection_events: np.ndarray) -> np.ndarray:
        if self.rounds_processed >= self.rounds:
            raise ValueError(f"shot already contains {self.rounds} rounds; call reset()")
        events = np.asarray(detection_events)
        if events.shape != (self.num_detectors_per_round,):
            raise ValueError(
                f"expected one round of shape ({self.num_detectors_per_round},), got {events.shape}"
            )
        self._buffer[self.rounds_processed % self.window_rounds] = events.astype(bool, copy=False)
        first_round = max(0, self.rounds_processed - self.window_rounds + 1)
        indices = np.arange(first_round, self.rounds_processed + 1) % self.window_rounds
        window = self._buffer[indices].reshape(1, -1)
        commit_round = self.rounds_processed - self.window_rounds + 1
        entry = self._decoders[self.rounds_processed]
        edges = entry.decoder.matching.decode_to_edges_array(
            window[0], enable_correlations=self.correlated
        )
        remaining = np.zeros(self.num_observables, dtype=bool)
        for u, v in edges:
            key = (int(u),) if v == -1 else tuple(sorted((int(u), int(v))))
            info = entry.matches[key]
            if commit_round >= 0 and info.internal and info.latest_round == commit_round:
                self._committed[list(info.observables)] ^= True
            else:
                remaining[list(info.observables)] ^= True
        self._latest = self._committed ^ remaining
        self.rounds_processed += 1
        return self._latest.copy()


def _mechanisms(dem: stim.DetectorErrorModel) -> list[_Mechanism]:
    mechanisms: list[_Mechanism] = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        probability = inst.args_copy()[0]
        components: list[_Component] = []
        component: list[stim.DemTarget] = []
        for target in [*inst.targets_copy(), None]:
            if target is not None and not target.is_separator():
                component.append(target)
                continue
            detectors = tuple(
                target.val for target in component if target.is_relative_detector_id()
            )
            observables = tuple(
                target.val for target in component if target.is_logical_observable_id()
            )
            if detectors:
                components.append(_Component(detectors, observables))
            component = []
        if components:
            mechanisms.append(_Mechanism(probability, tuple(components)))
    return mechanisms


def _project_dem(
    mechanisms: list[_Mechanism],
    start: int,
    end: int,
    num_observables: int,
    detectors_per_round: int,
    coordinates: dict[int, list[float]],
) -> tuple[stim.DetectorErrorModel, dict[tuple[int, ...], _MatchInfo]]:
    projected = stim.DetectorErrorModel()
    matches: dict[tuple[int, ...], _MatchInfo] = {}
    for mechanism in mechanisms:
        targets: list[stim.DemTarget] = []
        for component in mechanism.components:
            local_detectors = tuple(d - start for d in component.detectors if start <= d < end)
            if not local_detectors:
                continue
            if targets:
                targets.append(stim.DemTarget.separator())
            targets += [stim.target_relative_detector_id(d) for d in local_detectors]
            targets += [stim.target_logical_observable_id(o) for o in component.observables]
            key = tuple(sorted(local_detectors))
            info = _MatchInfo(
                mechanism.probability,
                component.observables,
                max(component.detectors) // detectors_per_round,
                len(local_detectors) == len(component.detectors),
            )
            if key not in matches or info.probability > matches[key].probability:
                matches[key] = info
        if targets:
            projected.append("error", mechanism.probability, targets)
    for local_id, full_id in enumerate(range(start, end)):
        projected.append(
            "detector", coordinates.get(full_id, []), [stim.target_relative_detector_id(local_id)]
        )
    for observable in range(num_observables):
        projected.append("logical_observable", [], [stim.target_logical_observable_id(observable)])
    return projected, matches
