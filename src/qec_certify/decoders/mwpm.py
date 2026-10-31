"""Minimum-weight perfect matching decoder (PyMatching)."""

from __future__ import annotations

import numpy as np
import pymatching
import stim

from qec_certify.decoders.base import Decoder


class MWPMDecoder(Decoder):
    """PyMatching on the graph of a detector error model.

    Hyperedge errors (e.g. Y errors in the surface code) must be decomposed into graphlike
    components; their X/Z correlation is then ignored, as in standard MWPM. With
    ``correlated=True`` PyMatching's correlated matching uses the decomposition: after a
    first matching it lowers the weight of edges that share a hyperedge with a matched one
    and matches again.

    Each edge of the graph carries one set of observables, but different mechanisms (or
    decomposition components) can flip the same detectors with different observables.
    PyMatching keeps the observables of whichever is added first, and Stim orders the DEM by
    targets, not probability, so a rare two-qubit component can relabel a boundary edge and
    turn single-qubit errors into logical failures. The decoder therefore gives every edge the
    observables of its most likely source; weights are unchanged, and DEMs without such
    conflicts (e.g. single-qubit Pauli noise) are untouched.

    Attributes:
        relabelled_edges: Number of edges whose observables were changed that way.
    """

    def __init__(self, dem: stim.DetectorErrorModel, *, correlated: bool = False) -> None:
        self.num_detectors = dem.num_detectors
        self.num_observables = dem.num_observables
        self.correlated = correlated
        self.matching = pymatching.Matching.from_detector_error_model(
            dem, enable_correlations=correlated
        )
        self.relabelled_edges = _use_most_likely_observables(self.matching, dem)

    @classmethod
    def from_circuit(cls, circuit: stim.Circuit, *, correlated: bool = False) -> MWPMDecoder:
        return cls(circuit.detector_error_model(decompose_errors=True), correlated=correlated)

    def predict(self, detection_events: np.ndarray) -> np.ndarray:
        dets = self._check_detection_events(detection_events)
        predictions = self.matching.decode_batch(dets, enable_correlations=self.correlated)
        return np.asarray(predictions, dtype=bool).reshape(len(dets), self.num_observables)


def most_likely_observables(dem: stim.DetectorErrorModel) -> dict[tuple[int, ...], set[int]]:
    """For every graphlike detector set in ``dem`` (whole mechanisms or decomposition
    components), the observables of the most probable mechanism that produces it."""
    best: dict[tuple[int, ...], tuple[float, set[int]]] = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        component: list[stim.DemTarget] = []
        for target in [*inst.targets_copy(), None]:
            if target is not None and not target.is_separator():
                component.append(target)
                continue
            dets = tuple(sorted(t.val for t in component if t.is_relative_detector_id()))
            obs = {t.val for t in component if t.is_logical_observable_id()}
            if dets and (dets not in best or p > best[dets][0]):
                best[dets] = (p, obs)
            component = []
    return {dets: obs for dets, (_, obs) in best.items()}


def _use_most_likely_observables(
    matching: pymatching.Matching, dem: stim.DetectorErrorModel
) -> int:
    expected = most_likely_observables(dem)
    relabelled = 0
    for u, v, data in matching.edges():
        key = (u,) if v is None else tuple(sorted((u, v)))
        if key not in expected or set(data["fault_ids"]) == expected[key]:
            continue
        kwargs = {
            "fault_ids": expected[key],
            "weight": data["weight"],
            "error_probability": data["error_probability"],
            "merge_strategy": "replace",
        }
        if v is None:
            matching.add_boundary_edge(u, **kwargs)
        else:
            matching.add_edge(u, v, **kwargs)
        relabelled += 1
    return relabelled
