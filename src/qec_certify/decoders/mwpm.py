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
    """

    def __init__(self, dem: stim.DetectorErrorModel, *, correlated: bool = False) -> None:
        self.num_detectors = dem.num_detectors
        self.num_observables = dem.num_observables
        self.correlated = correlated
        self.matching = pymatching.Matching.from_detector_error_model(
            dem, enable_correlations=correlated
        )

    @classmethod
    def from_circuit(cls, circuit: stim.Circuit, *, correlated: bool = False) -> MWPMDecoder:
        return cls(circuit.detector_error_model(decompose_errors=True), correlated=correlated)

    def predict(self, detection_events: np.ndarray) -> np.ndarray:
        dets = self._check_detection_events(detection_events)
        predictions = self.matching.decode_batch(dets, enable_correlations=self.correlated)
        return np.asarray(predictions, dtype=bool).reshape(len(dets), self.num_observables)
