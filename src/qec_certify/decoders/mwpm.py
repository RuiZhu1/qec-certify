"""Minimum-weight perfect matching decoder (PyMatching)."""

from __future__ import annotations

import numpy as np
import pymatching
import stim

from qec_certify.decoders.base import Decoder


class MWPMDecoder(Decoder):
    """PyMatching on the graph of a detector error model.

    Hyperedge errors (e.g. Y errors in the surface code) must be decomposed into graphlike
    components; their X/Z correlation is then ignored, as in standard MWPM.
    """

    def __init__(self, dem: stim.DetectorErrorModel) -> None:
        self.num_detectors = dem.num_detectors
        self.num_observables = dem.num_observables
        self.matching = pymatching.Matching.from_detector_error_model(dem)

    @classmethod
    def from_circuit(cls, circuit: stim.Circuit) -> MWPMDecoder:
        return cls(circuit.detector_error_model(decompose_errors=True))

    def predict(self, detection_events: np.ndarray) -> np.ndarray:
        dets = self._check_detection_events(detection_events)
        predictions = self.matching.decode_batch(dets)
        return np.asarray(predictions, dtype=bool).reshape(len(dets), self.num_observables)
