"""Failure-aware fallback from a Relay-BP decoder to a reliable DEM decoder."""

from __future__ import annotations

import numpy as np

from qec_certify.decoders.base import Decoder
from qec_certify.decoders.relay import RelayBPDecoder


class FailureFallbackDecoder(Decoder):
    """Use the fallback decoder only for shots whose Relay-BP decode was unsuccessful."""

    def __init__(self, primary: RelayBPDecoder, fallback: Decoder) -> None:
        if primary.num_detectors != fallback.num_detectors:
            raise ValueError("primary and fallback detector counts differ")
        if primary.num_observables != fallback.num_observables:
            raise ValueError("primary and fallback observable counts differ")
        self.primary = primary
        self.fallback = fallback
        self.num_detectors = primary.num_detectors
        self.num_observables = primary.num_observables

    def predict(self, detection_events: np.ndarray) -> np.ndarray:
        dets = self._check_detection_events(detection_events)
        predictions = self.primary.predict(dets).copy()
        self.last_primary_success = self.primary.last_decode_success.copy()
        self.last_decode_iterations = self.primary.last_decode_iterations.copy()
        self.last_fallback = ~self.last_primary_success
        self.last_fallback_success = np.ones(len(dets), dtype=bool)
        if self.last_fallback.any():
            predictions[self.last_fallback] = self.fallback.predict(dets[self.last_fallback])
            if isinstance(self.fallback, RelayBPDecoder):
                self.last_fallback_success[self.last_fallback] = self.fallback.last_decode_success
        self.last_final_success = self.last_primary_success | self.last_fallback_success
        return predictions
