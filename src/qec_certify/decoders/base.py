"""Common decoder interface."""

from __future__ import annotations

import abc

import numpy as np

from qec_certify.sim import SyndromeData


class Decoder(abc.ABC):
    """Maps detection events to predicted logical-observable flips.

    Subclasses set ``num_detectors`` and ``num_observables`` and implement :meth:`predict`.
    """

    num_detectors: int
    num_observables: int

    def fit(self, data: SyndromeData | None = None) -> Decoder:
        """Train on sampled data. Classical decoders ignore ``data``."""
        return self

    @abc.abstractmethod
    def predict(self, detection_events: np.ndarray) -> np.ndarray:
        """Predict observable flips.

        Args:
            detection_events: Boolean array of shape ``(shots, num_detectors)``.

        Returns:
            Boolean array of shape ``(shots, num_observables)``.
        """

    def _check_detection_events(self, detection_events: np.ndarray) -> np.ndarray:
        dets = np.asarray(detection_events)
        if dets.ndim != 2 or dets.shape[1] != self.num_detectors:
            raise ValueError(
                f"expected detection events of shape (shots, {self.num_detectors}), "
                f"got {dets.shape}"
            )
        return dets.astype(bool, copy=False)
