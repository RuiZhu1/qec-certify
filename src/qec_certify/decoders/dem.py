"""Decoders that consume an arbitrary detector error model through ``ldpc``."""

from __future__ import annotations

import abc

import numpy as np
import stim
from ldpc import BpLsdDecoder, BpOsdDecoder
from ldpc import UnionFindDecoder as LdpcUnionFindDecoder
from scipy.sparse import csc_matrix

from qec_certify.decoders.base import Decoder
from qec_certify.decoders.ml_exact import error_mechanisms


class DEMDecoder(Decoder):
    """Shared DEM-to-parity-check conversion and logical-observable reconstruction."""

    def __init__(self, dem: stim.DetectorErrorModel) -> None:
        self.num_detectors = dem.num_detectors
        self.num_observables = dem.num_observables
        probabilities, masks = error_mechanisms(dem)
        self._constant_syndrome = np.zeros(self.num_detectors, dtype=np.uint8)
        self._constant_observables = np.zeros(self.num_observables, dtype=bool)
        rows: list[int] = []
        columns: list[int] = []
        observable_rows: list[int] = []
        observable_columns: list[int] = []
        channel_probabilities: list[float] = []
        for p, mask in zip(probabilities, masks, strict=True):
            detectors = [j for j in range(self.num_detectors) if (mask >> j) & 1]
            observables = [
                j for j in range(self.num_observables) if (mask >> (self.num_detectors + j)) & 1
            ]
            if p >= 1.0:
                self._constant_syndrome[detectors] ^= 1
                self._constant_observables[observables] ^= True
                continue
            if not detectors:
                if p > 0.5:
                    self._constant_observables[observables] ^= True
                continue
            column = len(channel_probabilities)
            channel_probabilities.append(float(p))
            rows.extend(detectors)
            columns.extend([column] * len(detectors))
            observable_rows.extend(observables)
            observable_columns.extend([column] * len(observables))

        self.num_mechanisms = len(channel_probabilities)
        self.channel_probabilities = np.asarray(channel_probabilities, dtype=np.float64)
        self.check_matrix = csc_matrix(
            (
                np.ones(len(rows), dtype=np.uint8),
                (np.asarray(rows), np.asarray(columns)),
            ),
            shape=(self.num_detectors, self.num_mechanisms),
            dtype=np.uint8,
        )
        self.observable_matrix = csc_matrix(
            (
                np.ones(len(observable_rows), dtype=np.uint8),
                (np.asarray(observable_rows), np.asarray(observable_columns)),
            ),
            shape=(self.num_observables, self.num_mechanisms),
            dtype=np.uint8,
        )

    @classmethod
    def from_circuit(cls, circuit: stim.Circuit, **kwargs) -> DEMDecoder:
        return cls(circuit.detector_error_model(), **kwargs)

    @abc.abstractmethod
    def _decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Return the estimated visible error mechanisms for one adjusted syndrome."""

    def predict(self, detection_events: np.ndarray) -> np.ndarray:
        dets = self._check_detection_events(detection_events)
        predictions = np.empty((len(dets), self.num_observables), dtype=bool)
        for k, det in enumerate(dets):
            if self.num_mechanisms:
                syndrome = det.astype(np.uint8) ^ self._constant_syndrome
                mechanisms = np.asarray(self._decode(syndrome), dtype=np.uint8)
                logical = np.asarray(self.observable_matrix @ mechanisms).ravel() % 2
                predictions[k] = logical.astype(bool) ^ self._constant_observables
            else:
                predictions[k] = self._constant_observables
        return predictions


class BPOSDDecoder(DEMDecoder):
    """Belief propagation with ordered-statistics decoding on the DEM's mechanisms."""

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        *,
        max_iter: int = 100,
        osd_order: int = 7,
        omp_thread_count: int = 1,
    ) -> None:
        super().__init__(dem)
        if osd_order < 0:
            raise ValueError(f"osd_order must be non-negative, got {osd_order}")
        self.osd_order = min(osd_order, self.num_mechanisms)
        self.decoder = (
            BpOsdDecoder(
                self.check_matrix,
                error_channel=self.channel_probabilities.tolist(),
                max_iter=max_iter,
                bp_method="product_sum",
                osd_method="OSD_CS",
                osd_order=self.osd_order,
                omp_thread_count=omp_thread_count,
            )
            if self.num_mechanisms
            else None
        )

    def _decode(self, syndrome: np.ndarray) -> np.ndarray:
        assert self.decoder is not None
        return self.decoder.decode(syndrome)


class BPLSDDecoder(DEMDecoder):
    """Belief propagation with localized-statistics decoding on DEM mechanisms."""

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        *,
        max_iter: int = 100,
        lsd_order: int = 0,
        lsd_method: str = "LSD_0",
        omp_thread_count: int = 1,
    ) -> None:
        super().__init__(dem)
        self.decoder = (
            BpLsdDecoder(
                self.check_matrix,
                error_channel=self.channel_probabilities.tolist(),
                max_iter=max_iter,
                bp_method="product_sum",
                lsd_order=lsd_order,
                lsd_method=lsd_method,
                omp_thread_count=omp_thread_count,
            )
            if self.num_mechanisms
            else None
        )

    def _decode(self, syndrome: np.ndarray) -> np.ndarray:
        assert self.decoder is not None
        return self.decoder.decode(syndrome)


class UnionFindDecoder(DEMDecoder):
    """Union-Find peeling decoder, included as a low-latency DEM baseline.

    Input syndromes must be realizable by the DEM, as Stim's detector samples are. In ldpc
    2.4.1, both Union-Find methods can fail to return for an inconsistent input syndrome.
    """

    def __init__(self, dem: stim.DetectorErrorModel, *, matrix_solve: bool = False) -> None:
        super().__init__(dem)
        self.decoder = (
            LdpcUnionFindDecoder(
                self.check_matrix, uf_method="matrix_solve" if matrix_solve else "peeling"
            )
            if self.num_mechanisms
            else None
        )

    def _decode(self, syndrome: np.ndarray) -> np.ndarray:
        assert self.decoder is not None
        return self.decoder.decode(syndrome)
