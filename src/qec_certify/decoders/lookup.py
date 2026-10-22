"""Empirical lookup-table decoder: the simplest learned decoder.

It memorises, for every syndrome seen in training, the most frequent logical class, and
defers to a fallback decoder elsewhere. On small codes with a lot of data it converges to
exact ML, so it is the baseline a learned decoder has to beat to show that it generalises
rather than memorises.
"""

from __future__ import annotations

import numpy as np

from qec_certify.decoders.base import Decoder
from qec_certify.sim import SyndromeData
from qec_certify.utils.bits import pack_bits, unpack_bits


class LookupTableDecoder(Decoder):
    """Majority vote per seen syndrome; unseen syndromes and ties go to ``fallback``.

    Without a fallback, unseen syndromes and ties decode to no flip.

    Attributes:
        syndromes: Sorted packed syndromes seen in training.
        classes: Predicted class (packed observable flips) for each entry of ``syndromes``,
            or ``-1`` where the vote is tied and the fallback decides.
    """

    def __init__(
        self, num_detectors: int, num_observables: int, fallback: Decoder | None = None
    ) -> None:
        self.num_detectors = num_detectors
        self.num_observables = num_observables
        self.fallback = fallback
        self.syndromes = np.zeros(0, dtype=np.int64)
        self.classes = np.zeros(0, dtype=np.int64)

    def fit(self, data: SyndromeData | None = None) -> LookupTableDecoder:
        if data is None:
            raise ValueError("LookupTableDecoder.fit requires training data")
        syndromes = pack_bits(self._check_detection_events(data.detection_events))
        labels = pack_bits(data.observable_flips)
        self.syndromes, index = np.unique(syndromes, return_inverse=True)
        counts = np.zeros((len(self.syndromes), 2**self.num_observables), dtype=np.int64)
        np.add.at(counts, (index, labels), 1)
        best = counts.max(axis=1, keepdims=True)
        tied = (counts == best).sum(axis=1) > 1
        self.classes = np.where(tied, -1, counts.argmax(axis=1))
        return self

    def predict(self, detection_events: np.ndarray) -> np.ndarray:
        dets = self._check_detection_events(detection_events)
        packed = pack_bits(dets)
        classes = np.full(len(dets), -1, dtype=np.int64)
        if len(self.syndromes):
            pos = np.minimum(np.searchsorted(self.syndromes, packed), len(self.syndromes) - 1)
            seen = self.syndromes[pos] == packed
            classes[seen] = self.classes[pos[seen]]
        undecided = classes < 0
        if self.fallback is not None and undecided.any():
            classes[undecided] = pack_bits(self.fallback.predict(dets[undecided]))
        classes[classes < 0] = 0
        return unpack_bits(classes, self.num_observables)


class TableDecoder(Decoder):
    """A decoder given by its prediction for every one of the ``2^D`` syndromes.

    :meth:`from_decoder` freezes a slower decoder into a table, so that it can be queried
    repeatedly over all syndromes (e.g. as the fallback of many lookup tables).

    Attributes:
        table: Predicted class (packed observable flips) for each packed syndrome.
    """

    def __init__(self, table: np.ndarray, num_detectors: int, num_observables: int) -> None:
        if table.shape != (2**num_detectors,):
            raise ValueError(f"table must have shape ({2**num_detectors},), got {table.shape}")
        self.table = table
        self.num_detectors = num_detectors
        self.num_observables = num_observables

    @classmethod
    def from_decoder(
        cls, decoder: Decoder, *, max_detectors: int = 26, chunk_size: int = 1 << 20
    ) -> TableDecoder:
        num_det = decoder.num_detectors
        if num_det > max_detectors:
            raise ValueError(f"{num_det} detectors exceed max_detectors={max_detectors}")
        table = np.empty(2**num_det, dtype=np.int64)
        for start in range(0, 2**num_det, chunk_size):
            syndromes = np.arange(start, min(start + chunk_size, 2**num_det))
            table[syndromes] = pack_bits(decoder.predict(unpack_bits(syndromes, num_det)))
        return cls(table, num_det, decoder.num_observables)

    def predict(self, detection_events: np.ndarray) -> np.ndarray:
        dets = self._check_detection_events(detection_events)
        return unpack_bits(self.table[pack_bits(dets)], self.num_observables)
