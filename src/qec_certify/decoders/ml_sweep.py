"""Exact maximum-likelihood decoding by sweeping the detector error model.

:class:`~qec_certify.decoders.ExactMLDecoder` tabulates ``P(s, l)`` for all ``2^D``
syndromes, which stops at ``D + L`` of about 26 bits (d=5 code capacity). Decoding a
*given* syndrome only needs ``P(s, l)`` for that ``s``, and that can be computed exactly
with far less memory:

Process the error mechanisms one at a time and keep the joint distribution of only the
*frontier*: the detectors touched by some processed and some unprocessed mechanism, plus
the observables. A detector enters the frontier just before its first mechanism. Right
after its last mechanism its value can no longer change, so the distribution is
conditioned on the observed value and the detector leaves the frontier. Every update is
``P <- (1 - p_i) P + p_i P[· XOR mask_i]`` on non-negative numbers, so there is no
cancellation; the result is exact up to float64 rounding, like the full table.

The cost is ``O(M 2^(w + L))`` per shot for frontier width ``w``. This is a contraction of
the DEM's tensor network along a sweep. For code-capacity noise on a 2-D code a sweep
along one coordinate keeps ``w`` near ``d``, so d=7 to 11 are cheap. The mechanism order
is chosen among a few candidate sweeps to minimise ``w``.
"""

from __future__ import annotations

import numpy as np
import stim

from qec_certify.decoders.base import Decoder
from qec_certify.decoders.ml_exact import TIE_RTOL, error_mechanisms
from qec_certify.utils.bits import unpack_bits

DEFAULT_MAX_FRONTIER = 20
# Shots are decoded in chunks of about this many float64 numbers (256 KiB), small enough
# to stay in a core's cache, so that parallel processes do not compete for memory bandwidth.
DEFAULT_CHUNK_ELEMENTS = 1 << 15


def frontier_width(masks: list[int], order: list[int], num_detectors: int) -> int:
    """Largest number of detectors open at once when mechanisms are processed in ``order``."""
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for t, i in enumerate(order):
        for j in _detectors(masks[i], num_detectors):
            first.setdefault(j, t)
            last[j] = t
    change = np.zeros(len(order) + 1, dtype=np.int64)
    for j, t in first.items():
        change[t] += 1
        change[last[j] + 1] -= 1
    return int(np.cumsum(change).max(initial=0))


def candidate_orders(dem: stim.DetectorErrorModel, masks: list[int]) -> list[list[int]]:
    """The DEM's own order, plus sweeps along the detector coordinate axes.

    Single-axis sweeps sort mechanisms by the smallest, mean or largest coordinate of the
    detectors they flip. Two-axis sweeps sort by the rounded mean along one axis, then the
    mean along another: for a 2-D code that is column by column, top to bottom within a
    column, which keeps the frontier near ``d``. Mechanisms that flip no detector go first.
    """
    num_det = dem.num_detectors
    orders = [list(range(len(masks)))]
    coords = dem.get_detector_coordinates()
    if num_det == 0 or len(coords) < num_det or not all(coords.values()):
        return orders
    num_axes = min(len(c) for c in coords.values())
    values = [
        np.array([coords[j][:num_axes] for j in _detectors(m, num_det)]).reshape(-1, num_axes)
        for m in masks
    ]

    def sweep(key) -> list[int]:
        keys = [key(v) if len(v) else (-np.inf,) for v in values]
        return sorted(range(len(masks)), key=lambda i: (keys[i], i))

    for a in range(num_axes):
        for reduce in (np.min, np.mean, np.max):
            orders.append(sweep(lambda v, a=a, r=reduce: (float(r(v[:, a])),)))
        for b in range(num_axes):
            if b != a:
                orders.append(
                    sweep(lambda v, a=a, b=b: (round(v[:, a].mean()), float(v[:, b].mean())))
                )
    return orders


class SweepMLDecoder(Decoder):
    """Exact maximum-likelihood decoder that contracts the DEM for each shot.

    Predicts ``argmax_l P(s, l)`` like :class:`~qec_certify.decoders.ExactMLDecoder`, with
    the same tie rule (classes within ``TIE_RTOL`` of the maximum are tied and the smallest
    index wins, so impossible syndromes decode to no flip), but without tabulating all
    syndromes. The two decoders therefore make the same decision on every shot.

    Attributes:
        order: Mechanism indices (into :func:`error_mechanisms`) in processing order.
        frontier: Largest number of detectors held in the distribution at once.
    """

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        *,
        max_frontier: int = DEFAULT_MAX_FRONTIER,
        chunk_elements: int = DEFAULT_CHUNK_ELEMENTS,
    ) -> None:
        self.num_detectors = dem.num_detectors
        self.num_observables = dem.num_observables
        self.chunk_elements = chunk_elements
        self._probs, self._masks = error_mechanisms(dem)
        widths = [
            (frontier_width(self._masks, order, self.num_detectors), k, order)
            for k, order in enumerate(candidate_orders(dem, self._masks))
        ]
        self.frontier, _, self.order = min(widths)
        if self.frontier > max_frontier:
            raise ValueError(
                f"frontier of {self.frontier} detectors exceeds max_frontier={max_frontier}"
            )
        self._plan = self._make_plan()

    @classmethod
    def from_circuit(cls, circuit: stim.Circuit, **kwargs) -> SweepMLDecoder:
        return cls(circuit.detector_error_model(), **kwargs)

    def _make_plan(self) -> list[tuple[float, list[int], list[int], list[int]]]:
        """Per step: ``(p, bits flipped, detectors to open before, detectors to close after)``."""
        num_det = self.num_detectors
        last: dict[int, int] = {}
        for t, i in enumerate(self.order):
            for j in _detectors(self._masks[i], num_det):
                last[j] = t
        self._untouched = [j for j in range(num_det) if j not in last]
        seen: set[int] = set()
        plan = []
        for t, i in enumerate(self.order):
            dets = _detectors(self._masks[i], num_det)
            bits = [b for b in range(num_det + self.num_observables) if (self._masks[i] >> b) & 1]
            opened = [j for j in dets if j not in seen]
            seen.update(opened)
            closed = [j for j in dets if last[j] == t]
            plan.append((float(self._probs[i]), bits, opened, closed))
        return plan

    def log_joint(self, detection_events: np.ndarray) -> np.ndarray:
        """``log P(s, l)`` for every shot and logical class, shape ``(shots, 2^L)``.

        Class ``c`` is the packed observable flips (observable ``j`` is bit ``j``).
        Impossible pairs give ``-inf``.
        """
        dets = self._check_detection_events(detection_events)
        per_shot = 2 ** (self.frontier + self.num_observables)
        chunk = max(1, self.chunk_elements // per_shot)
        out = np.empty((len(dets), 2**self.num_observables))
        for start in range(0, len(dets), chunk):
            out[start : start + chunk] = self._log_joint_chunk(dets[start : start + chunk])
        return out

    def _log_joint_chunk(self, dets: np.ndarray) -> np.ndarray:
        shots = len(dets)
        num_det, num_obs = self.num_detectors, self.num_observables
        # Axis k of ``dist`` holds bit ``active[k]``; the last axis is the shot. Observables
        # come first, highest bit first, so the final C-order flattening is the packed class
        # index. Keeping shots innermost leaves the flipped axes outer, so every update runs
        # over contiguous memory.
        active = [num_det + k for k in reversed(range(num_obs))]
        dist = np.zeros((2,) * num_obs + (shots,))
        dist[(0,) * num_obs] = 1.0
        log_norm = np.zeros(shots)
        possible = ~dets[:, self._untouched].any(axis=1)
        with np.errstate(divide="ignore"):
            log_norm[~possible] = -np.inf

        for p, bits, opened, closed in self._plan:
            for j in opened:
                dist = np.stack([dist, np.zeros_like(dist)], axis=-2)
                active.append(j)
            flipped = np.flip(dist, axis=tuple(active.index(b) for b in bits)) * p
            dist *= 1.0 - p
            dist += flipped
            for j in closed:
                halves = np.moveaxis(dist, active.index(j), 0)
                dist = np.where(dets[:, j], halves[1], halves[0])
                active.remove(j)
            if closed:
                total = dist.reshape(-1, shots).sum(axis=0)
                dist /= np.where(total > 0, total, 1.0)
                with np.errstate(divide="ignore"):
                    log_norm += np.log(total)

        flat = dist.reshape(2**num_obs, shots).T
        with np.errstate(divide="ignore"):
            return np.log(flat) + log_norm[:, None]

    def predict_proba(self, detection_events: np.ndarray) -> np.ndarray:
        """Posterior ``P(l | s)`` over logical classes; impossible syndromes give no-flip."""
        log_joint = self.log_joint(detection_events)
        peak = log_joint.max(axis=1, keepdims=True)
        finite = np.isfinite(peak[:, 0])
        weights = np.zeros_like(log_joint)
        weights[finite] = np.exp(log_joint[finite] - peak[finite])
        weights[~finite, 0] = 1.0
        return weights / weights.sum(axis=1, keepdims=True)

    def predict(self, detection_events: np.ndarray) -> np.ndarray:
        log_joint = self.log_joint(detection_events)
        best = log_joint.max(axis=1, keepdims=True)
        classes = np.argmax(log_joint >= best + np.log1p(-TIE_RTOL), axis=1)
        return unpack_bits(classes, self.num_observables)


def _detectors(mask: int, num_detectors: int) -> list[int]:
    return [j for j in range(num_detectors) if (mask >> j) & 1]
