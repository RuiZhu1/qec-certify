"""Logical error rates with Wilson confidence intervals, and gap metrics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import NormalDist

import numpy as np

from qec_certify.decoders.base import Decoder
from qec_certify.sim import SyndromeData


@dataclass(frozen=True)
class LERResult:
    """Observed logical error rate with a two-sided Wilson score interval."""

    num_errors: int
    shots: int
    confidence: float
    ci_low: float
    ci_high: float

    @property
    def ler(self) -> float:
        return self.num_errors / self.shots

    def to_dict(self) -> dict[str, float | int]:
        return {**asdict(self), "ler": self.ler}


def wilson_interval(num_errors: int, shots: int, confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion ``num_errors / shots``."""
    if shots <= 0:
        raise ValueError(f"shots must be positive, got {shots}")
    if not 0 <= num_errors <= shots:
        raise ValueError(f"num_errors must be in [0, {shots}], got {num_errors}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    z2 = z * z
    p_hat = num_errors / shots
    denom = 1.0 + z2 / shots
    center = (p_hat + z2 / (2.0 * shots)) / denom
    half = z * math.sqrt(p_hat * (1.0 - p_hat) / shots + z2 / (4.0 * shots * shots)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def logical_error_rate(
    predictions: np.ndarray, observable_flips: np.ndarray, confidence: float = 0.95
) -> LERResult:
    """LER of predicted observable flips; a shot fails if any observable is mispredicted."""
    predictions = np.asarray(predictions, dtype=bool)
    observable_flips = np.asarray(observable_flips, dtype=bool)
    if predictions.shape != observable_flips.shape or predictions.ndim != 2:
        raise ValueError(
            f"predictions {predictions.shape} and observable flips {observable_flips.shape} "
            "must have the same 2-D shape"
        )
    shots = len(predictions)
    num_errors = int(np.any(predictions != observable_flips, axis=1).sum())
    low, high = wilson_interval(num_errors, shots, confidence)
    return LERResult(num_errors, shots, confidence, low, high)


def evaluate_decoder(decoder: Decoder, data: SyndromeData, confidence: float = 0.95) -> LERResult:
    """LER of ``decoder`` on sampled shots."""
    predictions = decoder.predict(data.detection_events)
    return logical_error_rate(predictions, data.observable_flips, confidence)


def gap_ratio(ler: float, reference_ler: float) -> float:
    """``G = ler / reference_ler``, e.g. ``LER_learned / LER_ML``."""
    if reference_ler <= 0:
        raise ValueError("gap ratio is undefined for a zero reference LER")
    return ler / reference_ler


def gap_closed(ler_mwpm: float, ler_learned: float, ler_ml: float) -> float:
    """Fraction of the MWPM-to-ML gap closed: ``(LER_MWPM - LER_learned) / (LER_MWPM - LER_ML)``."""
    if ler_mwpm == ler_ml:
        raise ValueError("gap closed is undefined when LER_MWPM == LER_ML")
    return (ler_mwpm - ler_learned) / (ler_mwpm - ler_ml)
