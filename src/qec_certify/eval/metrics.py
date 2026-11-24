"""Logical error rates with Wilson confidence intervals, and gap metrics."""

from __future__ import annotations

import math
import time
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


@dataclass(frozen=True)
class PerformanceResult:
    """Offline batch-of-one prediction timings; these are not per-round streaming timings."""

    mode: str
    shots: int
    warmup_shots: int
    total_seconds: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_shots_per_second: float

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def benchmark_decoder(
    decoder: Decoder,
    detection_events: np.ndarray,
    *,
    shots: int,
    warmup_shots: int = 1,
) -> PerformanceResult:
    """Time individual predictions on the first ``shots`` detection-event rows.

    Warmup calls reuse the first row and are excluded. Python call overhead is included, so
    the result describes this offline API rather than a hardware streaming implementation.
    """
    dets = np.asarray(detection_events)
    if dets.ndim != 2:
        raise ValueError(f"detection_events must be 2-D, got {dets.shape}")
    if not 0 < shots <= len(dets):
        raise ValueError(f"latency shots must be in [1, {len(dets)}], got {shots}")
    if warmup_shots < 0:
        raise ValueError(f"warmup_shots must be non-negative, got {warmup_shots}")
    for _ in range(warmup_shots):
        decoder.predict(dets[:1])
    latencies_ms = np.empty(shots, dtype=np.float64)
    start = time.perf_counter()
    for k in range(shots):
        shot_start = time.perf_counter()
        decoder.predict(dets[k : k + 1])
        latencies_ms[k] = (time.perf_counter() - shot_start) * 1e3
    total_seconds = time.perf_counter() - start
    p50, p95, p99 = np.percentile(latencies_ms, [50, 95, 99])
    return PerformanceResult(
        mode="offline_batch1",
        shots=shots,
        warmup_shots=warmup_shots,
        total_seconds=total_seconds,
        mean_latency_ms=float(latencies_ms.mean()),
        p50_latency_ms=float(p50),
        p95_latency_ms=float(p95),
        p99_latency_ms=float(p99),
        throughput_shots_per_second=shots / total_seconds,
    )


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
