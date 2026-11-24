"""Per-round latency and arrival-backlog measurements for causal decoders."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from qec_certify.decoders.streaming import StreamingDecoder
from qec_certify.eval.metrics import LERResult, logical_error_rate
from qec_certify.sim import SyndromeData


@dataclass(frozen=True)
class StreamingPerformanceResult:
    """Measured decode time and simulated fixed-rate arrivals using those measurements."""

    mode: str
    shots: int
    rounds_per_shot: int
    rounds: int
    warmup_shots: int
    total_seconds: float
    mean_round_latency_ms: float
    p50_round_latency_ms: float
    p95_round_latency_ms: float
    p99_round_latency_ms: float
    throughput_rounds_per_second: float
    throughput_shots_per_second: float
    round_period_us: float | None
    meets_mean_input_rate: bool | None
    estimated_backlog_growth_rounds_per_round: float | None
    max_pending_rounds: int | None

    def to_dict(self) -> dict[str, float | int | bool | str | None]:
        return asdict(self)


def streaming_predictions(decoder: StreamingDecoder, data: SyndromeData) -> np.ndarray:
    """Final causal predictions for complete sampled shots."""
    expected = decoder.rounds * decoder.num_detectors_per_round
    if data.detection_events.shape[1] != expected:
        raise ValueError(
            f"expected {expected} detectors per shot, got {data.detection_events.shape[1]}"
        )
    predictions = np.empty((data.shots, decoder.num_observables), dtype=bool)
    for k, shot in enumerate(data.detection_events):
        decoder.reset()
        for events in shot.reshape(decoder.rounds, decoder.num_detectors_per_round):
            decoder.decode_round(events)
        predictions[k] = decoder.finalize()
    return predictions


def evaluate_streaming_decoder(
    decoder: StreamingDecoder, data: SyndromeData, confidence: float = 0.95
) -> LERResult:
    """Final-prediction LER of a causal decoder on complete sampled shots."""
    return logical_error_rate(
        streaming_predictions(decoder, data), data.observable_flips, confidence
    )


def benchmark_streaming_decoder(
    decoder: StreamingDecoder,
    detection_events: np.ndarray,
    *,
    shots: int,
    warmup_shots: int = 1,
    round_period_us: float | None = None,
) -> StreamingPerformanceResult:
    """Time causal rounds and replay their latencies against fixed-rate arrivals.

    ``max_pending_rounds`` includes the round being decoded and rounds still queued. Backlog
    growth is the mean decode time divided by the arrival period minus one, clipped at zero;
    it is a rate estimate, while pending rounds also reflect measured latency variation.
    """
    dets = np.asarray(detection_events)
    if dets.ndim != 2:
        raise ValueError(f"detection_events must be 2-D, got {dets.shape}")
    expected = decoder.rounds * decoder.num_detectors_per_round
    if dets.shape[1] != expected:
        raise ValueError(f"expected {expected} detectors per shot, got {dets.shape[1]}")
    if not 0 < shots <= len(dets):
        raise ValueError(f"streaming shots must be in [1, {len(dets)}], got {shots}")
    if warmup_shots < 0:
        raise ValueError(f"warmup_shots must be non-negative, got {warmup_shots}")
    if round_period_us is not None and round_period_us <= 0:
        raise ValueError(f"round_period_us must be positive, got {round_period_us}")

    for _ in range(warmup_shots):
        decoder.reset()
        for events in dets[0].reshape(decoder.rounds, decoder.num_detectors_per_round):
            decoder.decode_round(events)

    latencies_seconds = np.empty(shots * decoder.rounds, dtype=np.float64)
    completions: list[float] = []
    max_pending: int | None = None
    period = round_period_us * 1e-6 if round_period_us is not None else None
    index = 0
    for shot in range(shots):
        decoder.reset()
        rounds = dets[shot].reshape(decoder.rounds, decoder.num_detectors_per_round)
        for events in rounds:
            start = time.perf_counter()
            decoder.decode_round(events)
            latency = time.perf_counter() - start
            latencies_seconds[index] = latency
            if period is not None:
                arrival = index * period
                previous = completions[-1] if completions else 0.0
                pending = sum(completion > arrival for completion in completions) + 1
                max_pending = max(max_pending or 0, pending)
                completions.append(max(arrival, previous) + latency)
            index += 1
    total_seconds = float(latencies_seconds.sum())
    p50, p95, p99 = np.percentile(latencies_seconds, [50, 95, 99])
    mean_seconds = float(latencies_seconds.mean())
    return StreamingPerformanceResult(
        mode=decoder.mode,
        shots=shots,
        rounds_per_shot=decoder.rounds,
        rounds=len(latencies_seconds),
        warmup_shots=warmup_shots,
        total_seconds=total_seconds,
        mean_round_latency_ms=mean_seconds * 1e3,
        p50_round_latency_ms=float(p50) * 1e3,
        p95_round_latency_ms=float(p95) * 1e3,
        p99_round_latency_ms=float(p99) * 1e3,
        throughput_rounds_per_second=len(latencies_seconds) / total_seconds,
        throughput_shots_per_second=shots / total_seconds,
        round_period_us=round_period_us,
        meets_mean_input_rate=mean_seconds <= period if period is not None else None,
        estimated_backlog_growth_rounds_per_round=(
            max(0.0, mean_seconds / period - 1.0) if period is not None else None
        ),
        max_pending_rounds=max_pending,
    )
