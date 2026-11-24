"""Per-shot profiling for Relay-BP latency, iterations and implementation overhead."""

from __future__ import annotations

import time

import numpy as np

from qec_certify.decoders.relay import RelayBPDecoder


def _summary(values: np.ndarray) -> dict[str, float] | None:
    if not len(values):
        return None
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def profile_relay_decoder(
    decoder: RelayBPDecoder,
    detection_events: np.ndarray,
    *,
    shots: int,
    warmup_shots: int = 1,
) -> dict:
    """Profile independent batch-of-one predictions, separating successes and failures.

    Initialization and native decode timings are available for the fresh serial path. Parallel
    initialization is reported as unavailable because runner clone setup is not independently
    exposed by the public API.
    """
    dets = np.asarray(detection_events)
    if dets.ndim != 2:
        raise ValueError(f"detection_events must be 2-D, got {dets.shape}")
    if not 0 < shots <= len(dets):
        raise ValueError(f"profile shots must be in [1, {len(dets)}], got {shots}")
    if warmup_shots < 0:
        raise ValueError(f"warmup_shots must be non-negative, got {warmup_shots}")
    for _ in range(warmup_shots):
        decoder.predict(dets[:1])

    latencies = np.empty(shots)
    iterations = np.empty(shots, dtype=np.int64)
    successes = np.empty(shots, dtype=bool)
    initialization = np.empty(shots)
    native_decode = np.empty(shots)
    overhead = np.empty(shots)
    for k in range(shots):
        start = time.perf_counter()
        decoder.predict(dets[k : k + 1])
        latencies[k] = time.perf_counter() - start
        iterations[k] = decoder.last_decode_iterations[0]
        successes[k] = decoder.last_decode_success[0]
        initialization[k] = decoder.last_initialization_seconds[0]
        native_decode[k] = decoder.last_native_decode_seconds[0]
        overhead[k] = max(0.0, latencies[k] - initialization[k] - native_decode[k])

    return {
        "mode": "parallel" if decoder.parallel else "fresh_serial",
        "shots": shots,
        "warmup_shots": warmup_shots,
        "successes": int(successes.sum()),
        "failures": int((~successes).sum()),
        "decode_success_rate": float(successes.mean()),
        "throughput_shots_per_second": shots / float(latencies.sum()),
        "iterations": _summary(iterations.astype(float)),
        "success_iterations": _summary(iterations[successes].astype(float)),
        "failure_iterations": _summary(iterations[~successes].astype(float)),
        "latency_ms": _summary(latencies * 1e3),
        "success_latency_ms": _summary(latencies[successes] * 1e3),
        "failure_latency_ms": _summary(latencies[~successes] * 1e3),
        "initialization_ms": None if decoder.parallel else _summary(initialization * 1e3),
        "native_decode_ms": _summary(native_decode * 1e3),
        "python_overhead_ms": _summary(overhead * 1e3),
    }
