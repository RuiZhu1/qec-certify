"""Evaluation: logical error rates, confidence intervals and gap metrics."""

from qec_certify.eval.exact import exact_logical_error_rate
from qec_certify.eval.metrics import (
    LERResult,
    PerformanceResult,
    benchmark_decoder,
    evaluate_decoder,
    gap_closed,
    gap_ratio,
    logical_error_rate,
    wilson_interval,
)
from qec_certify.eval.streaming import (
    StreamingPerformanceResult,
    benchmark_streaming_decoder,
    evaluate_streaming_decoder,
    streaming_predictions,
)

__all__ = [
    "LERResult",
    "PerformanceResult",
    "StreamingPerformanceResult",
    "benchmark_decoder",
    "benchmark_streaming_decoder",
    "evaluate_decoder",
    "evaluate_streaming_decoder",
    "exact_logical_error_rate",
    "gap_closed",
    "gap_ratio",
    "logical_error_rate",
    "streaming_predictions",
    "wilson_interval",
]
