"""Evaluation: logical error rates, confidence intervals and gap metrics."""

from qec_certify.eval.exact import exact_logical_error_rate
from qec_certify.eval.metrics import (
    LERResult,
    evaluate_decoder,
    gap_closed,
    gap_ratio,
    logical_error_rate,
    wilson_interval,
)

__all__ = [
    "LERResult",
    "evaluate_decoder",
    "exact_logical_error_rate",
    "gap_closed",
    "gap_ratio",
    "logical_error_rate",
    "wilson_interval",
]
