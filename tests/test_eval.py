import numpy as np
import pytest
from scipy.stats import binomtest

from qec_certify.codes import code_capacity_circuit, repetition_code
from qec_certify.decoders import Decoder
from qec_certify.eval import (
    evaluate_decoder,
    exact_logical_error_rate,
    gap_closed,
    gap_ratio,
    logical_error_rate,
    wilson_interval,
)
from qec_certify.sim import SyndromeData


def test_wilson_interval_reference_value():
    low, high = wilson_interval(5, 10, 0.95)
    assert low == pytest.approx(0.2366, abs=1e-4)
    assert high == pytest.approx(0.7634, abs=1e-4)


@pytest.mark.parametrize(("k", "n"), [(0, 10), (1, 10), (3, 50), (17, 1000), (999, 1000)])
@pytest.mark.parametrize("confidence", [0.9, 0.95, 0.999])
def test_wilson_interval_matches_scipy(k, n, confidence):
    ci = binomtest(k, n).proportion_ci(confidence_level=confidence, method="wilson")
    low, high = wilson_interval(k, n, confidence)
    assert low == pytest.approx(ci.low, abs=1e-12)
    assert high == pytest.approx(ci.high, abs=1e-12)


def test_wilson_interval_edges_and_symmetry():
    assert wilson_interval(0, 100)[0] == 0.0
    assert wilson_interval(100, 100)[1] == 1.0
    low, high = wilson_interval(30, 100)
    mirror_low, mirror_high = wilson_interval(70, 100)
    assert low == pytest.approx(1 - mirror_high)
    assert high == pytest.approx(1 - mirror_low)
    wide = wilson_interval(30, 100, 0.99)
    assert wide[0] < low and wide[1] > high


@pytest.mark.parametrize(
    ("k", "n", "confidence"), [(1, 0, 0.95), (-1, 10, 0.95), (11, 10, 0.95), (1, 10, 1.0)]
)
def test_wilson_interval_rejects_invalid_input(k, n, confidence):
    with pytest.raises(ValueError):
        wilson_interval(k, n, confidence)


def test_logical_error_rate_counts_any_mispredicted_observable():
    predictions = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=bool)
    truth = np.array([[0, 0], [1, 1], [0, 1], [0, 0]], dtype=bool)
    result = logical_error_rate(predictions, truth)
    assert (result.num_errors, result.shots) == (2, 4)
    assert result.ler == 0.5
    assert (result.ci_low, result.ci_high) == wilson_interval(2, 4)
    assert result.to_dict()["ler"] == 0.5
    with pytest.raises(ValueError):
        logical_error_rate(predictions, truth[:3])


class _AlwaysNoFlip(Decoder):
    def __init__(self, num_detectors: int) -> None:
        self.num_detectors = num_detectors
        self.num_observables = 1

    def predict(self, detection_events):
        dets = self._check_detection_events(detection_events)
        return np.zeros((len(dets), 1), dtype=bool)


def test_evaluate_decoder():
    data = SyndromeData(np.zeros((5, 2), bool), np.array([[0], [1], [0], [1], [1]], bool))
    result = evaluate_decoder(_AlwaysNoFlip(2), data)
    assert (result.num_errors, result.shots) == (3, 5)


def test_exact_logical_error_rate_trivial_decoder():
    """Never correcting fails iff qubit 0 (the logical Z) suffers an X or Y error: 2p/3."""
    p = 0.1
    dem = code_capacity_circuit(repetition_code(3), p).detector_error_model()
    assert exact_logical_error_rate(_AlwaysNoFlip(2), dem) == pytest.approx(2 * p / 3)


def test_gap_metrics():
    assert gap_ratio(0.02, 0.01) == pytest.approx(2.0)
    assert gap_closed(ler_mwpm=0.03, ler_learned=0.02, ler_ml=0.01) == pytest.approx(0.5)
    assert gap_closed(ler_mwpm=0.03, ler_learned=0.03, ler_ml=0.01) == 0.0
    assert gap_closed(ler_mwpm=0.03, ler_learned=0.01, ler_ml=0.01) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        gap_ratio(0.1, 0.0)
    with pytest.raises(ValueError):
        gap_closed(0.02, 0.01, 0.02)
