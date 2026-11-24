import numpy as np
import pytest

from qec_certify.codes import circuit_level_circuit, rotated_surface_code
from qec_certify.decoders import (
    DelayedCommitMWPMDecoder,
    MWPMDecoder,
    SlidingWindowMWPMDecoder,
)
from qec_certify.eval import benchmark_streaming_decoder, evaluate_streaming_decoder
from qec_certify.sim import sample_syndromes


def window_setup(distance=3, rounds=3, shots=20):
    circuit = circuit_level_circuit(rotated_surface_code(distance), 0.01, rounds=rounds)
    data = sample_syndromes(circuit, shots, 23)
    return circuit, data


@pytest.mark.parametrize("correlated", [False, True])
@pytest.mark.parametrize("window_class", [SlidingWindowMWPMDecoder, DelayedCommitMWPMDecoder])
def test_full_window_matches_offline(window_class, correlated):
    circuit, data = window_setup()
    offline = MWPMDecoder.from_circuit(circuit, correlated=correlated)
    streaming = window_class.from_circuit(circuit, rounds=3, window_rounds=3, correlated=correlated)
    assert streaming.approximation == "exact_full_history"
    expected = offline.predict(data.detection_events)
    actual = []
    for shot in data.detection_events:
        streaming.reset()
        for events in shot.reshape(3, streaming.num_detectors_per_round):
            streaming.decode_round(events)
        actual.append(streaming.finalize())
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize("window_class", [SlidingWindowMWPMDecoder, DelayedCommitMWPMDecoder])
def test_partial_window_is_causal_and_evaluates_final_predictions(window_class):
    circuit, data = window_setup(distance=5, rounds=5, shots=10)
    first = window_class.from_circuit(circuit, rounds=5, window_rounds=2)
    second = window_class.from_circuit(circuit, rounds=5, window_rounds=2)
    assert first.approximation in {"projected_dem_boundary", "delayed_commit_boundary"}
    shot = data.detection_events[0].reshape(5, first.num_detectors_per_round)
    changed = shot.copy()
    changed[-1] ^= True
    for round_index in range(4):
        np.testing.assert_array_equal(
            first.decode_round(shot[round_index]), second.decode_round(shot[round_index])
        )
    first.decode_round(shot[-1])
    second.decode_round(changed[-1])
    result = evaluate_streaming_decoder(first, data)
    assert result.shots == 10
    assert 0 <= result.num_errors <= 10


def test_window_performance_mode_and_backlog_fields():
    circuit, data = window_setup(distance=5, rounds=5, shots=3)
    streaming = SlidingWindowMWPMDecoder.from_circuit(circuit, rounds=5, window_rounds=2)
    result = benchmark_streaming_decoder(
        streaming, data.detection_events, shots=2, warmup_shots=1, round_period_us=1000
    )
    assert result.mode == "sliding_window_mwpm_w2"
    assert result.rounds == 10
    assert result.round_period_us == 1000
    assert result.max_pending_rounds >= 1


def test_delayed_commit_mode():
    circuit, _ = window_setup()
    streaming = DelayedCommitMWPMDecoder.from_circuit(circuit, rounds=3, window_rounds=2)
    assert streaming.mode == "delayed_commit_mwpm_w2"
    assert streaming.approximation == "delayed_commit_boundary"


def test_window_validates_arguments_and_shape():
    circuit, _ = window_setup()
    with pytest.raises(ValueError, match="window_rounds"):
        SlidingWindowMWPMDecoder.from_circuit(circuit, rounds=3, window_rounds=0)
    with pytest.raises(ValueError, match="divisible"):
        SlidingWindowMWPMDecoder.from_circuit(circuit, rounds=5, window_rounds=2)
    streaming = SlidingWindowMWPMDecoder.from_circuit(circuit, rounds=3, window_rounds=2)
    with pytest.raises(ValueError, match="expected one round"):
        streaming.decode_round(np.zeros(2, dtype=bool))
    with pytest.raises(ValueError, match="expected 3 rounds"):
        streaming.finalize()
