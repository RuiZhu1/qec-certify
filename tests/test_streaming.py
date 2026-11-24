import numpy as np
import pytest

from qec_certify.codes import circuit_level_circuit, rotated_surface_code
from qec_certify.decoders import (
    CausalReplayDecoder,
    MWPMDecoder,
    UnionFindDecoder,
)
from qec_certify.eval import benchmark_streaming_decoder
from qec_certify.sim import sample_syndromes


def streaming_setup(rounds=3, shots=10):
    circuit = circuit_level_circuit(rotated_surface_code(3), 0.01, rounds=rounds)
    data = sample_syndromes(circuit, shots, 17)
    return circuit, data


def test_causal_replay_final_prediction_matches_offline():
    circuit, data = streaming_setup()
    offline = MWPMDecoder.from_circuit(circuit)
    streaming = CausalReplayDecoder(MWPMDecoder.from_circuit(circuit), rounds=3)
    expected = offline.predict(data.detection_events)
    for shot, expected_shot in zip(data.detection_events, expected, strict=True):
        streaming.reset()
        for events in shot.reshape(3, streaming.num_detectors_per_round):
            streaming.decode_round(events)
        np.testing.assert_array_equal(streaming.finalize(), expected_shot)


def test_causal_replay_does_not_use_future_rounds():
    circuit, data = streaming_setup()
    first = CausalReplayDecoder(MWPMDecoder.from_circuit(circuit), rounds=3)
    second = CausalReplayDecoder(MWPMDecoder.from_circuit(circuit), rounds=3)
    shot = data.detection_events[0].reshape(3, first.num_detectors_per_round)
    changed = shot.copy()
    changed[-1] ^= True
    for round_index in range(2):
        np.testing.assert_array_equal(
            first.decode_round(shot[round_index]), second.decode_round(shot[round_index])
        )
    first.decode_round(shot[-1])
    second.decode_round(changed[-1])


def test_causal_replay_validates_state_and_input():
    circuit, _ = streaming_setup()
    streaming = CausalReplayDecoder(MWPMDecoder.from_circuit(circuit), rounds=3)
    with pytest.raises(ValueError, match="expected 3 rounds"):
        streaming.finalize()
    with pytest.raises(ValueError, match="expected one round"):
        streaming.decode_round(np.zeros(2, dtype=bool))
    for _ in range(3):
        streaming.decode_round(np.zeros(streaming.num_detectors_per_round, dtype=bool))
    with pytest.raises(ValueError, match="call reset"):
        streaming.decode_round(np.zeros(streaming.num_detectors_per_round, dtype=bool))
    with pytest.raises(ValueError, match="divisible"):
        CausalReplayDecoder(MWPMDecoder.from_circuit(circuit), rounds=5)
    with pytest.raises(ValueError, match="inconsistent causal prefixes"):
        CausalReplayDecoder(UnionFindDecoder.from_circuit(circuit), rounds=3)


def test_streaming_performance_with_and_without_arrival_rate():
    circuit, data = streaming_setup(shots=3)
    streaming = CausalReplayDecoder(MWPMDecoder.from_circuit(circuit), rounds=3)
    result = benchmark_streaming_decoder(
        streaming, data.detection_events, shots=2, warmup_shots=1, round_period_us=1e9
    )
    assert result.mode == "causal_replay_batch1"
    assert result.rounds == 6
    assert result.rounds_per_shot == 3
    assert result.meets_mean_input_rate
    assert result.estimated_backlog_growth_rounds_per_round == 0
    assert result.max_pending_rounds == 1
    assert result.p50_round_latency_ms <= result.p95_round_latency_ms
    assert result.p95_round_latency_ms <= result.p99_round_latency_ms
    assert result.total_seconds == pytest.approx(result.mean_round_latency_ms * result.rounds / 1e3)

    without_rate = benchmark_streaming_decoder(
        streaming, data.detection_events, shots=2, warmup_shots=0
    )
    assert without_rate.rounds == 6
    assert without_rate.max_pending_rounds is None
    assert without_rate.meets_mean_input_rate is None
    assert without_rate.total_seconds > 0


def test_streaming_performance_reports_backlog_growth():
    circuit, data = streaming_setup(shots=2)
    streaming = CausalReplayDecoder(MWPMDecoder.from_circuit(circuit), rounds=3)
    result = benchmark_streaming_decoder(
        streaming, data.detection_events, shots=2, warmup_shots=0, round_period_us=1e-9
    )
    assert not result.meets_mean_input_rate
    assert result.estimated_backlog_growth_rounds_per_round > 0
    assert result.max_pending_rounds > 1


def test_streaming_performance_rejects_invalid_input():
    circuit, data = streaming_setup(shots=2)
    streaming = CausalReplayDecoder(MWPMDecoder.from_circuit(circuit), rounds=3)
    with pytest.raises(ValueError, match="streaming shots"):
        benchmark_streaming_decoder(streaming, data.detection_events, shots=3)
    with pytest.raises(ValueError, match="round_period_us"):
        benchmark_streaming_decoder(streaming, data.detection_events, shots=1, round_period_us=0)
    with pytest.raises(ValueError, match="warmup_shots"):
        benchmark_streaming_decoder(streaming, data.detection_events, shots=1, warmup_shots=-1)
    with pytest.raises(ValueError, match="detectors per shot"):
        benchmark_streaming_decoder(streaming, data.detection_events[:, :2], shots=1)
