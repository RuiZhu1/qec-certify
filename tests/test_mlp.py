import numpy as np
import pytest

from qec_certify.codes import code_capacity_circuit, make_code
from qec_certify.decoders import ExactMLDecoder
from qec_certify.decoders.nn import MLPConfig, MLPDecoder
from qec_certify.eval.exact import exact_logical_error_rate
from qec_certify.sim import sample_syndromes
from qec_certify.train.run import run_mlp_experiment

SMALL = MLPConfig(hidden_sizes=(32,), epochs=8, batch_size=256, seed=3)


@pytest.fixture(scope="module")
def d3():
    circuit = code_capacity_circuit(make_code("rotated_surface", 3), 0.1, "Z")
    return circuit, circuit.detector_error_model()


def test_predict_shapes_and_probabilities(d3):
    _, dem = d3
    decoder = MLPDecoder(dem.num_detectors, dem.num_observables, SMALL)
    events = np.zeros((5, dem.num_detectors), dtype=bool)
    assert decoder.predict(events).shape == (5, dem.num_observables)
    assert decoder.predict(events).dtype == bool
    proba = decoder.predict_proba(events)
    assert proba.shape == (5, 2**dem.num_observables)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_fit_requires_data(d3):
    _, dem = d3
    decoder = MLPDecoder(dem.num_detectors, dem.num_observables, SMALL)
    with pytest.raises(ValueError, match="requires training data"):
        decoder.fit()


def test_rejects_mismatched_observables(d3):
    circuit, dem = d3
    data = sample_syndromes(circuit, 100, seed=0)
    decoder = MLPDecoder(dem.num_detectors, dem.num_observables + 1, SMALL)
    with pytest.raises(ValueError, match="observables"):
        decoder.fit(data)


def test_fit_is_deterministic(d3):
    circuit, dem = d3
    data = sample_syndromes(circuit, 2000, seed=1)
    events = np.random.default_rng(0).random((64, dem.num_detectors)) < 0.3
    a = MLPDecoder(dem.num_detectors, dem.num_observables, SMALL).fit(data)
    b = MLPDecoder(dem.num_detectors, dem.num_observables, SMALL).fit(data)
    np.testing.assert_array_equal(a.predict_proba(events), b.predict_proba(events))


def test_fit_does_not_touch_global_rng(d3):
    import torch

    circuit, dem = d3
    data = sample_syndromes(circuit, 500, seed=1)
    torch.manual_seed(7)
    expected = torch.rand(3)
    torch.manual_seed(7)
    MLPDecoder(dem.num_detectors, dem.num_observables, SMALL).fit(data)
    assert torch.equal(torch.rand(3), expected)


def test_validation_selects_best_epoch(d3):
    circuit, dem = d3
    train = sample_syndromes(circuit, 2000, seed=1)
    val = sample_syndromes(circuit, 2000, seed=2)
    decoder = MLPDecoder(dem.num_detectors, dem.num_observables, SMALL).fit(train, val)
    assert [h["epoch"] for h in decoder.history] == list(range(1, SMALL.epochs + 1))
    assert all(h["val_loss"] is not None for h in decoder.history)


def test_min_steps_extends_epochs_for_small_sets(d3):
    circuit, dem = d3
    data = sample_syndromes(circuit, 600, seed=1)  # 3 steps per epoch at batch size 256
    config = MLPConfig(hidden_sizes=(8,), epochs=2, min_steps=30, batch_size=256)
    decoder = MLPDecoder(dem.num_detectors, dem.num_observables, config).fit(data)
    assert len(decoder.history) == 10


def test_learned_decoder_cannot_beat_exact_ml_and_gets_close(d3):
    """Exact ML is optimal, so no MLP can go below it; with enough data it should approach it."""
    circuit, dem = d3
    optimum = ExactMLDecoder.from_circuit(circuit).optimal_logical_error_rate
    train = sample_syndromes(circuit, 50_000, seed=1)
    val = sample_syndromes(circuit, 10_000, seed=2)
    config = MLPConfig(hidden_sizes=(32, 32), epochs=10, batch_size=512, seed=0)
    decoder = MLPDecoder(dem.num_detectors, dem.num_observables, config).fit(train, val)
    exact = exact_logical_error_rate(decoder, dem)
    assert exact >= optimum - 1e-12
    assert exact <= optimum * 1.05


def test_run_mlp_experiment_report():
    config = {
        "name": "test",
        "seed": 1,
        "code": {"family": "rotated_surface", "distance": 3},
        "noise": {"model": "depolarizing", "p": [0.1]},
        "memory_basis": "Z",
        "model": {"type": "mlp", "hidden_sizes": [16], "epochs": 3, "batch_size": 256},
        "train_sizes": [500, 2000],
        "val_shots": 500,
        "test_shots": 1000,
        "confidence": 0.95,
    }
    report = run_mlp_experiment(config)
    (entry,) = report["results"]
    refs = entry["references"]
    assert refs["ml_exact"]["exact_ler"] <= refs["mwpm"]["exact_ler"]
    assert [row["train_shots"] for row in entry["mlp"]] == [500, 2000]
    for row in entry["mlp"]:
        assert row["exact_ler"] >= refs["ml_exact"]["exact_ler"] - 1e-12
        assert row["gap_ratio"] >= 1.0 - 1e-12
        assert row["ci_low"] <= row["ler"] <= row["ci_high"]
        assert 1 <= row["best_epoch"] <= 3
    assert "torch" in report["versions"]


def test_run_mlp_experiment_rejects_unknown_model():
    with pytest.raises(ValueError, match="model type"):
        run_mlp_experiment(
            {
                "seed": 1,
                "code": {"family": "rotated_surface", "distance": 3},
                "noise": {"model": "depolarizing", "p": [0.1]},
                "model": {"type": "transformer"},
                "train_sizes": [10],
                "val_shots": 10,
                "test_shots": 10,
            }
        )
