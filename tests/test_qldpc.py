import json
from pathlib import Path

import numpy as np
import pytest
import stim

from qec_certify.codes import load_reference_circuit
from qec_certify.decoders import BPLSDDecoder, FailureFallbackDecoder, RelayBPDecoder
from qec_certify.eval.profiling import profile_relay_decoder
from qec_certify.eval.qldpc import main, run_qldpc_experiment
from qec_certify.eval.qldpc_pareto import _pareto_front, run_qldpc_pareto
from qec_certify.eval.qldpc_precision_sweep import run_qldpc_precision_sweep
from qec_certify.eval.qldpc_sweep import run_qldpc_sweep
from qec_certify.eval.qldpc_tune import _select_validation_row, run_qldpc_tuning
from qec_certify.eval.qldpc_tune import main as tuning_main
from qec_certify.sim import sample_syndromes
from qec_certify.utils import load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def small_config(**overrides):
    config = {
        "name": "test_bb18",
        "seed": 19,
        "reference_circuit": "bb18_d3_p003",
        "shots": 50,
        "confidence": 0.95,
        "decoders": ["bp_osd", "relay_bp"],
        "bp_osd": {"max_iter": 10, "osd_order": 1},
        "relay_bp": {
            "pre_iter": 10,
            "num_sets": 5,
            "set_max_iter": 10,
            "gamma_dist_interval": [-0.24, 0.66],
        },
        "latency_shots": 0,
    }
    config.update(overrides)
    return config


def test_reference_circuits_are_checksummed_and_have_expected_shape():
    circuit, metadata = load_reference_circuit("bb18_d3_p003")
    assert (metadata["n"], metadata["k"], metadata["distance"]) == (18, 4, 3)
    assert (circuit.num_detectors, circuit.num_observables) == (54, 4)
    gross, gross_metadata = load_reference_circuit("bb144_d12_p003")
    assert gross_metadata["source"].startswith("https://github.com/trmue/relay/tree/")
    assert (gross.num_detectors, gross.num_observables) == (1728, 12)
    with pytest.raises(ValueError, match="unknown reference circuit"):
        load_reference_circuit("missing")


@pytest.mark.parametrize("implementation", ["f32", "i32", "i64"])
def test_relay_bp_reconstructs_logical_observables(implementation):
    dem = stim.DetectorErrorModel("error(0.1) D0 D1 L0")
    decoder = RelayBPDecoder(
        dem,
        pre_iter=5,
        num_sets=2,
        set_max_iter=5,
        implementation=implementation,
        data_scale_value=8 if implementation != "f32" else None,
        max_data_value=65535 if implementation != "f32" else None,
    )
    predictions = decoder.predict(np.array([[0, 0], [1, 1]], dtype=bool))
    if implementation == "f32":
        assert predictions.tolist() == [[False], [True]]
        assert decoder.last_decode_success.all()
    else:
        assert predictions.tolist() == [[False], [False]]
        assert not decoder.last_decode_success.all()


def test_bp_lsd_reconstructs_logical_observables():
    dem = stim.DetectorErrorModel("error(0.1) D0 D1 L0")
    decoder = BPLSDDecoder(dem, max_iter=5)
    predictions = decoder.predict(np.array([[0, 0], [1, 1]], dtype=bool))
    assert predictions.tolist() == [[False], [True]]


def test_qldpc_runner_and_performance_are_reproducible():
    first = run_qldpc_experiment(small_config())
    second = run_qldpc_experiment(small_config())
    assert first == second
    assert first["reference"]["n"] == 18
    assert set(first["decoders"]) == {"bp_osd", "relay_bp"}
    assert first["versions"]["relay-bp"] == "0.2.2"
    paired = first["paired_comparison"]
    assert paired["prediction_disagreements"] >= (
        paired["bp_osd_correct_relay_bp_wrong"] + paired["bp_osd_wrong_relay_bp_correct"]
    )
    assert 0 <= paired["mcnemar_pvalue"] <= 1
    assert first["decoders"]["relay_bp"]["num_errors"] == (
        first["decoders"]["bp_osd"]["num_errors"]
        + paired["bp_osd_correct_relay_bp_wrong"]
        - paired["bp_osd_wrong_relay_bp_correct"]
    )
    for name, row in first["decoders"].items():
        assert row["shots"] == 50
        assert row["ci_low"] <= row["ler"] <= row["ci_high"]
        if name == "relay_bp":
            assert 0 <= row["decode_success_rate"] <= 1
            assert row["decode_failures"] == round(50 * (1 - row["decode_success_rate"]))


def test_qldpc_runner_adds_offline_performance():
    report = run_qldpc_experiment(small_config(decoders=["relay_bp"], latency_shots=2))
    performance = report["decoders"]["relay_bp"]["performance"]
    assert performance["mode"] == "offline_batch1"
    assert performance["shots"] == 2
    assert performance["p99_latency_ms"] >= performance["p50_latency_ms"]


def test_qldpc_pairwise_comparison_supports_bp_lsd():
    report = run_qldpc_experiment(
        small_config(decoders=["bp_lsd", "relay_bp"], bp_lsd={"max_iter": 10, "lsd_order": 0})
    )
    paired = report["pairwise_comparisons"]["bp_lsd__vs__relay_bp"]
    assert report["decoders"]["relay_bp"]["num_errors"] == (
        report["decoders"]["bp_lsd"]["num_errors"]
        + paired["bp_lsd_correct_relay_bp_wrong"]
        - paired["bp_lsd_wrong_relay_bp_correct"]
    )
    assert 0 <= paired["mcnemar_pvalue"] <= 1


def test_shipped_qldpc_configs_run():
    for filename in ("qldpc_bb18.yaml", "qldpc_gross.yaml"):
        config = load_config(CONFIG_DIR / filename)
        config["shots"] = 2
        config["latency_shots"] = 1
        config["relay_bp"]["num_sets"] = 2
        config["relay_bp"]["pre_iter"] = 5
        config["relay_bp"]["set_max_iter"] = 5
        if "bp_osd" in config:
            config["bp_osd"]["max_iter"] = 5
            config["bp_osd"]["osd_order"] = 1
        assert run_qldpc_experiment(config)["decoders"]


def test_qldpc_cli_writes_json(tmp_path, capsys):
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps(small_config(decoders=["relay_bp"])))
    main(["--config", str(path), "--out", str(tmp_path / "run")])
    report = json.loads((tmp_path / "run" / "qldpc.json").read_text())
    assert report["decoders"]["relay_bp"]["shots"] == 50
    assert "relay_bp" in capsys.readouterr().out


def tuning_config(**overrides):
    candidates = [
        {
            "name": "long",
            "relay_bp": {
                "gamma0": 0.125,
                "pre_iter": 5,
                "num_sets": 3,
                "set_max_iter": 5,
                "gamma_dist_interval": [-0.24, 0.66],
                "stop_nconv": 1,
            },
        },
        {
            "name": "short",
            "relay_bp": {
                "gamma0": 0.125,
                "pre_iter": 5,
                "num_sets": 2,
                "set_max_iter": 5,
                "gamma_dist_interval": [-0.24, 0.66],
                "stop_nconv": 1,
            },
        },
    ]
    config = {
        "name": "test_tune",
        "seed": 29,
        "reference_circuit": "bb18_d3_p003",
        "confidence": 0.95,
        "validation_shots": 20,
        "validation_latency_shots": 0,
        "test_shots": 30,
        "test_latency_shots": 0,
        "candidates": candidates,
        "bp_osd": {"max_iter": 10, "osd_order": 1},
    }
    config.update(overrides)
    return config


def test_tuning_uses_validation_then_independent_test():
    first = run_qldpc_tuning(tuning_config())
    second = run_qldpc_tuning(tuning_config())
    assert first == second
    assert len(first["validation"]) == 2
    assert first["selected_candidate"] in {"long", "short"}
    assert first["validation_seed"] != first["test"]["seed"]
    assert first["test"]["config"]["shots"] == 30
    assert first["test"]["decoders"]["relay_bp"]["shots"] == 30
    assert first["test"]["paired_comparison"] is not None
    selected = first["selected_candidate"]
    expected_options = next(
        c["relay_bp"] for c in first["config"]["candidates"] if c["name"] == selected
    )
    assert first["test"]["config"]["relay_bp"] == expected_options


def test_shipped_tuning_config_runs():
    for filename, count in (("qldpc_gross_tune.yaml", 3), ("qldpc_realtime_tune.yaml", 5)):
        config = load_config(CONFIG_DIR / filename)
        config["validation_shots"] = 2
        config["test_shots"] = 2
        config["validation_latency_shots"] = 0
        config["test_latency_shots"] = 0
        config.pop("max_validation_p99_ms", None)
        for candidate in config["candidates"]:
            candidate["relay_bp"]["pre_iter"] = 5
            candidate["relay_bp"]["num_sets"] = 2
            candidate["relay_bp"]["set_max_iter"] = 5
        if "bp_lsd" in config:
            config["bp_lsd"]["max_iter"] = 5
        report = run_qldpc_tuning(config)
        assert len(report["validation"]) == count
        assert report["test"]["decoders"]["relay_bp"]["shots"] == 2


def test_tuning_cli_writes_json(tmp_path, capsys):
    path = tmp_path / "tuning.yaml"
    path.write_text(json.dumps(tuning_config(validation_shots=5, test_shots=5)))
    tuning_main(["--config", str(path), "--out", str(tmp_path / "run")])
    report = json.loads((tmp_path / "run" / "tuning.json").read_text())
    assert report["selected_candidate"] in {"long", "short"}
    assert report["test"]["decoders"]["relay_bp"]["shots"] == 5
    assert "selected:" in capsys.readouterr().out


def test_additional_noise_reference_circuits():
    for name, p in (
        ("bb144_d12_p0001", 0.001),
        ("bb144_d12_p0002", 0.002),
        ("bb144_d12_p0004", 0.004),
        ("bb144_d12_p0005", 0.005),
    ):
        circuit, metadata = load_reference_circuit(name)
        assert metadata["p"] == p
        assert (circuit.num_detectors, circuit.num_observables) == (1728, 12)


def test_qldpc_sweep_preserves_fixed_parameters_and_seeds():
    relay = {
        "gamma0": 0.125,
        "pre_iter": 5,
        "num_sets": 2,
        "set_max_iter": 5,
        "gamma_dist_interval": [-0.24, 0.66],
        "stop_nconv": 1,
    }
    report = run_qldpc_sweep(
        {
            "name": "test_sweep",
            "reference_circuits": ["bb18_d3_p003", "bb18_d3_p003"],
            "experiment_seeds": [31, 37],
            "shots": 10,
            "decoders": ["bp_lsd", "relay_bp"],
            "bp_lsd": {"max_iter": 5, "lsd_order": 0},
            "relay_bp": relay,
            "latency_shots": 0,
        }
    )
    assert len(report["cells"]) == 4
    assert [cell["experiment_seed"] for cell in report["cells"]] == [31, 37, 31, 37]
    assert len({cell["report"]["seed"] for cell in report["cells"]}) == 2
    for cell in report["cells"]:
        assert cell["report"]["config"]["relay_bp"] == relay
        assert "bp_lsd__vs__relay_bp" in cell["report"]["pairwise_comparisons"]


def test_qldpc_sweep_checkpoint_callback_and_skip():
    saved = []
    report = run_qldpc_sweep(
        {
            "name": "test_sweep_resume",
            "reference_circuits": ["bb18_d3_p003", "bb18_d3_p003"],
            "experiment_seeds": [31, 37],
            "shots": 5,
            "decoders": ["relay_bp"],
            "relay_bp": {
                "gamma0": 0.125,
                "pre_iter": 5,
                "num_sets": 2,
                "set_max_iter": 5,
                "gamma_dist_interval": [-0.24, 0.66],
                "stop_nconv": 1,
            },
            "latency_shots": 0,
        },
        on_cell=saved.append,
        skip={("bb18_d3_p003", 31)},
    )
    assert len(report["cells"]) == 2
    assert len(saved) == 2
    assert all(cell["experiment_seed"] == 37 for cell in saved)


def test_latency_budget_selection_and_explicit_fallback():
    rows = [
        {
            "candidate": "accurate_slow",
            "num_errors": 0,
            "iteration_budget": 100,
            "performance": {"p99_latency_ms": 200, "mean_latency_ms": 100},
        },
        {
            "candidate": "balanced",
            "num_errors": 1,
            "iteration_budget": 50,
            "performance": {"p99_latency_ms": 80, "mean_latency_ms": 50},
        },
        {
            "candidate": "fast",
            "num_errors": 2,
            "iteration_budget": 10,
            "performance": {"p99_latency_ms": 50, "mean_latency_ms": 40},
        },
    ]
    selected, met = _select_validation_row(rows, 100)
    assert selected["candidate"] == "balanced"
    assert met is True
    selected, met = _select_validation_row(rows, 10)
    assert selected["candidate"] == "fast"
    assert met is False
    selected, met = _select_validation_row(rows, None)
    assert selected["candidate"] == "accurate_slow"
    assert met is None


def test_pareto_front_removes_only_dominated_candidates():
    rows = [
        {"candidate": "accurate", "num_errors": 0, "performance": {"p99_latency_ms": 200}},
        {"candidate": "balanced", "num_errors": 1, "performance": {"p99_latency_ms": 80}},
        {"candidate": "fast", "num_errors": 2, "performance": {"p99_latency_ms": 50}},
        {"candidate": "dominated", "num_errors": 1, "performance": {"p99_latency_ms": 100}},
    ]
    assert [row["candidate"] for row in _pareto_front(rows)] == ["accurate", "balanced", "fast"]


def test_pareto_runner_shares_test_seed_and_evaluates_front():
    report = run_qldpc_pareto(
        tuning_config(
            baseline="bp_osd",
            validation_shots=10,
            validation_latency_shots=2,
            test_shots=10,
            test_latency_shots=0,
            profile_shots=2,
        )
    )
    assert report["test"]["baseline"] == "bp_osd"
    assert report["test"]["seed"] != report["validation_seed"]
    assert report["test"]["candidates"]
    assert {row["candidate"] for row in report["test"]["candidates"]} == set(
        report["pareto_candidates"]
    )
    assert all(row["shots"] == 10 for row in report["test"]["candidates"])
    assert all(
        "mcnemar_pvalue" in row["paired_with_baseline"] for row in report["test"]["candidates"]
    )
    assert all(row["profile"]["shots"] == 2 for row in report["test"]["candidates"])


def test_relay_rejects_unknown_implementation():
    dem = stim.DetectorErrorModel("error(0.1) D0")
    with pytest.raises(ValueError, match="unknown Relay-BP implementation"):
        RelayBPDecoder(dem, implementation="fixed16")


def test_pareto_can_evaluate_all_candidates_for_precision_comparison():
    report = run_qldpc_pareto(
        tuning_config(
            baseline="bp_osd",
            evaluate_all=True,
            validation_shots=5,
            validation_latency_shots=2,
            test_shots=5,
            test_latency_shots=0,
        )
    )
    assert set(report["evaluated_candidates"]) == {"long", "short"}
    assert set(report["pareto_candidates"]) <= set(report["evaluated_candidates"])


def test_precision_sweep_shares_shots_and_reports_pairwise_decisions():
    candidates = [
        {
            "name": "f32",
            "relay_bp": {
                "implementation": "f32",
                "gamma0": 0.125,
                "pre_iter": 5,
                "num_sets": 2,
                "set_max_iter": 5,
                "gamma_dist_interval": [-0.24, 0.66],
                "stop_nconv": 1,
            },
        },
        {
            "name": "i32_scaled",
            "relay_bp": {
                "implementation": "i32",
                "data_scale_value": 8,
                "max_data_value": 65535,
                "gamma0": 0.125,
                "pre_iter": 5,
                "num_sets": 2,
                "set_max_iter": 5,
                "gamma_dist_interval": [-0.24, 0.66],
                "stop_nconv": 1,
            },
        },
    ]
    saved = []
    report = run_qldpc_precision_sweep(
        {
            "reference_circuits": ["bb18_d3_p003", "bb18_d3_p003"],
            "experiment_seeds": [31, 37],
            "shots": 5,
            "candidates": candidates,
            "latency_shots": 0,
            "profile_shots": 2,
        },
        on_cell=saved.append,
        skip={("bb18_d3_p003", 31)},
    )
    assert len(report["cells"]) == len(saved) == 2
    for cell in report["cells"]:
        assert cell["experiment_seed"] == 37
        assert set(cell["decoders"]) == {"f32", "i32_scaled"}
        assert "f32__vs__i32_scaled" in cell["pairwise_comparisons"]
        assert all(0 <= row["decode_success_rate"] <= 1 for row in cell["decoders"].values())
        assert all(row["evaluation_seconds"] > 0 for row in cell["decoders"].values())
        assert all(row["profile"]["shots"] == 2 for row in cell["decoders"].values())


def test_failure_fallback_recovers_failed_relay_shot():
    dem = stim.DetectorErrorModel("error(0.1) D0 D1 L0")
    primary = RelayBPDecoder(
        dem,
        pre_iter=5,
        num_sets=2,
        set_max_iter=5,
        implementation="i32",
        data_scale_value=8,
        max_data_value=65535,
    )
    decoder = FailureFallbackDecoder(primary, BPLSDDecoder(dem, max_iter=5))
    predictions = decoder.predict(np.array([[0, 0], [1, 1]], dtype=bool))
    assert predictions.tolist() == [[False], [True]]
    assert decoder.last_fallback.tolist() == [False, True]


def test_qldpc_fallback_runner_reports_calls_and_pairwise_comparison():
    report = run_qldpc_experiment(
        small_config(
            decoders=["bp_lsd", "relay_bp_fallback"],
            bp_lsd={"max_iter": 10, "lsd_order": 0},
            fallback_decoder="bp_lsd",
        )
    )
    row = report["decoders"]["relay_bp_fallback"]
    assert row["fallback_calls"] == row["decode_failures"]
    assert 0 <= row["fallback_rate"] <= 1
    assert "bp_lsd__vs__relay_bp_fallback" in report["pairwise_comparisons"]


def test_relay_retry_recovers_and_reports_final_success():
    dem = stim.DetectorErrorModel("error(0.1) D0 D1 L0")
    primary = RelayBPDecoder(
        dem,
        pre_iter=5,
        num_sets=2,
        set_max_iter=5,
        implementation="i32",
        data_scale_value=8,
        max_data_value=65535,
    )
    fallback = RelayBPDecoder(dem, pre_iter=5, num_sets=2, set_max_iter=5, seed=1)
    decoder = FailureFallbackDecoder(primary, fallback)
    predictions = decoder.predict(np.array([[0, 0], [1, 1]], dtype=bool))
    assert predictions.tolist() == [[False], [True]]
    assert decoder.last_fallback.tolist() == [False, True]
    assert decoder.last_final_success.all()


def test_qldpc_relay_retry_runner_reports_final_failures():
    fallback_options = dict(small_config()["relay_bp"])
    fallback_options["num_sets"] = 2
    report = run_qldpc_experiment(
        small_config(
            decoders=["relay_bp_fallback"],
            fallback_decoder="relay_bp",
            fallback_relay_bp=fallback_options,
        )
    )
    row = report["decoders"]["relay_bp_fallback"]
    assert row["final_decode_failures"] == 0
    assert row["final_decode_success_rate"] == 1
    assert 0 <= row["fallback_rate"] <= 1


def test_relay_parallel_batch_matches_serial_exactly():
    circuit, _ = load_reference_circuit("bb18_d3_p003")
    data = sample_syndromes(circuit, 20, 41)
    options = {
        "pre_iter": 10,
        "num_sets": 5,
        "set_max_iter": 10,
        "gamma_dist_interval": (-0.24, 0.66),
        "seed": 41,
    }
    serial = RelayBPDecoder.from_circuit(circuit, **options, parallel=False)
    parallel = RelayBPDecoder.from_circuit(circuit, **options, parallel=True)
    np.testing.assert_array_equal(
        parallel.predict(data.detection_events), serial.predict(data.detection_events)
    )
    np.testing.assert_array_equal(parallel.last_decode_success, serial.last_decode_success)
    np.testing.assert_array_equal(parallel.last_decode_iterations, serial.last_decode_iterations)
    np.testing.assert_array_equal(
        serial.predict(data.detection_events),
        serial.predict(data.detection_events[::-1])[::-1],
    )


def test_relay_profile_breakdown_serial_and_parallel():
    circuit, _ = load_reference_circuit("bb18_d3_p003")
    data = sample_syndromes(circuit, 10, 43)
    options = {
        "pre_iter": 10,
        "num_sets": 5,
        "set_max_iter": 10,
        "gamma_dist_interval": (-0.24, 0.66),
        "seed": 43,
    }
    serial = RelayBPDecoder.from_circuit(circuit, **options, parallel=False)
    profile = profile_relay_decoder(serial, data.detection_events, shots=5, warmup_shots=1)
    assert profile["successes"] + profile["failures"] == 5
    assert profile["iterations"]["mean"] > 0
    assert profile["initialization_ms"]["mean"] > 0
    assert profile["native_decode_ms"]["mean"] > 0
    assert profile["python_overhead_ms"]["mean"] >= 0
    assert profile["throughput_shots_per_second"] > 0

    parallel = RelayBPDecoder.from_circuit(circuit, **options, parallel=True)
    parallel_profile = profile_relay_decoder(
        parallel, data.detection_events, shots=2, warmup_shots=0
    )
    assert parallel_profile["mode"] == "parallel"
    assert parallel_profile["initialization_ms"] is None
    assert parallel_profile["native_decode_ms"]["mean"] > 0


def test_pareto_supports_fallback_candidates():
    config = tuning_config(
        baseline="bp_lsd",
        bp_lsd={"max_iter": 5, "lsd_order": 0},
        fallback_decoder="bp_lsd",
        evaluate_all=True,
        validation_shots=5,
        validation_latency_shots=2,
        test_shots=5,
        test_latency_shots=0,
    )
    for candidate in config["candidates"]:
        candidate["decoder"] = "relay_bp_fallback"
    report = run_qldpc_pareto(config)
    assert set(report["evaluated_candidates"]) == {"long", "short"}
    for row in report["test"]["candidates"]:
        assert "fallback_calls" in row
        assert "final_decode_success_rate" in row


def test_relay_alpha_options_and_validation():
    dem = stim.DetectorErrorModel("error(0.1) D0 D1 L0")
    for alpha in (None, 0.95, 0.0):
        decoder = RelayBPDecoder(
            dem,
            pre_iter=5,
            num_sets=2,
            set_max_iter=5,
            alpha=alpha,
            alpha_iteration_scaling_factor=10,
        )
        assert decoder.predict(np.array([[0, 0], [1, 1]], dtype=bool)).tolist() == [
            [False],
            [True],
        ]
    with pytest.raises(ValueError, match="alpha_iteration_scaling_factor"):
        RelayBPDecoder(dem, alpha=0, alpha_iteration_scaling_factor=0)
