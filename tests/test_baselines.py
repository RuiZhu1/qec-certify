import json
import math
from pathlib import Path

import pytest

from qec_certify.eval.baselines import format_report, main, run_baselines
from qec_certify.utils import load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def small_config(**overrides):
    config = {
        "name": "test",
        "seed": 1,
        "code": {"family": "rotated_surface", "distance": 3},
        "noise": {"model": "depolarizing", "p": [0.05, 0.1]},
        "memory_basis": "Z",
        "shots": 2000,
        "confidence": 0.95,
        "decoders": ["mwpm", "ml_exact"],
    }
    config.update(overrides)
    return config


def test_run_baselines_report_structure():
    report = run_baselines(small_config())
    assert [entry["p"] for entry in report["results"]] == [0.05, 0.1]
    assert len({entry["seed"] for entry in report["results"]}) == 2
    for entry in report["results"]:
        assert set(entry["decoders"]) == {"mwpm", "ml_exact"}
        for row in entry["decoders"].values():
            assert row["shots"] == 2000
            assert row["ci_low"] <= row["ler"] <= row["ci_high"]
        assert 0 < entry["decoders"]["ml_exact"]["exact_ler"] < 1
    assert set(report["versions"]) == {"qec_certify", "stim", "pymatching", "ldpc"}


def test_run_baselines_is_reproducible():
    assert run_baselines(small_config())["results"] == run_baselines(small_config())["results"]


def test_list_of_distances_and_all_decoders():
    config = small_config(
        code={"family": "rotated_surface", "distance": [3, 5]},
        noise={"model": "depolarizing", "p": 0.1},
        decoders=["mwpm", "mwpm_correlated", "ml_sweep"],
    )
    report = run_baselines(config)
    assert [entry["distance"] for entry in report["results"]] == [3, 5]
    for entry in report["results"]:
        assert set(entry["decoders"]) == {"mwpm", "mwpm_correlated", "ml_sweep"}
        assert entry["decoders"]["ml_sweep"]["frontier"] <= entry["distance"] + 3
    assert "ml_sweep" in format_report(report)


def test_first_distance_keeps_single_distance_seeds():
    decoders = ["mwpm", "ml_sweep"]
    single = run_baselines(small_config(decoders=decoders))
    multi = run_baselines(
        small_config(code={"family": "rotated_surface", "distance": [3, 5]}, decoders=decoders)
    )
    assert multi["results"][: len(single["results"])] == single["results"]


def test_sweep_and_table_ml_agree_on_shots():
    report = run_baselines(small_config(decoders=["ml_exact", "ml_sweep"]))
    for entry in report["results"]:
        rows = entry["decoders"]
        # Same decisions, ties included, so the same failures on the same shots.
        assert rows["ml_exact"]["num_errors"] == rows["ml_sweep"]["num_errors"]


def test_workers_do_not_change_results():
    config = small_config(decoders=["mwpm", "ml_sweep"])
    assert run_baselines(config, workers=2)["results"] == run_baselines(config)["results"]


def test_structured_noise_grid_and_descriptors():
    config = small_config(
        noise={"model": "biased", "p": 0.1, "bias": [0.5, 10]},
        memory_basis="X",
        decoders=["mwpm", "ml_sweep"],
    )
    report = run_baselines(config)
    assert [(e["distance"], e["noise"]["bias"]) for e in report["results"]] == [(3, 0.5), (3, 10)]
    assert len({e["seed"] for e in report["results"]}) == 2
    for entry in report["results"]:
        assert entry["p"] == 0.1
        assert entry["descriptors"]["bias"] == pytest.approx(entry["noise"]["bias"])
    assert "bias=10" in format_report(report)


def test_shots_per_distance():
    config = small_config(
        code={"family": "rotated_surface", "distance": [3, 5]},
        noise={"model": "depolarizing", "p": 0.1},
        shots={3: 300, "5": 100},
        decoders=["mwpm"],
    )
    report = run_baselines(config)
    assert [e["decoders"]["mwpm"]["shots"] for e in report["results"]] == [300, 100]


def circuit_level_config(**overrides):
    config = small_config(
        noise={"model": "circuit_level", "p": [0.01, 0.05]},
        rounds=3,
        shots=50,
        decoders=["mwpm", "mwpm_correlated", "bp_osd", "union_find"],
        latency_shots=5,
        latency_warmup=2,
    )
    config.update(overrides)
    return config


def test_circuit_level_baselines_report_structure():
    report = run_baselines(circuit_level_config())
    assert [entry["p"] for entry in report["results"]] == [0.01, 0.05]
    for entry in report["results"]:
        assert entry["experiment"] == "circuit_level"
        assert entry["rounds"] == 3
        assert "descriptors" not in entry
        for row in entry["decoders"].values():
            assert row["per_round_model"] == "independent_equal_hazard"
            assert row["ler_per_round"] == pytest.approx(1 - (1 - row["ler"]) ** (1 / 3))
            assert row["ler_per_round_ci_low"] <= row["ler_per_round"]
            assert row["ler_per_round"] <= row["ler_per_round_ci_high"]
            performance = row["performance"]
            assert performance["mode"] == "offline_batch1"
            assert performance["shots"] == 5
            assert performance["warmup_shots"] == 2
            assert performance["p99_latency_ms"] >= performance["p50_latency_ms"]
            assert math.isfinite(performance["throughput_shots_per_second"])


def test_circuit_level_reproducibility_and_rounds_per_distance():
    config = circuit_level_config(
        code={"family": "rotated_surface", "distance": [3, 5]},
        noise={"model": "circuit_level", "p": 0.01},
        rounds={3: 2, "5": 4},
        shots={3: 200, "5": 100},
        decoders=["mwpm"],
        latency_shots=0,
    )
    first = run_baselines(config)
    second = run_baselines(config, workers=2)
    assert first["results"] == second["results"]
    assert [entry["rounds"] for entry in first["results"]] == [2, 4]


def test_circuit_level_streaming_performance_and_unsupported_decoder():
    report = run_baselines(
        circuit_level_config(
            decoders=["mwpm", "union_find"],
            streaming_latency_shots=2,
            streaming_warmup=1,
            round_period_us=1000,
        )
    )
    for entry in report["results"]:
        streaming = entry["decoders"]["mwpm"]["streaming_performance"]
        assert streaming["mode"] == "causal_replay_batch1"
        assert streaming["rounds"] == 6
        assert streaming["round_period_us"] == 1000
        assert streaming["max_pending_rounds"] >= 1
        unsupported = entry["decoders"]["union_find"]["streaming_performance"]
        assert unsupported == {"mode": "unsupported_inconsistent_prefixes"}


def test_sliding_window_baselines_report_separate_streaming_ler():
    report = run_baselines(
        circuit_level_config(
            decoders=["mwpm", "mwpm_correlated", "bp_osd", "union_find"],
            streaming_method="sliding_window",
            window_rounds=3,
            streaming_latency_shots=2,
        )
    )
    for entry in report["results"]:
        row = entry["decoders"]["mwpm"]
        assert row["streaming_approximation"] == "exact_full_history"
        assert row["streaming_ler"]["num_errors"] == row["num_errors"]
        assert row["streaming_performance"]["mode"] == "sliding_window_mwpm_w3"
        correlated = entry["decoders"]["mwpm_correlated"]
        assert correlated["streaming_ler"]["num_errors"] == correlated["num_errors"]
        for comparison in (row["streaming_comparison"], correlated["streaming_comparison"]):
            assert comparison["prediction_disagreements"] == 0
            assert comparison["offline_correct_streaming_wrong"] == 0
            assert comparison["offline_wrong_streaming_correct"] == 0
        assert entry["decoders"]["bp_osd"]["streaming_performance"] == {
            "mode": "unsupported_sliding_window_decoder"
        }
        assert entry["decoders"]["union_find"]["streaming_performance"] == {
            "mode": "unsupported_inconsistent_prefixes"
        }


def test_delayed_commit_baselines_full_window_matches_offline():
    report = run_baselines(
        circuit_level_config(
            decoders=["mwpm", "mwpm_correlated"],
            streaming_method="delayed_commit",
            window_rounds=3,
            streaming_latency_shots=2,
        )
    )
    for entry in report["results"]:
        for row in entry["decoders"].values():
            assert row["streaming_approximation"] == "exact_full_history"
            assert row["streaming_ler"]["num_errors"] == row["num_errors"]
            assert row["streaming_performance"]["mode"] == "delayed_commit_mwpm_w3"


def test_window_grid_reuses_shots_and_reports_paired_errors():
    report = run_baselines(
        circuit_level_config(
            code={"family": "rotated_surface", "distance": [3, 5]},
            noise={"model": "circuit_level", "p": 0.01},
            rounds={3: 3, 5: 5},
            shots={3: 100, 5: 50},
            decoders=["mwpm"],
            streaming_method="delayed_commit",
            window_rounds={3: [2, 3], 5: [2, 5]},
            streaming_latency_shots=1,
            latency_shots=0,
        )
    )
    assert [(e["distance"], e["window_rounds"]) for e in report["results"]] == [
        (3, 2),
        (3, 3),
        (5, 2),
        (5, 5),
    ]
    for distance in (3, 5):
        entries = [e for e in report["results"] if e["distance"] == distance]
        assert len({e["seed"] for e in entries}) == 1
        for entry in entries:
            row = entry["decoders"]["mwpm"]
            comparison = row["streaming_comparison"]
            assert row["streaming_ler"]["num_errors"] == (
                row["num_errors"]
                + comparison["offline_correct_streaming_wrong"]
                - comparison["offline_wrong_streaming_correct"]
            )
            assert comparison["prediction_disagreements"] >= (
                comparison["offline_correct_streaming_wrong"]
                + comparison["offline_wrong_streaming_correct"]
            )
    full = [report["results"][1], report["results"][3]]
    assert all(
        e["decoders"]["mwpm"]["streaming_comparison"]["prediction_disagreements"] == 0 for e in full
    )
    assert "w" in format_report(report).splitlines()[0]


def test_run_baselines_rejects_unknown_settings():
    with pytest.raises(ValueError, match="unknown noise model"):
        run_baselines(small_config(noise={"model": "coherent", "p": 0.1}))
    with pytest.raises(ValueError, match="needs its 'bias' parameter"):
        run_baselines(small_config(noise={"model": "biased", "p": 0.1}))
    with pytest.raises(ValueError, match="unknown decoders"):
        run_baselines(small_config(decoders=["mwpm", "transformer"]))
    with pytest.raises(ValueError, match="rotated_surface"):
        run_baselines(
            circuit_level_config(code={"family": "repetition", "distance": 3}, decoders=["mwpm"])
        )
    with pytest.raises(ValueError, match="rounds"):
        run_baselines(circuit_level_config(rounds=0, decoders=["mwpm"]))
    with pytest.raises(ValueError, match="latency shots"):
        run_baselines(circuit_level_config(latency_shots=1000, decoders=["mwpm"]))
    with pytest.raises(ValueError, match="streaming latency requires"):
        run_baselines(small_config(decoders=["mwpm"], streaming_latency_shots=1))
    with pytest.raises(ValueError, match="unknown streaming method"):
        run_baselines(
            circuit_level_config(
                decoders=["mwpm"], streaming_latency_shots=1, streaming_method="future"
            )
        )
    with pytest.raises(ValueError, match="window_rounds grids"):
        run_baselines(
            circuit_level_config(decoders=["mwpm"], streaming_latency_shots=1, window_rounds=[2, 3])
        )


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("m[134]_*.yaml")), ids=lambda p: p.name)
def test_shipped_baseline_configs_run(path):
    config = load_config(path)
    config["shots"] = 200
    config["latency_shots"] = 2
    if "streaming_latency_shots" in config:
        config["streaming_latency_shots"] = 1
    distances = config["code"]["distance"]
    if isinstance(distances, list):  # smoke test: the smallest codes are enough
        config["code"]["distance"] = distances[:2]
    assert run_baselines(config)["results"]


def test_cli_writes_json(tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(small_config(noise={"model": "depolarizing", "p": 0.1})))
    main(["--config", str(config_path), "--out", str(tmp_path / "run")])
    report = json.loads((tmp_path / "run" / "baselines.json").read_text())
    assert report["results"][0]["decoders"]["mwpm"]["shots"] == 2000
    assert "ml_exact" in capsys.readouterr().out
