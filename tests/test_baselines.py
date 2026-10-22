import json
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
    assert set(report["versions"]) == {"qec_certify", "stim", "pymatching"}


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


def test_run_baselines_rejects_unknown_settings():
    with pytest.raises(ValueError, match="noise model"):
        run_baselines(small_config(noise={"model": "biased", "p": 0.1}))
    with pytest.raises(ValueError, match="unknown decoders"):
        run_baselines(small_config(decoders=["mwpm", "transformer"]))


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("m1_*.yaml")), ids=lambda p: p.name)
def test_shipped_m1_configs_run(path):
    config = load_config(path)
    config["shots"] = 200
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
