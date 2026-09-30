import json
from pathlib import Path

import pytest

from qec_certify.eval.baselines import main, run_baselines
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


def test_run_baselines_rejects_unknown_settings():
    with pytest.raises(ValueError, match="noise model"):
        run_baselines(small_config(noise={"model": "biased", "p": 0.1}))
    with pytest.raises(ValueError, match="unknown decoders"):
        run_baselines(small_config(decoders=["mwpm", "transformer"]))


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("m1_*.yaml")), ids=lambda p: p.name)
def test_shipped_m1_configs_run(path):
    config = load_config(path)
    config["shots"] = 200
    assert run_baselines(config)["results"]


def test_cli_writes_json(tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(small_config(noise={"model": "depolarizing", "p": 0.1})))
    main(["--config", str(config_path), "--out", str(tmp_path / "run")])
    report = json.loads((tmp_path / "run" / "baselines.json").read_text())
    assert report["results"][0]["decoders"]["mwpm"]["shots"] == 2000
    assert "ml_exact" in capsys.readouterr().out
