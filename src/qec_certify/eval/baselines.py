"""Evaluate the classical baseline decoders (Tier 0 MWPM, Tier 1 exact ML) from a config.

Usage::

    python -m qec_certify.eval.baselines --config configs/m1_d3_depolarizing.yaml \
        [--out runs/m1_d3_depolarizing]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pymatching
import stim

import qec_certify
from qec_certify.codes import code_capacity_circuit, make_code
from qec_certify.decoders import Decoder, ExactMLDecoder, MWPMDecoder
from qec_certify.eval.metrics import evaluate_decoder
from qec_certify.sim import sample_syndromes
from qec_certify.utils import git_commit_hash, load_config, spawn_seeds, split_seeds

DECODERS: dict[str, Callable[[stim.Circuit], Decoder]] = {
    "mwpm": MWPMDecoder.from_circuit,
    "ml_exact": ExactMLDecoder.from_circuit,
}


def run_baselines(config: dict[str, Any]) -> dict[str, Any]:
    """Sample test shots for every noise strength and evaluate each configured decoder."""
    noise = config["noise"]
    if noise["model"] != "depolarizing":
        raise ValueError(f"unsupported noise model {noise['model']!r}; M1 supports 'depolarizing'")
    unknown = set(config["decoders"]) - set(DECODERS)
    if unknown:
        raise ValueError(f"unknown decoders {sorted(unknown)}; choose from {sorted(DECODERS)}")

    code = make_code(config["code"]["family"], config["code"]["distance"])
    basis = config.get("memory_basis", "Z")
    shots = int(config["shots"])
    confidence = float(config.get("confidence", 0.95))
    probabilities = noise["p"] if isinstance(noise["p"], list) else [noise["p"]]
    seeds = spawn_seeds(split_seeds(config["seed"])["test"], len(probabilities))

    results = []
    for p, seed in zip(probabilities, seeds, strict=True):
        circuit = code_capacity_circuit(code, float(p), basis)
        data = sample_syndromes(circuit, shots, seed)
        entry: dict[str, Any] = {"p": p, "seed": seed, "decoders": {}}
        for name in config["decoders"]:
            decoder = DECODERS[name](circuit).fit()
            row: dict[str, Any] = evaluate_decoder(decoder, data, confidence).to_dict()
            if isinstance(decoder, ExactMLDecoder):
                row["exact_ler"] = decoder.optimal_logical_error_rate
            entry["decoders"][name] = row
        results.append(entry)

    return {
        "config": config,
        "git_commit": git_commit_hash(),
        "versions": {
            "qec_certify": qec_certify.__version__,
            "stim": stim.__version__,
            "pymatching": pymatching.__version__,
        },
        "results": results,
    }


def format_report(report: dict[str, Any]) -> str:
    lines = [f"{'p':>8}  {'decoder':<9} {'errors/shots':>15}  {'LER':>10}  CI"]
    for entry in report["results"]:
        for name, row in entry["decoders"].items():
            lines.append(
                f"{entry['p']:>8g}  {name:<9} {row['num_errors']:>7}/{row['shots']:<7}  "
                f"{row['ler']:>10.3e}  [{row['ci_low']:.3e}, {row['ci_high']:.3e}]"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", type=Path, help="directory to write baselines.json into")
    args = parser.parse_args(argv)

    report = run_baselines(load_config(args.config))
    print(format_report(report))
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "baselines.json"
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
