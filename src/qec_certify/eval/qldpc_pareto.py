"""Evaluate the validation accuracy/latency Pareto frontier on shared held-out shots.

Usage::

    python -m qec_certify.eval.qldpc_pareto --config configs/qldpc_pareto.yaml \
        --out runs/qldpc_pareto
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

from qec_certify.codes import load_reference_circuit
from qec_certify.decoders import RelayBPDecoder
from qec_certify.eval.metrics import benchmark_decoder, logical_error_rate
from qec_certify.eval.profiling import profile_relay_decoder
from qec_certify.eval.qldpc import _make_decoder, decoder_diagnostics
from qec_certify.eval.qldpc_tune import _validation_row
from qec_certify.sim import sample_syndromes
from qec_certify.utils import git_commit_hash, load_config, split_seeds


def _pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Nondominated rows by validation error count and p99 latency."""

    def dominates(first: dict[str, Any], second: dict[str, Any]) -> bool:
        first_values = (first["num_errors"], first["performance"]["p99_latency_ms"])
        second_values = (second["num_errors"], second["performance"]["p99_latency_ms"])
        return all(a <= b for a, b in zip(first_values, second_values, strict=True)) and any(
            a < b for a, b in zip(first_values, second_values, strict=True)
        )

    front = [
        row for row in rows if not any(dominates(other, row) for other in rows if other is not row)
    ]
    return sorted(front, key=lambda row: (row["num_errors"], row["performance"]["p99_latency_ms"]))


def _paired(baseline, candidate, truth, baseline_name: str, candidate_name: str) -> dict[str, Any]:
    baseline_wrong = np.any(baseline != truth, axis=1)
    candidate_wrong = np.any(candidate != truth, axis=1)
    regressions = int((~baseline_wrong & candidate_wrong).sum())
    corrections = int((baseline_wrong & ~candidate_wrong).sum())
    discordant = regressions + corrections
    return {
        f"{baseline_name}_correct_{candidate_name}_wrong": regressions,
        f"{baseline_name}_wrong_{candidate_name}_correct": corrections,
        "prediction_disagreements": int(np.any(baseline != candidate, axis=1).sum()),
        "mcnemar_pvalue": (
            binomtest(min(regressions, corrections), discordant, 0.5).pvalue if discordant else 1.0
        ),
    }


def run_qldpc_pareto(config: dict[str, Any]) -> dict[str, Any]:
    config.setdefault("confidence", 0.95)
    circuit, reference = load_reference_circuit(config["reference_circuit"])
    seeds = split_seeds(config["seed"])
    validation_data = sample_syndromes(circuit, int(config["validation_shots"]), seeds["val"])
    validation_rows = [
        _validation_row(candidate, circuit, validation_data, config)
        for candidate in config["candidates"]
    ]
    front = _pareto_front(validation_rows)
    evaluated_names = (
        {candidate["name"] for candidate in config["candidates"]}
        if config.get("evaluate_all", False)
        else {row["candidate"] for row in front}
    )

    test_data = sample_syndromes(circuit, int(config["test_shots"]), seeds["test"])
    baseline_name = config.get("baseline", "bp_lsd")
    baseline_decoder = _make_decoder(baseline_name, circuit, config)
    baseline_predicted = baseline_decoder.predict(test_data.detection_events)
    baseline_row = logical_error_rate(
        baseline_predicted, test_data.observable_flips, config["confidence"]
    ).to_dict()
    if config.get("test_latency_shots", 0):
        baseline_row["performance"] = benchmark_decoder(
            baseline_decoder,
            test_data.detection_events,
            shots=int(config["test_latency_shots"]),
            warmup_shots=int(config.get("latency_warmup", 1)),
        ).to_dict()

    candidate_rows = []
    for candidate in config["candidates"]:
        if candidate["name"] not in evaluated_names:
            continue
        decoder = _make_decoder(
            candidate.get("decoder", "relay_bp"),
            circuit,
            {**config, "seed": config["seed"], "relay_bp": candidate["relay_bp"]},
        )
        predicted = decoder.predict(test_data.detection_events)
        row: dict[str, Any] = {
            "candidate": candidate["name"],
            **logical_error_rate(
                predicted, test_data.observable_flips, config["confidence"]
            ).to_dict(),
            **decoder_diagnostics(decoder),
            "paired_with_baseline": _paired(
                baseline_predicted,
                predicted,
                test_data.observable_flips,
                baseline_name,
                candidate["name"],
            ),
        }
        if config.get("test_latency_shots", 0):
            row["performance"] = benchmark_decoder(
                decoder,
                test_data.detection_events,
                shots=int(config["test_latency_shots"]),
                warmup_shots=int(config.get("latency_warmup", 1)),
            ).to_dict()
        if config.get("profile_shots", 0) and isinstance(decoder, RelayBPDecoder):
            row["profile"] = profile_relay_decoder(
                decoder,
                test_data.detection_events,
                shots=int(config["profile_shots"]),
                warmup_shots=int(config.get("profile_warmup", 1)),
            )
        candidate_rows.append(row)
    return {
        "config": config,
        "reference": reference,
        "git_commit": git_commit_hash(),
        "validation_seed": seeds["val"],
        "validation": validation_rows,
        "pareto_candidates": [row["candidate"] for row in front],
        "evaluated_candidates": [row["candidate"] for row in candidate_rows],
        "test": {
            "seed": seeds["test"],
            "shots": int(config["test_shots"]),
            "baseline": baseline_name,
            "baseline_result": baseline_row,
            "candidates": candidate_rows,
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_qldpc_pareto(load_config(args.config))
    print(f"validation Pareto candidates: {report['pareto_candidates']}")
    baseline = report["test"]["baseline_result"]
    print(f"{report['test']['baseline']}: {baseline['num_errors']}/{baseline['shots']}")
    for row in report["test"]["candidates"]:
        print(f"{row['candidate']}: {row['num_errors']}/{row['shots']}")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "pareto.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
