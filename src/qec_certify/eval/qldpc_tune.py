"""Select Relay-BP parameters on validation shots, then evaluate once on held-out test shots.

Usage::

    python -m qec_certify.eval.qldpc_tune --config configs/qldpc_gross_tune.yaml \
        --out runs/qldpc_gross_tune
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import stim

from qec_certify.codes import load_reference_circuit
from qec_certify.eval.metrics import benchmark_decoder, logical_error_rate
from qec_certify.eval.qldpc import (
    _make_decoder,
    decoder_diagnostics,
    format_qldpc_report,
    run_qldpc_experiment,
)
from qec_certify.sim import SyndromeData, sample_syndromes
from qec_certify.utils import load_config, split_seeds

SELECTION_RULE = (
    "minimum validation errors; then minimum measured mean latency; "
    "then pre_iter + num_sets * set_max_iter; then candidate name"
)


def _validation_row(
    candidate: dict[str, Any],
    circuit: stim.Circuit,
    data: SyndromeData,
    config: dict[str, Any],
) -> dict[str, Any]:
    options = candidate["relay_bp"]
    decoder = _make_decoder(
        candidate.get("decoder", "relay_bp"),
        circuit,
        {**config, "seed": config["seed"], "relay_bp": options},
    )
    predicted = decoder.predict(data.detection_events)
    row: dict[str, Any] = {
        "candidate": candidate["name"],
        **logical_error_rate(predicted, data.observable_flips, config["confidence"]).to_dict(),
        "iteration_budget": options["pre_iter"] + options["num_sets"] * options["set_max_iter"],
        **decoder_diagnostics(decoder),
    }
    if config.get("validation_latency_shots", 0):
        row["performance"] = benchmark_decoder(
            decoder,
            data.detection_events,
            shots=int(config["validation_latency_shots"]),
            warmup_shots=int(config.get("latency_warmup", 1)),
        ).to_dict()
    return row


def _selection_key(row: dict[str, Any]) -> tuple[float, float, int, str]:
    latency = row.get("performance", {}).get("mean_latency_ms", float("inf"))
    return row["num_errors"], latency, row["iteration_budget"], row["candidate"]


def _select_validation_row(
    rows: list[dict[str, Any]], max_p99_ms: float | None
) -> tuple[dict[str, Any], bool | None]:
    if max_p99_ms is None:
        return min(rows, key=_selection_key), None
    eligible = [row for row in rows if row["performance"]["p99_latency_ms"] <= max_p99_ms]
    if eligible:
        return min(eligible, key=_selection_key), True

    def fallback_key(row: dict[str, Any]) -> tuple[float, int, float, int, str]:
        return (
            row["performance"]["p99_latency_ms"],
            row["num_errors"],
            row["performance"]["mean_latency_ms"],
            row["iteration_budget"],
            row["candidate"],
        )

    return min(rows, key=fallback_key), False


def run_qldpc_tuning(config: dict[str, Any]) -> dict[str, Any]:
    """Tune on the ``val`` split and delegate one final test to ``run_qldpc_experiment``."""
    config.setdefault("confidence", 0.95)
    circuit, reference = load_reference_circuit(config["reference_circuit"])
    seeds = split_seeds(config["seed"])
    validation_data = sample_syndromes(circuit, int(config["validation_shots"]), seeds["val"])
    validation_rows = [
        _validation_row(candidate, circuit, validation_data, config)
        for candidate in config["candidates"]
    ]
    selected, budget_met = _select_validation_row(
        validation_rows, config.get("max_validation_p99_ms")
    )
    candidate = next(c for c in config["candidates"] if c["name"] == selected["candidate"])

    baseline_decoders = [name for name in ("bp_lsd", "bp_osd") if name in config]
    test_config = {
        "name": config["name"],
        "seed": config["seed"],
        "reference_circuit": config["reference_circuit"],
        "shots": config["test_shots"],
        "confidence": config["confidence"],
        "decoders": ["relay_bp", *baseline_decoders],
        "relay_bp": candidate["relay_bp"],
        "latency_shots": config.get("test_latency_shots", 0),
        "latency_warmup": config.get("latency_warmup", 1),
    }
    for name in baseline_decoders:
        test_config[name] = config[name]
    test_report = run_qldpc_experiment(test_config)
    return {
        "config": config,
        "reference": reference,
        "selection_rule": SELECTION_RULE,
        "latency_budget": {
            "max_validation_p99_ms": config.get("max_validation_p99_ms"),
            "met": budget_met,
        },
        "validation_seed": seeds["val"],
        "validation": validation_rows,
        "selected_candidate": selected["candidate"],
        "test": test_report,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_qldpc_tuning(load_config(args.config))
    for row in report["validation"]:
        print(
            f"validation {row['candidate']}: {row['num_errors']}/{row['shots']} "
            f"budget={row['iteration_budget']}"
        )
    print(f"selected: {report['selected_candidate']}")
    if report["latency_budget"]["max_validation_p99_ms"] is not None:
        print(f"latency budget met: {report['latency_budget']['met']}")
    print(format_qldpc_report(report["test"]))
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "tuning.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
