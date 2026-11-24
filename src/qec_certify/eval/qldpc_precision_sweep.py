"""Compare Relay-BP numerical implementations across noise strengths and seeds.

Usage::

    python -m qec_certify.eval.qldpc_precision_sweep \
        --config configs/qldpc_precision_robustness.yaml --out runs/qldpc_precision_robustness
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qec_certify.codes import load_reference_circuit
from qec_certify.eval.metrics import benchmark_decoder, logical_error_rate
from qec_certify.eval.profiling import profile_relay_decoder
from qec_certify.eval.qldpc import _make_decoder, _pairwise_comparisons, decoder_diagnostics
from qec_certify.sim import sample_syndromes
from qec_certify.utils import git_commit_hash, load_config, split_seeds


def run_qldpc_precision_sweep(
    config: dict[str, Any],
    *,
    on_cell: Callable[[dict[str, Any]], None] | None = None,
    skip: set[tuple[str, int]] | None = None,
) -> dict[str, Any]:
    """Compare implementations on shared shots without changing algorithm parameters."""
    cells = []
    skipped = skip or set()
    for reference_name in config["reference_circuits"]:
        circuit, reference = load_reference_circuit(reference_name)
        for experiment_seed in config["experiment_seeds"]:
            if (reference_name, experiment_seed) in skipped:
                continue
            seed = split_seeds(experiment_seed)["test"]
            data = sample_syndromes(circuit, int(config["shots"]), seed)
            predictions = {}
            rows = {}
            for candidate in config["candidates"]:
                name = candidate["name"]
                start = time.perf_counter()
                decoder = _make_decoder(
                    "relay_bp",
                    circuit,
                    {"seed": experiment_seed, "relay_bp": candidate["relay_bp"]},
                )
                predicted = decoder.predict(data.detection_events)
                predictions[name] = predicted
                row = logical_error_rate(
                    predicted, data.observable_flips, config.get("confidence", 0.95)
                ).to_dict()
                row.update(decoder_diagnostics(decoder))
                row["evaluation_seconds"] = time.perf_counter() - start
                if config.get("latency_shots", 0):
                    row["performance"] = benchmark_decoder(
                        decoder,
                        data.detection_events,
                        shots=int(config["latency_shots"]),
                        warmup_shots=int(config.get("latency_warmup", 1)),
                    ).to_dict()
                if config.get("profile_shots", 0):
                    row["profile"] = profile_relay_decoder(
                        decoder,
                        data.detection_events,
                        shots=int(config["profile_shots"]),
                        warmup_shots=int(config.get("profile_warmup", 1)),
                    )
                rows[name] = row
            cell = {
                "reference": reference,
                "experiment_seed": experiment_seed,
                "seed": seed,
                "decoders": rows,
                "pairwise_comparisons": _pairwise_comparisons(predictions, data.observable_flips),
            }
            cells.append(cell)
            if on_cell is not None:
                on_cell(cell)
    return {"config": config, "git_commit": git_commit_hash(), "cells": cells}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "cells.jsonl"
    cells = []
    if checkpoint.exists():
        cells = [json.loads(line) for line in checkpoint.read_text().splitlines() if line]
    completed = {(cell["reference"]["name"], cell["experiment_seed"]) for cell in cells}
    if completed:
        print(f"resuming with {len(completed)} checkpointed cells")

    def save_cell(cell: dict[str, Any]) -> None:
        cells.append(cell)
        with checkpoint.open("a", encoding="utf-8") as f:
            f.write(json.dumps(cell) + "\n")
        summary = ", ".join(
            f"{name}={row['num_errors']}/{row['shots']}" for name, row in cell["decoders"].items()
        )
        print(f"{cell['reference']['name']} seed={cell['experiment_seed']}: {summary}", flush=True)

    report = run_qldpc_precision_sweep(config, on_cell=save_cell, skip=completed)
    report["cells"] = cells
    path = args.out / "precision_sweep.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
