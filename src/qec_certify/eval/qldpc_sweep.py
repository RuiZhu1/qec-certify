"""Run fixed decoder parameters across reference noise strengths and independent seeds.

Usage::

    python -m qec_certify.eval.qldpc_sweep --config configs/qldpc_robustness.yaml \
        --out runs/qldpc_robustness
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qec_certify.eval.qldpc import run_qldpc_experiment
from qec_certify.utils import load_config


def run_qldpc_sweep(
    config: dict[str, Any],
    *,
    on_cell: Callable[[dict[str, Any]], None] | None = None,
    skip: set[tuple[str, int]] | None = None,
) -> dict[str, Any]:
    """Evaluate fixed configurations without retuning on any sweep cell."""
    cells = []
    skipped = skip or set()
    for reference in config["reference_circuits"]:
        for experiment_seed in config["experiment_seeds"]:
            if (reference, experiment_seed) in skipped:
                continue
            cell_config = {
                "name": config["name"],
                "seed": experiment_seed,
                "reference_circuit": reference,
                "shots": config["shots"],
                "confidence": config.get("confidence", 0.95),
                "decoders": config["decoders"],
                "latency_shots": config.get("latency_shots", 0),
                "latency_warmup": config.get("latency_warmup", 1),
            }
            for key in ("relay_bp", "bp_lsd", "bp_osd"):
                if key in config:
                    cell_config[key] = config[key]
            report = run_qldpc_experiment(cell_config)
            cell = {"experiment_seed": experiment_seed, "report": report}
            cells.append(cell)
            if on_cell is not None:
                on_cell(cell)
    return {"config": config, "cells": cells}


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
    completed = {(cell["report"]["reference"]["name"], cell["experiment_seed"]) for cell in cells}
    if completed:
        print(f"resuming with {len(completed)} checkpointed cells")

    def save_cell(cell: dict[str, Any]) -> None:
        cells.append(cell)
        with checkpoint.open("a", encoding="utf-8") as f:
            f.write(json.dumps(cell) + "\n")
        result = cell["report"]
        summary = ", ".join(
            f"{name}={row['num_errors']}/{row['shots']}" for name, row in result["decoders"].items()
        )
        print(
            f"{result['reference']['name']} seed={cell['experiment_seed']}: {summary}",
            flush=True,
        )

    run_qldpc_sweep(config, on_cell=save_cell, skip=completed)
    report = {"config": config, "cells": cells}
    path = args.out / "sweep.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
