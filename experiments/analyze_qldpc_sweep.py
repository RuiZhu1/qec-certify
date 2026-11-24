"""Summarize qLDPC robustness cells and pool shots by noise strength and decoder.

Usage::

    python experiments/analyze_qldpc_sweep.py runs/qldpc_robustness/sweep.json \
        --out runs/qldpc_robustness/analysis.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from qec_certify.eval.metrics import wilson_interval


def analyze(sweep: dict) -> dict:
    cells = []
    pooled = defaultdict(lambda: {"errors": 0, "shots": 0})
    for cell in sweep["cells"]:
        report = cell["report"]
        p = report["reference"]["p"]
        for name, row in report["decoders"].items():
            pooled[p, name]["errors"] += row["num_errors"]
            pooled[p, name]["shots"] += row["shots"]
        comparisons = {
            key: {
                "mcnemar_pvalue": value["mcnemar_pvalue"],
                "prediction_disagreements": value["prediction_disagreements"],
            }
            for key, value in report["pairwise_comparisons"].items()
        }
        cells.append(
            {
                "p": p,
                "experiment_seed": cell["experiment_seed"],
                "decoders": {
                    name: {
                        "errors": row["num_errors"],
                        "shots": row["shots"],
                        "ler": row["ler"],
                        "mean_latency_ms": row["performance"]["mean_latency_ms"],
                        "p99_latency_ms": row["performance"]["p99_latency_ms"],
                    }
                    for name, row in report["decoders"].items()
                },
                "pairwise_comparisons": comparisons,
            }
        )
    pooled_rows = []
    for (p, name), counts in sorted(pooled.items()):
        low, high = wilson_interval(counts["errors"], counts["shots"])
        pooled_rows.append(
            {
                "p": p,
                "decoder": name,
                **counts,
                "ler": counts["errors"] / counts["shots"],
                "ci_low": low,
                "ci_high": high,
            }
        )
    return {
        "pooling": "descriptive pooling across two independent experiment seeds",
        "paired_tests": "computed per cell, not pooled",
        "cells": cells,
        "pooled": pooled_rows,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    analysis = analyze(json.loads(args.report.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(analysis, indent=2) + "\n")
    for cell in analysis["cells"]:
        summary = ", ".join(
            f"{name}={row['errors']}/{row['shots']}" for name, row in cell["decoders"].items()
        )
        print(f"p={cell['p']:g} seed={cell['experiment_seed']}: {summary}")
    print("pooled:")
    for row in analysis["pooled"]:
        print(
            f"p={row['p']:g} {row['decoder']}: {row['errors']}/{row['shots']} "
            f"LER={row['ler']:.4g} [{row['ci_low']:.3g}, {row['ci_high']:.3g}]"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
