"""Paired analysis of delayed-window LER against full-history MWPM.

Usage::

    python experiments/analyze_m4_window_sweep.py runs/m4_window_sweep/baselines.json \
        --out runs/m4_window_sweep/paired_analysis.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.stats import binomtest


def analyze(report: dict) -> dict:
    results = report["results"]
    rows = []
    for distance in sorted({entry["distance"] for entry in results}):
        entries = [entry for entry in results if entry["distance"] == distance]
        full_entry = next(entry for entry in entries if entry["window_rounds"] == distance)
        for name in full_entry["decoders"]:
            full = full_entry["decoders"][name]
            for entry in entries:
                row = entry["decoders"][name]
                comparison = row["streaming_comparison"]
                regressions = comparison["offline_correct_streaming_wrong"]
                corrections = comparison["offline_wrong_streaming_correct"]
                discordant = regressions + corrections
                performance = row["streaming_performance"]
                rows.append(
                    {
                        "distance": distance,
                        "window_rounds": entry["window_rounds"],
                        "decoder": name,
                        "shots": row["shots"],
                        "full_history_ler": full["streaming_ler"]["ler"],
                        "window_ler": row["streaming_ler"]["ler"],
                        "full_correct_window_wrong": regressions,
                        "full_wrong_window_correct": corrections,
                        "mcnemar_pvalue": (
                            binomtest(min(regressions, corrections), discordant, 0.5).pvalue
                            if discordant
                            else 1.0
                        ),
                        "p99_round_latency_ms": performance["p99_round_latency_ms"],
                        "throughput_rounds_per_second": performance["throughput_rounds_per_second"],
                    }
                )
    rows.sort(key=lambda row: (row["distance"], row["decoder"], row["window_rounds"]))
    return {
        "method": "exact_two_sided_mcnemar",
        "reference": "window_rounds == distance",
        "multiple_testing": "unadjusted",
        "rows": rows,
    }


def format_analysis(analysis: dict) -> str:
    lines = [" d  w  decoder           full LER  window LER   full->W  W->full   p-value"]
    for row in analysis["rows"]:
        lines.append(
            f"{row['distance']:>2} {row['window_rounds']:>2}  {row['decoder']:<16} "
            f"{row['full_history_ler']:>8.4f}  {row['window_ler']:>10.4f}  "
            f"{row['full_correct_window_wrong']:>8}  {row['full_wrong_window_correct']:>8}  "
            f"{row['mcnemar_pvalue']:>9.2e}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    analysis = analyze(json.loads(args.report.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(analysis, indent=2) + "\n")
    print(format_analysis(analysis))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
