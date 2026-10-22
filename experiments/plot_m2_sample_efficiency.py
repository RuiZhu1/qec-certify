"""Figure: gap to exact ML against training-set size, MLP next to the lookup table.

Reads one or more ``mlp_report.json`` files written by::

    python -m qec_certify.train.run --config configs/m2_mlp_d3.yaml --out runs/m2_mlp_d3

and draws one row per code distance and one panel per physical error rate. Lines are the
mean over repeats; the shaded band spans the smallest to the largest repeat. The MWPM and
correlated MWPM levels are drawn for reference; ``G = 1`` is exact ML.

Usage::

    python experiments/plot_m2_sample_efficiency.py runs/m2_mlp_d3/mlp_report.json \
        runs/m2_mlp_d5/mlp_report.json --out runs/m2_sample_efficiency.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from plotting import BASELINE, INK_SECONDARY, SERIES, apply_style, ratio_ticks

LEARNED = (("mlp", "MLP"), ("lookup", "lookup table"))
REFERENCES = (("mwpm", "MWPM", (0, (4, 2))), ("mwpm_correlated", "correlated MWPM", (0, (1, 2))))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    reports = [json.loads(path.read_text()) for path in args.reports]
    reports.sort(key=lambda r: r["config"]["code"]["distance"])
    probabilities = sorted({e["p"] for r in reports for e in r["results"]})
    apply_style()
    fig, axes = plt.subplots(
        len(reports),
        len(probabilities),
        figsize=(1.9 * len(probabilities), 2.2 * len(reports) + 0.3),
        squeeze=False,
    )
    for row, report in zip(axes, reports, strict=True):
        d = report["config"]["code"]["distance"]
        by_p = {e["p"]: e for e in report["results"]}
        for ax, p in zip(row, probabilities, strict=True):
            entry = by_p.get(p)
            ax.set_title(f"d = {d}, p = {p:g}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            if entry is None:
                ax.set_visible(False)
                continue
            ml = entry["references"]["ml_exact"]["exact_ler"]
            ax.axhline(1.0, color=BASELINE, linewidth=1.0, zorder=1, label="exact ML")
            for name, label, dash in REFERENCES:
                if name in entry["references"]:
                    level = entry["references"][name]["exact_ler"] / ml
                    ax.axhline(
                        level,
                        color=INK_SECONDARY,
                        linestyle=dash,
                        linewidth=1.0,
                        label=label,
                        zorder=2,
                    )
            for (key, label), color in zip(LEARNED, SERIES, strict=True):
                aggs = entry.get(key, [])
                if not aggs:
                    continue
                n = [a["train_shots"] for a in aggs]
                g = [a["gap_ratio"] for a in aggs]
                ax.fill_between(
                    n,
                    [s["min"] for s in g],
                    [s["max"] for s in g],
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                    zorder=3,
                )
                ax.plot(n, [s["mean"] for s in g], color=color, marker="o", label=label, zorder=4)
            ax.set_xlabel("training shots")
            ratio_ticks(ax)
        row[0].set_ylabel("LER / LER of exact ML")
    handles, labels = axes[0][0].get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i] not in dict(LEARNED).values())
    fig.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
