"""Figure: gap of MWPM and correlated MWPM to exact ML as the code distance grows.

Reads the ``baselines.json`` written by::

    python -m qec_certify.eval.baselines --config configs/m1_scaling_depolarizing.yaml \
        --out runs/m1_scaling_depolarizing --workers 8

and plots ``G = LER / LER_ML`` against ``d`` on a log scale, one panel per physical error
rate. Error bars are approximate 95% intervals that treat the two LERs as independent; the
decoders see the same shots, so these are conservative.

Usage::

    python experiments/plot_m1_scaling.py runs/m1_scaling_depolarizing/baselines.json \
        --out runs/m1_scaling_depolarizing/gap_vs_distance.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from plotting import BASELINE, SERIES, apply_style, ratio_ticks

DECODERS = (("mwpm", "MWPM"), ("mwpm_correlated", "correlated MWPM"))
ML = "ml_sweep"
Z95 = 1.959964


def ratio_interval(row: dict, ml: dict) -> tuple[float, float, float]:
    """``G`` and an approximate 95% interval from the delta method on ``log G``.

    The two LERs are treated as independent. Both decoders see the same shots and tend to
    fail on the same ones, so the true interval is narrower: this one is conservative.
    """
    g = row["ler"] / ml["ler"]
    var = sum((1 - r["ler"]) / r["num_errors"] for r in (row, ml))
    half = Z95 * math.sqrt(var)
    return g, g * math.exp(-half), g * math.exp(half)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    results = json.loads(args.report.read_text())["results"]
    probabilities = sorted({entry["p"] for entry in results})
    apply_style()
    fig, axes = plt.subplots(
        1, len(probabilities), figsize=(1.9 * len(probabilities), 2.4), sharey=True
    )
    for ax, p in zip(axes, probabilities, strict=True):
        entries = sorted((e for e in results if e["p"] == p), key=lambda e: e["distance"])
        ax.axhline(1.0, color=BASELINE, linewidth=1.0, zorder=1, label="exact ML")
        for (name, label), color in zip(DECODERS, SERIES, strict=True):
            points = [
                (e["distance"], *ratio_interval(e["decoders"][name], e["decoders"][ML]))
                for e in entries
                if e["decoders"][ML]["num_errors"] and e["decoders"][name]["num_errors"]
            ]
            d, g, low, high = zip(*points, strict=True)
            ax.errorbar(
                d,
                g,
                yerr=[
                    [gi - lo for gi, lo in zip(g, low, strict=True)],
                    [hi - gi for gi, hi in zip(g, high, strict=True)],
                ],
                color=color,
                marker="o",
                capsize=0,
                elinewidth=1.0,
                label=label,
                zorder=3,
            )
        ax.set_title(f"p = {p:g}")
        ax.set_xticks([e["distance"] for e in entries])
        ax.set_xlabel("distance d")
        # Log scale: below threshold the gap grows geometrically with d, a straight line here.
        ax.set_yscale("log")
    for ax in axes:
        ratio_ticks(ax)
    axes[0].set_ylabel("LER / LER of exact ML")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=len(DECODERS) + 1, bbox_to_anchor=(0.5, 1.02)
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
