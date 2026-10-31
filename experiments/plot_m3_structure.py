"""Figure: gap of MWPM and correlated MWPM to exact ML as the noise structure varies.

Reads ``baselines.json`` files of structured-noise configs, e.g.::

    python -m qec_certify.eval.baselines --config configs/m3_biased.yaml \
        --out runs/m3_biased --workers 8

and draws one row per report and one panel per Tier 0 decoder: ``G = LER / LER_ML``
against the noise model's structure parameter (``bias`` on a log axis, ``correlation`` on
a linear one), one line per code distance, darker for larger codes. Error bars are
approximate 95% intervals that treat the two LERs as independent (conservative, see
``plot_m1_scaling.py``).

Usage::

    python experiments/plot_m3_structure.py runs/m3_biased/baselines.json \
        runs/m3_correlated/baselines.json --out runs/m3_structure.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from plot_m1_scaling import ML, ratio_interval
from plotting import BASELINE, apply_style, ratio_ticks

DECODERS = (("mwpm", "MWPM"), ("mwpm_correlated", "correlated MWPM"))
# Ordinal steps of the reference blue ramp, light to dark (steps 250 to 650).
RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")
LOG_AXIS = {"bias"}


def ramp(n: int) -> list[str]:
    """``n`` ramp colours spread over the full light-to-dark range."""
    if n == 1:
        return [RAMP[-1]]
    return [RAMP[round(i * (len(RAMP) - 1) / (n - 1))] for i in range(n)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    reports = [json.loads(path.read_text()) for path in args.reports]
    apply_style()
    fig, axes = plt.subplots(
        len(reports), len(DECODERS), figsize=(7.2, 2.6 * len(reports) + 0.3), squeeze=False
    )
    for row, report in zip(axes, reports, strict=True):
        results = report["results"]
        structure = next(k for k in results[0]["noise"] if k not in ("model", "p"))
        model = results[0]["noise"]["model"]
        distances = sorted({e["distance"] for e in results})
        for ax, (name, label) in zip(row, DECODERS, strict=True):
            ax.axhline(1.0, color=BASELINE, linewidth=1.0, zorder=1)
            for d, color in zip(distances, ramp(len(distances)), strict=True):
                entries = sorted(
                    (e for e in results if e["distance"] == d), key=lambda e: e["noise"][structure]
                )
                points = [
                    (e["noise"][structure], *ratio_interval(e["decoders"][name], e["decoders"][ML]))
                    for e in entries
                    if e["decoders"][ML]["num_errors"] and e["decoders"][name]["num_errors"]
                ]
                x, g, low, high = zip(*points, strict=True)
                ax.errorbar(
                    x,
                    g,
                    yerr=[
                        [gi - lo for gi, lo in zip(g, low, strict=True)],
                        [hi - gi for gi, hi in zip(g, high, strict=True)],
                    ],
                    color=color,
                    marker="o",
                    capsize=0,
                    elinewidth=1.0,
                    label=f"d = {d}",
                    zorder=3,
                )
            p = results[0]["p"]
            ax.set_title(f"{label}, {model} noise, p = {p:g}")
            ax.set_xlabel(structure)
            if structure in LOG_AXIS:
                ax.set_xscale("log")
            ax.set_yscale("log")
        low = min(ax.get_ylim()[0] for ax in row)
        high = max(ax.get_ylim()[1] for ax in row)
        for ax in row:
            ax.set_ylim(low, high)
            ratio_ticks(ax)
        row[0].set_ylabel("LER / LER of exact ML")
        row[-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
