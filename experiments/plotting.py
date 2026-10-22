"""Shared figure style for the scripts in ``experiments/``.

Colors are the first categorical slots of the reference data-viz palette (blue, orange),
which pass the adjacent-pair colour-vision checks; references are drawn in neutral ink.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

SERIES = ("#2a78d6", "#eb6834")
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 9,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlesize": 9,
            "axes.titlecolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "legend.frameon": False,
            "legend.labelcolor": INK,
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
        }
    )


NICE_RATIOS = (1, 1.1, 1.2, 1.5, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)


def ratio_ticks(ax) -> None:
    """Plain-number ticks on a log ratio axis (1, 1.2, 1.5, 2, ...), spaced at least a sixth
    of the axis apart so that they never collide."""
    low, high = ax.get_ylim()
    min_gap = (math.log(high) - math.log(low)) / 6
    ticks: list[float] = []
    for t in NICE_RATIOS:
        if low <= t <= high and (not ticks or math.log(t / ticks[-1]) >= min_gap):
            ticks.append(t)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([f"{t:g}" for t in ticks])
