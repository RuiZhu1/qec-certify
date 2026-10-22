"""Evaluate the classical baseline decoders (Tier 0 MWPM, Tier 1 exact ML) from a config.

``code.distance`` and ``noise.p`` may each be a single value or a list; every
(distance, p) cell gets its own test shots, decoded by every configured decoder.

Usage::

    python -m qec_certify.eval.baselines --config configs/m1_d3_depolarizing.yaml \
        [--out runs/m1_d3_depolarizing] [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import pymatching
import stim

import qec_certify
from qec_certify.codes import code_capacity_circuit, make_code
from qec_certify.decoders import Decoder, ExactMLDecoder, MWPMDecoder, SweepMLDecoder
from qec_certify.eval.metrics import evaluate_decoder
from qec_certify.sim import sample_syndromes
from qec_certify.utils import git_commit_hash, load_config, spawn_seeds, split_seeds

DECODERS: dict[str, Callable[[stim.Circuit], Decoder]] = {
    "mwpm": MWPMDecoder.from_circuit,
    "mwpm_correlated": partial(MWPMDecoder.from_circuit, correlated=True),
    "ml_exact": ExactMLDecoder.from_circuit,
    "ml_sweep": SweepMLDecoder.from_circuit,
}
ML_DECODERS = ("ml_exact", "ml_sweep")


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def run_baselines(config: dict[str, Any], workers: int = 1) -> dict[str, Any]:
    """Sample test shots for every (distance, p) cell and evaluate each configured decoder.

    ``workers > 1`` evaluates cells in parallel processes; results do not depend on it.
    """
    noise = config["noise"]
    if noise["model"] != "depolarizing":
        raise ValueError(f"unsupported noise model {noise['model']!r}; M1 supports 'depolarizing'")
    unknown = set(config["decoders"]) - set(DECODERS)
    if unknown:
        raise ValueError(f"unknown decoders {sorted(unknown)}; choose from {sorted(DECODERS)}")

    distances = _as_list(config["code"]["distance"])
    probabilities = _as_list(noise["p"])
    # Distance k takes test seeds [k * len(p), (k + 1) * len(p)), so a single-distance config
    # keeps the seeds it had before lists of distances were allowed.
    seeds = spawn_seeds(split_seeds(config["seed"])["test"], len(distances) * len(probabilities))
    cells = [
        (config, d, p, seeds[k * len(probabilities) + j])
        for k, d in enumerate(distances)
        for j, p in enumerate(probabilities)
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            # Largest codes first, so that the slowest cells do not start last.
            order = sorted(range(len(cells)), key=lambda k: -cells[k][1])
            futures = {k: pool.submit(_run_cell, cells[k]) for k in order}
            results = [futures[k].result() for k in range(len(cells))]
    else:
        results = [_run_cell(cell) for cell in cells]

    return {
        "config": config,
        "git_commit": git_commit_hash(),
        "versions": {
            "qec_certify": qec_certify.__version__,
            "stim": stim.__version__,
            "pymatching": pymatching.__version__,
        },
        "results": results,
    }


def _run_cell(cell: tuple[dict[str, Any], int, float, int]) -> dict[str, Any]:
    config, distance, p, seed = cell
    code = make_code(config["code"]["family"], distance)
    basis = config.get("memory_basis", "Z")
    confidence = float(config.get("confidence", 0.95))
    circuit = code_capacity_circuit(code, float(p), basis)
    data = sample_syndromes(circuit, int(config["shots"]), seed)
    entry: dict[str, Any] = {"distance": distance, "p": p, "seed": seed, "decoders": {}}
    for name in config["decoders"]:
        start = time.perf_counter()
        decoder = DECODERS[name](circuit).fit()
        row: dict[str, Any] = evaluate_decoder(decoder, data, confidence).to_dict()
        seconds = time.perf_counter() - start
        if isinstance(decoder, ExactMLDecoder):
            row["exact_ler"] = decoder.optimal_logical_error_rate
        if isinstance(decoder, SweepMLDecoder):
            row["frontier"] = decoder.frontier
        entry["decoders"][name] = row
        print(
            f"[{config.get('name', '?')}] d={distance} p={p} {name}: "
            f"{row['num_errors']}/{row['shots']} in {seconds:.1f}s",
            file=sys.stderr,
            flush=True,
        )
    return entry


def format_report(report: dict[str, Any]) -> str:
    """Sampled LERs with Wilson intervals. ``G`` is the LER over the ML decoder's LER on the
    same shots (blank without an ML decoder or when it made no errors)."""
    lines = [f"{'d':>3}  {'p':>6}  {'decoder':<16} {'errors/shots':>17}  {'LER':>10}  {'G':>6}  CI"]
    for entry in report["results"]:
        ml = next((entry["decoders"][n] for n in ML_DECODERS if n in entry["decoders"]), None)
        for name, row in entry["decoders"].items():
            g = row["ler"] / ml["ler"] if ml is not None and ml["num_errors"] else None
            lines.append(
                f"{entry.get('distance', '-'):>3}  {entry['p']:>6g}  {name:<16} "
                f"{row['num_errors']:>8}/{row['shots']:<8}  {row['ler']:>10.3e}  "
                f"{'' if g is None else format(g, '.3f'):>6}  "
                f"[{row['ci_low']:.3e}, {row['ci_high']:.3e}]"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", type=Path, help="directory to write baselines.json into")
    parser.add_argument("--workers", type=int, default=1, help="parallel processes over cells")
    args = parser.parse_args(argv)

    report = run_baselines(load_config(args.config), workers=args.workers)
    print(format_report(report))
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "baselines.json"
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
