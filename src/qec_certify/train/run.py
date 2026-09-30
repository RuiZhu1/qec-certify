"""Train MLP decoders and compare them with the MWPM and exact-ML references.

For each noise strength, training-set size and repeat, a fresh MLP is trained and scored
against the references. Repeat ``i`` uses its own training shots and its own network seed
(``config seed + i``), so the spread over repeats covers both data and initialisation.
Validation and test shots are shared by all repeats; the test shots are the M1 baseline
ones. Because ``d=3`` and ``d=5`` have few detectors, every decoder's LER is also computed
*exactly* (by decoding all ``2^D`` syndromes), so the gap to optimal carries no sampling
noise. Sampled test LERs with Wilson intervals are reported alongside.

Usage::

    python -m qec_certify.train.run --config configs/m2_mlp_d3.yaml \
        [--out runs/m2_mlp_d3]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import pymatching
import stim
import torch

import qec_certify
from qec_certify.codes import code_capacity_circuit, make_code
from qec_certify.decoders import ExactMLDecoder, MWPMDecoder
from qec_certify.decoders.ml_exact import DEFAULT_MAX_BITS, syndrome_class_distribution
from qec_certify.decoders.nn import MLPConfig, MLPDecoder
from qec_certify.eval.exact import exact_logical_error_rate
from qec_certify.eval.metrics import evaluate_decoder, gap_closed, gap_ratio
from qec_certify.sim import SyndromeData, sample_syndromes
from qec_certify.utils import git_commit_hash, load_config, spawn_seeds, split_seeds

GAP_TOLERANCE = 1e-12


def _mlp_config(model: dict[str, Any], seed: int) -> MLPConfig:
    fields = {k: v for k, v in model.items() if k != "type"}
    if "hidden_sizes" in fields:
        fields["hidden_sizes"] = tuple(fields["hidden_sizes"])
    return MLPConfig(**fields, seed=seed)


def _head(data: SyndromeData, n: int) -> SyndromeData:
    return SyndromeData(data.detection_events[:n], data.observable_flips[:n])


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    closed = [row["gap_closed"] for row in rows if row["gap_closed"] is not None]
    return {
        "train_shots": rows[0]["train_shots"],
        "repeats": len(rows),
        "exact_ler": _summary([row["exact_ler"] for row in rows]),
        "gap_ratio": _summary([row["gap_ratio"] for row in rows]),
        "gap_closed": _summary(closed) if len(closed) == len(rows) else None,
        "runs": rows,
    }


def run_mlp_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Train MLPs per (noise strength, training size, repeat) and score them against baselines."""
    noise = config["noise"]
    if noise["model"] != "depolarizing":
        raise ValueError(f"unsupported noise model {noise['model']!r}; M2 supports 'depolarizing'")
    model_cfg = config.get("model", {})
    if model_cfg.get("type", "mlp") != "mlp":
        raise ValueError(f"unsupported model type {model_cfg['type']!r}; M2 supports 'mlp'")

    code = make_code(config["code"]["family"], config["code"]["distance"])
    basis = config.get("memory_basis", "Z")
    confidence = float(config.get("confidence", 0.95))
    max_bits = int(config.get("max_bits", DEFAULT_MAX_BITS))
    train_sizes = sorted(int(n) for n in config["train_sizes"])
    repeats = int(config.get("repeats", 1))
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    probabilities = noise["p"] if isinstance(noise["p"], list) else [noise["p"]]
    count = len(probabilities)
    splits = split_seeds(config["seed"])
    # Repeat 0 keeps the seeds of a single-repeat run: children of a SeedSequence do not
    # depend on how many are spawned, and repeat r takes children [r * count, (r + 1) * count).
    train_seeds = spawn_seeds(splits["train"], count * repeats)
    val_seeds = spawn_seeds(splits["val"], count)
    # Same test seeds as the M1 baselines, so sampled LERs are comparable shot for shot.
    test_seeds = spawn_seeds(splits["test"], count)

    results = []
    for j, p in enumerate(probabilities):
        circuit = code_capacity_circuit(code, float(p), basis)
        dem: stim.DetectorErrorModel = circuit.detector_error_model()
        joint = syndrome_class_distribution(dem, max_bits=max_bits)
        val = sample_syndromes(circuit, int(config["val_shots"]), val_seeds[j])
        test = sample_syndromes(circuit, int(config["test_shots"]), test_seeds[j])

        references: dict[str, Any] = {}
        decoders = {
            "mwpm": MWPMDecoder.from_circuit(circuit),
            "ml_exact": ExactMLDecoder.from_circuit(circuit, max_bits=max_bits),
        }
        for name, decoder in decoders.items():
            row = evaluate_decoder(decoder, test, confidence).to_dict()
            row["exact_ler"] = exact_logical_error_rate(decoder, dem, joint=joint)
            references[name] = row
        exact_mwpm = references["mwpm"]["exact_ler"]
        exact_ml = references["ml_exact"]["exact_ler"]

        runs: dict[int, list[dict[str, Any]]] = {n: [] for n in train_sizes}
        for repeat in range(repeats):
            train = sample_syndromes(circuit, max(train_sizes), train_seeds[repeat * count + j])
            for n in train_sizes:
                decoder = MLPDecoder(
                    dem.num_detectors,
                    dem.num_observables,
                    _mlp_config(model_cfg, seed=config["seed"] + repeat),
                ).fit(_head(train, n), val)
                row = evaluate_decoder(decoder, test, confidence).to_dict()
                exact = exact_logical_error_rate(decoder, dem, joint=joint)
                row.update(
                    train_shots=n,
                    repeat=repeat,
                    exact_ler=exact,
                    gap_ratio=gap_ratio(exact, exact_ml),
                    gap_closed=(
                        gap_closed(exact_mwpm, exact, exact_ml)
                        if abs(exact_mwpm - exact_ml) > GAP_TOLERANCE
                        else None
                    ),
                    best_epoch=min(
                        decoder.history,
                        key=lambda h: float("inf") if h["val_loss"] is None else h["val_loss"],
                    )["epoch"],
                    history=decoder.history,
                )
                runs[n].append(row)
                print(
                    f"[{config.get('name', '?')}] p={p} repeat={repeat + 1}/{repeats} "
                    f"train={n}: exact G={row['gap_ratio']:.3f} "
                    f"(MWPM {exact_mwpm / exact_ml:.3f}, {time.strftime('%H:%M:%S')})",
                    file=sys.stderr,
                    flush=True,
                )
        results.append(
            {
                "p": p,
                "seeds": {
                    "train": train_seeds[j::count],
                    "val": val_seeds[j],
                    "test": test_seeds[j],
                },
                "references": references,
                "mlp": [_aggregate(runs[n]) for n in train_sizes],
            }
        )

    return {
        "config": config,
        "git_commit": git_commit_hash(),
        "versions": {
            "qec_certify": qec_certify.__version__,
            "stim": stim.__version__,
            "pymatching": pymatching.__version__,
            "torch": torch.__version__,
        },
        "results": results,
    }


def format_report(report: dict[str, Any]) -> str:
    """Exact LERs and gap metrics. ``gap closed`` is 0 for MWPM and 1 for exact ML.

    MLP rows show the mean over repeats; ``G`` also gives the standard deviation and range.
    """
    lines = [
        f"{'p':>7}  {'decoder':<9} {'train':>8}  {'exact LER':>10}  "
        f"{'G = LER / LER_ML  (mean +- std, [min, max])':<46}  {'gap closed':>10}"
    ]
    for entry in report["results"]:
        refs = entry["references"]
        ml = refs["ml_exact"]["exact_ler"]
        for name, label in (("mwpm", "MWPM"), ("ml_exact", "exact ML")):
            exact = refs[name]["exact_ler"]
            lines.append(
                f"{entry['p']:>7g}  {label:<9} {'-':>8}  {exact:>10.3e}  "
                f"{exact / ml:<46.3f}  {'-':>10}"
            )
        for agg in entry["mlp"]:
            g = agg["gap_ratio"]
            closed = agg["gap_closed"]
            spread = f"{g['mean']:.3f} +- {g['std']:.3f}  [{g['min']:.3f}, {g['max']:.3f}]"
            lines.append(
                f"{entry['p']:>7g}  {'MLP':<9} {agg['train_shots']:>8}  "
                f"{agg['exact_ler']['mean']:>10.3e}  {spread:<46}  "
                f"{'n/a' if closed is None else format(closed['mean'], '.3f'):>10}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", type=Path, help="directory to write mlp_report.json into")
    args = parser.parse_args(argv)

    report = run_mlp_experiment(load_config(args.config))
    print(format_report(report))
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "mlp_report.json"
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
