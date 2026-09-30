"""Train MLP decoders and compare them with the MWPM and exact-ML references.

For each noise strength and each training-set size, a fresh MLP is trained and scored
against the references. Because ``d=3`` has few detectors, every decoder's LER is also
computed *exactly* (by decoding all ``2^D`` syndromes), so the gap to optimal carries no
sampling noise. Sampled test LERs with Wilson intervals are reported alongside.

Usage::

    python -m qec_certify.train.run --config configs/m2_mlp_d3.yaml \
        [--out runs/m2_mlp_d3]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pymatching
import stim
import torch

import qec_certify
from qec_certify.codes import code_capacity_circuit, make_code
from qec_certify.decoders import ExactMLDecoder, MWPMDecoder
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


def run_mlp_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Train an MLP per (noise strength, training size) and evaluate it against the baselines."""
    noise = config["noise"]
    if noise["model"] != "depolarizing":
        raise ValueError(f"unsupported noise model {noise['model']!r}; M2 supports 'depolarizing'")
    model_cfg = config.get("model", {})
    if model_cfg.get("type", "mlp") != "mlp":
        raise ValueError(f"unsupported model type {model_cfg['type']!r}; M2 supports 'mlp'")

    code = make_code(config["code"]["family"], config["code"]["distance"])
    basis = config.get("memory_basis", "Z")
    confidence = float(config.get("confidence", 0.95))
    train_sizes = sorted(int(n) for n in config["train_sizes"])
    probabilities = noise["p"] if isinstance(noise["p"], list) else [noise["p"]]
    count = len(probabilities)
    splits = split_seeds(config["seed"])
    train_seeds = spawn_seeds(splits["train"], count)
    val_seeds = spawn_seeds(splits["val"], count)
    # Same test seeds as the M1 baselines, so sampled LERs are comparable shot for shot.
    test_seeds = spawn_seeds(splits["test"], count)

    results = []
    for p, train_seed, val_seed, test_seed in zip(
        probabilities, train_seeds, val_seeds, test_seeds, strict=True
    ):
        circuit = code_capacity_circuit(code, float(p), basis)
        dem: stim.DetectorErrorModel = circuit.detector_error_model()
        train = sample_syndromes(circuit, max(train_sizes), train_seed)
        val = sample_syndromes(circuit, int(config["val_shots"]), val_seed)
        test = sample_syndromes(circuit, int(config["test_shots"]), test_seed)

        references: dict[str, Any] = {}
        ml = ExactMLDecoder.from_circuit(circuit)
        for name, decoder in (("mwpm", MWPMDecoder.from_circuit(circuit)), ("ml_exact", ml)):
            row = evaluate_decoder(decoder, test, confidence).to_dict()
            row["exact_ler"] = exact_logical_error_rate(decoder, dem)
            references[name] = row
        exact_mwpm = references["mwpm"]["exact_ler"]
        exact_ml = references["ml_exact"]["exact_ler"]

        learned = []
        for n in train_sizes:
            decoder = MLPDecoder(
                dem.num_detectors,
                dem.num_observables,
                _mlp_config(model_cfg, seed=config["seed"]),
            ).fit(_head(train, n), val)
            row = evaluate_decoder(decoder, test, confidence).to_dict()
            exact = exact_logical_error_rate(decoder, dem)
            row.update(
                train_shots=n,
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
            learned.append(row)
        results.append(
            {
                "p": p,
                "seeds": {"train": train_seed, "val": val_seed, "test": test_seed},
                "references": references,
                "mlp": learned,
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
    """Exact LERs and gap metrics. ``gap closed`` is 0 for MWPM and 1 for exact ML."""
    lines = [
        f"{'p':>7}  {'decoder':<11} {'train':>8}  {'exact LER':>10}  {'G=LER/ML':>9}  "
        f"{'gap closed':>10}  {'sampled LER [CI]'}"
    ]

    def sampled(row: dict[str, Any]) -> str:
        return f"{row['ler']:.3e} [{row['ci_low']:.3e}, {row['ci_high']:.3e}]"

    for entry in report["results"]:
        refs = entry["references"]
        ml = refs["ml_exact"]["exact_ler"]
        for name, label in (("mwpm", "MWPM"), ("ml_exact", "exact ML")):
            row = refs[name]
            lines.append(
                f"{entry['p']:>7g}  {label:<11} {'-':>8}  {row['exact_ler']:>10.3e}  "
                f"{row['exact_ler'] / ml:>9.3f}  {'-':>10}  {sampled(row)}"
            )
        for row in entry["mlp"]:
            closed = "n/a" if row["gap_closed"] is None else f"{row['gap_closed']:.3f}"
            lines.append(
                f"{entry['p']:>7g}  {'MLP':<11} {row['train_shots']:>8}  "
                f"{row['exact_ler']:>10.3e}  {row['gap_ratio']:>9.3f}  {closed:>10}  "
                f"{sampled(row)}"
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
