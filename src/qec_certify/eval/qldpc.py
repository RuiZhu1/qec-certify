"""Evaluate decoders on pinned public qLDPC reference circuits.

Usage::

    python -m qec_certify.eval.qldpc --config configs/qldpc_bb18.yaml \
        [--out runs/qldpc_bb18]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from importlib.metadata import version
from itertools import combinations
from pathlib import Path
from typing import Any

import ldpc
import numpy as np
import stim
from scipy.stats import binomtest

import qec_certify
from qec_certify.codes import load_reference_circuit
from qec_certify.decoders import (
    BPLSDDecoder,
    BPOSDDecoder,
    Decoder,
    FailureFallbackDecoder,
    RelayBPDecoder,
)
from qec_certify.eval.metrics import benchmark_decoder, logical_error_rate
from qec_certify.sim import sample_syndromes
from qec_certify.utils import git_commit_hash, load_config, split_seeds


def decoder_diagnostics(decoder: Decoder) -> dict[str, float | int]:
    if isinstance(decoder, FailureFallbackDecoder):
        success = decoder.last_primary_success
        diagnostics = {
            "decode_failures": int((~success).sum()),
            "decode_success_rate": float(success.mean()) if len(success) else 1.0,
            "mean_iterations": float(decoder.last_decode_iterations.mean()),
            "max_iterations": int(decoder.last_decode_iterations.max(initial=0)),
            "fallback_calls": int(decoder.last_fallback.sum()),
            "fallback_rate": float(decoder.last_fallback.mean()) if len(success) else 0.0,
            "final_decode_failures": int((~decoder.last_final_success).sum()),
            "final_decode_success_rate": float(decoder.last_final_success.mean()),
        }
        if isinstance(decoder.fallback, RelayBPDecoder) and decoder.last_fallback.any():
            diagnostics["fallback_decode_failures"] = int(
                (~decoder.fallback.last_decode_success).sum()
            )
            diagnostics["fallback_mean_iterations"] = float(
                decoder.fallback.last_decode_iterations.mean()
            )
        return diagnostics
    if not isinstance(decoder, RelayBPDecoder):
        return {}
    success = decoder.last_decode_success
    return {
        "decode_failures": int((~success).sum()),
        "decode_success_rate": float(success.mean()) if len(success) else 1.0,
        "mean_iterations": float(decoder.last_decode_iterations.mean()),
        "max_iterations": int(decoder.last_decode_iterations.max(initial=0)),
    }


def _make_decoder(name: str, circuit: stim.Circuit, config: dict[str, Any]) -> Decoder:
    if name == "bp_osd":
        return BPOSDDecoder.from_circuit(circuit, **config.get("bp_osd", {}))
    if name == "bp_lsd":
        return BPLSDDecoder.from_circuit(circuit, **config.get("bp_lsd", {}))
    if name == "relay_bp":
        options = dict(config.get("relay_bp", {}))
        if "gamma_dist_interval" in options:
            options["gamma_dist_interval"] = tuple(options["gamma_dist_interval"])
        options.setdefault("seed", int(config["seed"]))
        return RelayBPDecoder.from_circuit(circuit, **options)
    if name == "relay_bp_fallback":
        options = dict(config.get("relay_bp", {}))
        if "gamma_dist_interval" in options:
            options["gamma_dist_interval"] = tuple(options["gamma_dist_interval"])
        options.setdefault("seed", int(config["seed"]))
        primary = RelayBPDecoder.from_circuit(circuit, **options)
        fallback_name = config.get("fallback_decoder", "bp_lsd")
        if fallback_name == "relay_bp_fallback":
            raise ValueError("fallback decoder cannot itself be relay_bp_fallback")
        if fallback_name == "relay_bp":
            fallback_options = dict(config.get("fallback_relay_bp", config["relay_bp"]))
            if "gamma_dist_interval" in fallback_options:
                fallback_options["gamma_dist_interval"] = tuple(
                    fallback_options["gamma_dist_interval"]
                )
            fallback_options["seed"] = int(config["seed"]) + int(
                config.get("fallback_seed_offset", 1)
            )
            fallback = RelayBPDecoder.from_circuit(circuit, **fallback_options)
        else:
            fallback = _make_decoder(fallback_name, circuit, config)
        return FailureFallbackDecoder(primary, fallback)
    raise ValueError(
        f"unknown qLDPC decoder {name!r}; "
        "choose from ['bp_lsd', 'bp_osd', 'relay_bp', 'relay_bp_fallback']"
    )


def _pairwise_comparisons(predictions, observable_flips) -> dict[str, Any]:
    comparisons = {}
    for first, second in combinations(sorted(predictions), 2):
        first_wrong = np.any(predictions[first] != observable_flips, axis=1)
        second_wrong = np.any(predictions[second] != observable_flips, axis=1)
        regressions = int((~first_wrong & second_wrong).sum())
        corrections = int((first_wrong & ~second_wrong).sum())
        discordant = regressions + corrections
        comparisons[f"{first}__vs__{second}"] = {
            f"{first}_correct_{second}_wrong": regressions,
            f"{first}_wrong_{second}_correct": corrections,
            "prediction_disagreements": int(
                np.any(predictions[first] != predictions[second], axis=1).sum()
            ),
            "mcnemar_pvalue": (
                binomtest(min(regressions, corrections), discordant, 0.5).pvalue
                if discordant
                else 1.0
            ),
        }
    return comparisons


def run_qldpc_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Run all configured decoders on the same reference-circuit shots."""
    circuit, reference = load_reference_circuit(config["reference_circuit"])
    seed = split_seeds(config["seed"])["test"]
    data = sample_syndromes(circuit, int(config["shots"]), seed)
    confidence = float(config.get("confidence", 0.95))
    rows: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    for name in config["decoders"]:
        start = time.perf_counter()
        decoder = _make_decoder(name, circuit, config)
        predicted = decoder.predict(data.detection_events)
        predictions[name] = predicted
        row: dict[str, Any] = logical_error_rate(
            predicted, data.observable_flips, confidence
        ).to_dict()
        row.update(decoder_diagnostics(decoder))
        seconds = time.perf_counter() - start
        if config.get("latency_shots", 0):
            row["performance"] = benchmark_decoder(
                decoder,
                data.detection_events,
                shots=int(config["latency_shots"]),
                warmup_shots=int(config.get("latency_warmup", 1)),
            ).to_dict()
        rows[name] = row
        print(
            f"[{config.get('name', '?')}] {name}: {row['num_errors']}/{row['shots']} "
            f"in {seconds:.2f}s",
            file=sys.stderr,
            flush=True,
        )
    paired = None
    if {"bp_osd", "relay_bp"} <= predictions.keys():
        bp_wrong = np.any(predictions["bp_osd"] != data.observable_flips, axis=1)
        relay_wrong = np.any(predictions["relay_bp"] != data.observable_flips, axis=1)
        regressions = int((~bp_wrong & relay_wrong).sum())
        corrections = int((bp_wrong & ~relay_wrong).sum())
        discordant = regressions + corrections
        paired = {
            "bp_osd_correct_relay_bp_wrong": regressions,
            "bp_osd_wrong_relay_bp_correct": corrections,
            "prediction_disagreements": int(
                np.any(predictions["bp_osd"] != predictions["relay_bp"], axis=1).sum()
            ),
            "mcnemar_pvalue": (
                binomtest(min(regressions, corrections), discordant, 0.5).pvalue
                if discordant
                else 1.0
            ),
        }
    return {
        "config": config,
        "reference": reference,
        "paired_comparison": paired,
        "pairwise_comparisons": _pairwise_comparisons(predictions, data.observable_flips),
        "git_commit": git_commit_hash(),
        "versions": {
            "qec_certify": qec_certify.__version__,
            "stim": stim.__version__,
            "ldpc": ldpc.__version__,
            "relay-bp": version("relay-bp"),
        },
        "seed": seed,
        "decoders": rows,
    }


def format_qldpc_report(report: dict[str, Any]) -> str:
    reference = report["reference"]
    lines = [
        f"[[{reference['n']},{reference['k']},{reference['distance']}]] "
        f"rounds={reference['rounds']} p={reference['p']:g}",
        "decoder           errors/shots         LER        CI",
    ]
    for name, row in report["decoders"].items():
        lines.append(
            f"{name:<16} {row['num_errors']:>8}/{row['shots']:<8} {row['ler']:>10.3e}  "
            f"[{row['ci_low']:.3e}, {row['ci_high']:.3e}]"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", type=Path, help="directory to write qldpc.json into")
    args = parser.parse_args(argv)
    report = run_qldpc_experiment(load_config(args.config))
    print(format_qldpc_report(report))
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "qldpc.json"
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
