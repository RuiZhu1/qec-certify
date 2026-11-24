"""Evaluate the classical baseline decoders (Tier 0 MWPM, Tier 1 exact ML) from a config.

``code.distance`` may be a single value or a list, and so may ``noise.p`` and the noise
model's structure parameter (``noise.bias``, ``noise.correlation``; see
:func:`qec_certify.noise.noise_grid`). ``noise.model: circuit_level`` selects Stim's
multi-round rotated-surface memory and requires ``rounds``; other models select the original
code-capacity circuit. Every (distance, noise model) cell gets its own test shots, decoded by
every configured decoder. ``shots``, ``rounds`` and per-distance values may be mappings.
Optional ``latency_shots`` measures offline batch-of-one predictions;
``streaming_latency_shots`` measures causal per-round decoding with optional fixed-rate
arrivals. ``streaming_method`` selects ``causal_replay``, MWPM ``sliding_window`` or
``delayed_commit``; window methods use ``window_rounds`` and report approximate final LERs
separately. Window grids may be per-distance lists and reuse shots for paired comparisons.

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

import ldpc
import numpy as np
import pymatching
import stim

import qec_certify
from qec_certify.codes import circuit_level_circuit, code_capacity_circuit, make_code
from qec_certify.decoders import (
    BPOSDDecoder,
    CausalReplayDecoder,
    Decoder,
    DelayedCommitMWPMDecoder,
    ExactMLDecoder,
    MWPMDecoder,
    SlidingWindowMWPMDecoder,
    SweepMLDecoder,
    UnionFindDecoder,
)
from qec_certify.eval.metrics import benchmark_decoder, evaluate_decoder, logical_error_rate
from qec_certify.eval.streaming import benchmark_streaming_decoder, streaming_predictions
from qec_certify.noise import CircuitLevelNoise, ExperimentNoise, describe, noise_grid
from qec_certify.sim import sample_syndromes
from qec_certify.utils import git_commit_hash, load_config, spawn_seeds, split_seeds

DECODERS: dict[str, Callable[[stim.Circuit], Decoder]] = {
    "mwpm": MWPMDecoder.from_circuit,
    "mwpm_correlated": partial(MWPMDecoder.from_circuit, correlated=True),
    "bp_osd": BPOSDDecoder.from_circuit,
    "union_find": UnionFindDecoder.from_circuit,
    "ml_exact": ExactMLDecoder.from_circuit,
    "ml_sweep": SweepMLDecoder.from_circuit,
}
ML_DECODERS = ("ml_exact", "ml_sweep")


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _window_grid(config: dict[str, Any], distance: int) -> list[int | None]:
    method = config.get("streaming_method", "causal_replay")
    value = config.get("window_rounds")
    if value is None:
        return [None]
    if method not in {"sliding_window", "delayed_commit"}:
        if isinstance(value, (list, dict)):
            raise ValueError("window_rounds grids require a window streaming method")
        return [None]
    if isinstance(value, dict):
        value = value[distance] if distance in value else value[str(distance)]
    return [int(v) for v in _as_list(value)]


def run_baselines(config: dict[str, Any], workers: int = 1) -> dict[str, Any]:
    """Sample test shots for every (distance, noise model) cell and evaluate each decoder.

    ``workers > 1`` evaluates cells in parallel processes; results do not depend on it.
    """
    unknown = set(config["decoders"]) - set(DECODERS)
    if unknown:
        raise ValueError(f"unknown decoders {sorted(unknown)}; choose from {sorted(DECODERS)}")

    distances = _as_list(config["code"]["distance"])
    noises = noise_grid(config["noise"])
    # Cell (distance k, noise j) takes test seed k * len(noises) + j. Depolarizing noise has
    # one model per p, so its configs keep the seeds they had before structured noise.
    seeds = spawn_seeds(split_seeds(config["seed"])["test"], len(distances) * len(noises))
    cells = [
        (config, d, noise, seeds[k * len(noises) + j], window)
        for k, d in enumerate(distances)
        for j, noise in enumerate(noises)
        for window in _window_grid(config, d)
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
            "ldpc": ldpc.__version__,
        },
        "results": results,
    }


def _shots(config: dict[str, Any], distance: int) -> int:
    shots = config["shots"]
    if not isinstance(shots, dict):
        return int(shots)
    # YAML keeps integer keys; a JSON config turns them into strings.
    return int(shots[distance] if distance in shots else shots[str(distance)])


def _rounds(config: dict[str, Any], distance: int) -> int:
    rounds = config["rounds"]
    if not isinstance(rounds, dict):
        return int(rounds)
    return int(rounds[distance] if distance in rounds else rounds[str(distance)])


def _run_cell(
    cell: tuple[dict[str, Any], int, ExperimentNoise, int, int | None],
) -> dict[str, Any]:
    config, distance, noise, seed, window_rounds = cell
    code = make_code(config["code"]["family"], distance)
    basis = config.get("memory_basis", "Z")
    confidence = float(config.get("confidence", 0.95))
    is_circuit_level = isinstance(noise, CircuitLevelNoise)
    rounds = _rounds(config, distance) if is_circuit_level else None
    circuit = (
        circuit_level_circuit(code, noise, rounds, basis)
        if rounds is not None
        else code_capacity_circuit(code, noise, basis)
    )
    data = sample_syndromes(circuit, _shots(config, distance), seed)
    entry: dict[str, Any] = {
        "experiment": "circuit_level" if is_circuit_level else "code_capacity",
        "distance": distance,
        "p": noise.p,
        "noise": noise.parameters(),
        "seed": seed,
        "decoders": {},
    }
    if rounds is not None:
        entry["rounds"] = rounds
    else:
        entry["descriptors"] = describe(noise, code)
    if window_rounds is not None:
        entry["window_rounds"] = window_rounds
    label = " ".join(f"{k}={v}" for k, v in noise.parameters().items() if k != "model")
    for name in config["decoders"]:
        start = time.perf_counter()
        decoder = DECODERS[name](circuit).fit()
        row: dict[str, Any] = evaluate_decoder(decoder, data, confidence).to_dict()
        seconds = time.perf_counter() - start
        if rounds is not None:
            row["per_round_model"] = "independent_equal_hazard"
            for key in ("ler", "ci_low", "ci_high"):
                per_round_key = "ler_per_round" if key == "ler" else f"ler_per_round_{key}"
                row[per_round_key] = 1.0 - (1.0 - row[key]) ** (1.0 / rounds)
        if config.get("latency_shots", 0):
            row["performance"] = benchmark_decoder(
                decoder,
                data.detection_events,
                shots=int(config["latency_shots"]),
                warmup_shots=int(config.get("latency_warmup", 1)),
            ).to_dict()
        if config.get("streaming_latency_shots", 0):
            if rounds is None:
                raise ValueError("streaming latency requires a circuit-level experiment")
            streaming = None
            if isinstance(decoder, UnionFindDecoder):
                row["streaming_performance"] = {"mode": "unsupported_inconsistent_prefixes"}
            else:
                method = config.get("streaming_method", "causal_replay")
                if method == "causal_replay":
                    streaming = CausalReplayDecoder(decoder, rounds=rounds)
                elif method in {"sliding_window", "delayed_commit"}:
                    if isinstance(decoder, MWPMDecoder):
                        window_class = (
                            SlidingWindowMWPMDecoder
                            if method == "sliding_window"
                            else DelayedCommitMWPMDecoder
                        )
                        streaming = window_class.from_circuit(
                            circuit,
                            rounds=rounds,
                            window_rounds=window_rounds or rounds,
                            correlated=decoder.correlated,
                        )
                        row["streaming_approximation"] = streaming.approximation
                        predicted = streaming_predictions(streaming, data)
                        row["streaming_ler"] = logical_error_rate(
                            predicted, data.observable_flips, confidence
                        ).to_dict()
                        offline = decoder.predict(data.detection_events)
                        offline_wrong = np.any(offline != data.observable_flips, axis=1)
                        streaming_wrong = np.any(predicted != data.observable_flips, axis=1)
                        row["streaming_comparison"] = {
                            "prediction_disagreements": int(
                                np.any(predicted != offline, axis=1).sum()
                            ),
                            "offline_correct_streaming_wrong": int(
                                (~offline_wrong & streaming_wrong).sum()
                            ),
                            "offline_wrong_streaming_correct": int(
                                (offline_wrong & ~streaming_wrong).sum()
                            ),
                        }
                    else:
                        row["streaming_performance"] = {
                            "mode": "unsupported_sliding_window_decoder"
                        }
                else:
                    raise ValueError(f"unknown streaming method {method!r}")
            if streaming is not None:
                period = config.get("round_period_us")
                row["streaming_performance"] = benchmark_streaming_decoder(
                    streaming,
                    data.detection_events,
                    shots=int(config["streaming_latency_shots"]),
                    warmup_shots=int(config.get("streaming_warmup", 1)),
                    round_period_us=float(period) if period is not None else None,
                ).to_dict()
        if isinstance(decoder, ExactMLDecoder):
            row["exact_ler"] = decoder.optimal_logical_error_rate
        if isinstance(decoder, SweepMLDecoder):
            row["frontier"] = decoder.frontier
        entry["decoders"][name] = row
        print(
            f"[{config.get('name', '?')}] d={distance} {label} {name}: "
            f"{row['num_errors']}/{row['shots']} in {seconds:.1f}s",
            file=sys.stderr,
            flush=True,
        )
    return entry


def format_report(report: dict[str, Any]) -> str:
    """Sampled LERs with Wilson intervals. ``G`` is the LER over the ML decoder's LER on the
    same shots (blank without an ML decoder or when it made no errors)."""
    lines = [
        f"{'d':>3}  {'w':>3}  {'p':>6}  {'structure':<12} {'decoder':<16} {'errors/shots':>17}  "
        f"{'LER':>10}  {'G':>6}  CI"
    ]
    for entry in report["results"]:
        ml = next((entry["decoders"][n] for n in ML_DECODERS if n in entry["decoders"]), None)
        structure = " ".join(
            f"{k}={v:g}" for k, v in entry.get("noise", {}).items() if k not in ("model", "p")
        )
        for name, row in entry["decoders"].items():
            g = row["ler"] / ml["ler"] if ml is not None and ml["num_errors"] else None
            lines.append(
                f"{entry.get('distance', '-'):>3}  {entry.get('window_rounds', '-'):>3}  "
                f"{entry['p']:>6g}  {structure:<12} {name:<16} "
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
