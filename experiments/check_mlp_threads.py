"""Check what makes MLP training reproducible: PyTorch's thread count, not machine load.

Part 1 trains one MLP (d=3, p = 0.1, 1e5 shots, the M2 settings) several times and prints a
hash of the trained weights: twice with the same thread count, with other thread counts,
and once while busy processes load every CPU.

Part 2 (``--report``) re-trains repeat 0 of two cells of an M2 run and compares the
per-epoch validation losses with the ones logged in its ``mlp_report.json``, once with the
thread count the run used and once with a single thread.

Usage::

    python experiments/check_mlp_threads.py [--report runs/m2_mlp_d3/mlp_report.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import time
from pathlib import Path

import torch

from qec_certify.codes import code_capacity_circuit, make_code
from qec_certify.decoders.nn import MLPConfig, MLPDecoder
from qec_certify.sim import sample_syndromes
from qec_certify.train.run import _head, _mlp_config
from qec_certify.utils import spawn_seeds, split_seeds

REPRODUCE_CELLS = ((0.1, 100_000), (0.01, 1000))


def weights_hash(num_threads: int) -> str:
    circuit = code_capacity_circuit(make_code("rotated_surface", 3), 0.1, "Z")
    dem = circuit.detector_error_model()
    train = sample_syndromes(circuit, 100_000, seed=11)
    val = sample_syndromes(circuit, 100_000, seed=12)
    config = MLPConfig(hidden_sizes=(64, 64), min_steps=4000, seed=0, num_threads=num_threads)
    decoder = MLPDecoder(dem.num_detectors, dem.num_observables, config).fit(train, val)
    digest = hashlib.sha256()
    for tensor in decoder.model.state_dict().values():
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()[:16]


def _busy() -> None:
    while True:
        pass


def check_threads() -> None:
    default = torch.get_num_threads()
    runs = [(default, "idle"), (default, "idle"), (1, "idle"), (4, "idle")]
    for threads, load in runs:
        start = time.perf_counter()
        print(
            f"threads={threads:>2} {load}: weights {weights_hash(threads)} "
            f"({time.perf_counter() - start:.1f}s)",
            flush=True,
        )
    hogs = [multiprocessing.Process(target=_busy, daemon=True) for _ in range(os.cpu_count() or 1)]
    for hog in hogs:
        hog.start()
    try:
        for threads in (default, 1):
            start = time.perf_counter()
            print(
                f"threads={threads:>2} all CPUs busy: weights {weights_hash(threads)} "
                f"({time.perf_counter() - start:.1f}s)",
                flush=True,
            )
    finally:
        for hog in hogs:
            hog.terminate()


def check_report(path: Path, run_threads: int) -> None:
    report = json.loads(path.read_text())
    config = report["config"]
    probabilities = config["noise"]["p"]
    count, repeats = len(probabilities), int(config.get("repeats", 1))
    splits = split_seeds(config["seed"])
    train_seeds = spawn_seeds(splits["train"], count * repeats)
    val_seeds = spawn_seeds(splits["val"], count)
    code = make_code(config["code"]["family"], config["code"]["distance"])
    for p, n in REPRODUCE_CELLS:
        j = probabilities.index(p)
        circuit = code_capacity_circuit(code, p, config.get("memory_basis", "Z"))
        dem = circuit.detector_error_model()
        val = sample_syndromes(circuit, int(config["val_shots"]), val_seeds[j])
        train = sample_syndromes(circuit, max(config["train_sizes"]), train_seeds[j])
        aggregate = next(a for a in report["results"][j]["mlp"] if a["train_shots"] == n)
        logged = [h["val_loss"] for h in aggregate["runs"][0]["history"]]
        for threads in (run_threads, 1):
            cfg = _mlp_config({**config["model"], "num_threads": threads}, seed=config["seed"])
            decoder = MLPDecoder(dem.num_detectors, dem.num_observables, cfg)
            decoder.fit(_head(train, n), val)
            same = [h["val_loss"] for h in decoder.history] == logged
            print(
                f"p={p} train={n} threads={threads:>2}: validation losses identical to "
                f"{path}: {same}",
                flush=True,
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, help="mlp_report.json of an M2 run to re-check")
    parser.add_argument(
        "--run-threads",
        type=int,
        default=torch.get_num_threads(),
        help="thread count the report was produced with (default: this machine's)",
    )
    args = parser.parse_args(argv)
    check_threads()
    if args.report is not None:
        check_report(args.report, args.run_threads)


if __name__ == "__main__":
    main()
