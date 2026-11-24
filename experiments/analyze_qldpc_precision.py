"""Compare per-shot decisions of Relay-BP numerical implementations on the same test shots.

Usage::

    python experiments/analyze_qldpc_precision.py runs/qldpc_precision/pareto.json \
        --out runs/qldpc_precision/implementation_pairs.json
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from qec_certify.codes import load_reference_circuit
from qec_certify.eval.qldpc import _make_decoder
from qec_certify.sim import sample_syndromes
from qec_certify.utils import split_seeds


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text())
    config = report["config"]
    circuit, _ = load_reference_circuit(config["reference_circuit"])
    data = sample_syndromes(circuit, int(config["test_shots"]), split_seeds(config["seed"])["test"])
    predictions = {}
    diagnostics = {}
    for candidate in config["candidates"]:
        name = candidate["name"]
        decoder = _make_decoder(
            "relay_bp", circuit, {"seed": config["seed"], "relay_bp": candidate["relay_bp"]}
        )
        predictions[name] = decoder.predict(data.detection_events)
        diagnostics[name] = {
            "decode_failures": int((~decoder.last_decode_success).sum()),
            "mean_iterations": float(decoder.last_decode_iterations.mean()),
            "max_iterations": int(decoder.last_decode_iterations.max()),
        }
    pairs = []
    for first, second in combinations(sorted(predictions), 2):
        disagreements = int(np.any(predictions[first] != predictions[second], axis=1).sum())
        pairs.append({"first": first, "second": second, "prediction_disagreements": disagreements})
        print(f"{first} vs {second}: {disagreements}/{len(data.detection_events)} disagreements")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"diagnostics": diagnostics, "pairs": pairs}, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
