"""Pinned public reference circuits for qLDPC experiments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import stim

_SOURCE_COMMIT = "72341ef85cd45779e6483c338aef7ba85094eac1"
_DATA = Path(__file__).with_name("data")


@dataclass(frozen=True)
class ReferenceCircuit:
    name: str
    n: int
    k: int
    distance: int
    rounds: int
    p: float
    filename: str
    sha256: str

    @property
    def path(self) -> Path:
        return _DATA / self.filename

    def metadata(self) -> dict[str, str | int | float]:
        return {
            "name": self.name,
            "n": self.n,
            "k": self.k,
            "distance": self.distance,
            "rounds": self.rounds,
            "p": self.p,
            "sha256": self.sha256,
            "source": f"https://github.com/trmue/relay/tree/{_SOURCE_COMMIT}",
            "notice": "Public Relay-BP test data; not a supported IBM product",
        }


REFERENCE_CIRCUITS = {
    "bb18_d3_p003": ReferenceCircuit(
        "bb18_d3_p003",
        18,
        4,
        3,
        3,
        0.003,
        "bb18_d3_p003.stim",
        "3821d7f09bd931d774e777945fee0811546779830bc110d9b412465b74a63800",
    ),
    "bb144_d12_p0001": ReferenceCircuit(
        "bb144_d12_p0001",
        144,
        12,
        12,
        12,
        0.001,
        "bb144_d12_p0001.stim",
        "97ed68e3859e10ebdfd0a13c5c6abd623084ee1bd11d88287af6a67abfdd24f2",
    ),
    "bb144_d12_p0002": ReferenceCircuit(
        "bb144_d12_p0002",
        144,
        12,
        12,
        12,
        0.002,
        "bb144_d12_p0002.stim",
        "d935f128b5ea8d225f7ba523f7dab4561580e2cc6ab95942c6d15245892f5cde",
    ),
    "bb144_d12_p003": ReferenceCircuit(
        "bb144_d12_p003",
        144,
        12,
        12,
        12,
        0.003,
        "bb144_d12_p003.stim",
        "839d2fa8ce95c62e32e2d465fb646a835e184be6ceca6e4b32f046f5c506cd20",
    ),
    "bb144_d12_p0004": ReferenceCircuit(
        "bb144_d12_p0004",
        144,
        12,
        12,
        12,
        0.004,
        "bb144_d12_p0004.stim",
        "a4ad4c07875ff213ce2cd6dc978f9460f10783225e449f17c9039df0f1972e53",
    ),
    "bb144_d12_p0005": ReferenceCircuit(
        "bb144_d12_p0005",
        144,
        12,
        12,
        12,
        0.005,
        "bb144_d12_p0005.stim",
        "a75a7c020ccc28ea83549b1d742410d08d22e603b126700cea098dd5965a5a21",
    ),
}


def load_reference_circuit(name: str) -> tuple[stim.Circuit, dict[str, str | int | float]]:
    """Load a checksummed circuit and its provenance metadata."""
    try:
        reference = REFERENCE_CIRCUITS[name]
    except KeyError:
        raise ValueError(
            f"unknown reference circuit {name!r}; choose from {sorted(REFERENCE_CIRCUITS)}"
        ) from None
    payload = reference.path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != reference.sha256:
        raise ValueError(f"checksum mismatch for {reference.path}")
    return stim.Circuit(payload.decode("utf-8")), reference.metadata()
