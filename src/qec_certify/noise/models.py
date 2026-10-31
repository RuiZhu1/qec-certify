"""Code-capacity noise models: Pauli channels on the data qubits.

Every model does two things with the same channels. :meth:`NoiseModel.append_to` writes them
into a Stim circuit (the noise layer of
:func:`~qec_certify.codes.code_capacity_circuit`), and :meth:`NoiseModel.channels` lists
them as explicit probability tables, which the noise descriptors and the brute-force checks
read. The tables index Paulis as ``x + 2 z`` per qubit (0 = I, 1 = X, 2 = Z, 3 = Y), and
``e_0 + 4 e_1 + 16 e_2 + ...`` over the channel's qubits in order.

Each model has a total strength ``p`` and at most one *structure parameter*
(:attr:`NoiseModel.structure`), the knob that RQ2 varies:

- :class:`Depolarizing`: i.i.d. single-qubit depolarizing noise, no structure parameter.
- :class:`BiasedPauli`: i.i.d. Pauli noise with bias ``eta = p_Z / (p_X + p_Y)``.
- :class:`CorrelatedPairs`: depolarizing noise plus two-qubit depolarizing errors on every
  pair of nearest-neighbour data qubits (crosstalk-like), with ``correlation`` the fraction
  of a bulk qubit's error probability that comes from pair events.
"""

from __future__ import annotations

import abc
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import stim

if TYPE_CHECKING:
    from qec_certify.codes.base import StabilizerCode

# Two-qubit depolarizing errors touch a given qubit in 12 of their 15 Paulis, and a bulk
# qubit of a square grid has 4 nearest neighbours.
_PAIR_WEIGHT_ON_QUBIT = 12 / 15
_BULK_NEIGHBOURS = 4


@dataclass(frozen=True)
class Channel:
    """A Pauli channel on ``qubits``: ``probabilities[g]`` is the chance of Pauli ``g``."""

    qubits: tuple[int, ...]
    probabilities: np.ndarray


class NoiseModel(abc.ABC):
    """A code-capacity noise layer acting on the data qubits of a code."""

    name: ClassVar[str]
    structure: ClassVar[str | None] = None
    p: float

    @abc.abstractmethod
    def append_to(self, circuit: stim.Circuit, code: StabilizerCode) -> None:
        """Append this noise layer to ``circuit``."""

    @abc.abstractmethod
    def channels(self, code: StabilizerCode) -> list[Channel]:
        """The independent Pauli channels that :meth:`append_to` applies."""

    def parameters(self) -> dict[str, float | str]:
        """``{"model": name, "p": ..., <structure parameter>: ...}``, as in the configs."""
        return {"model": self.name, **asdict(self)}


@dataclass(frozen=True)
class Depolarizing(NoiseModel):
    """``DEPOLARIZE1(p)`` on every data qubit: X, Y and Z each with probability ``p / 3``."""

    name: ClassVar[str] = "depolarizing"
    p: float

    def __post_init__(self) -> None:
        _check_range("depolarizing probability", self.p, 0.75)

    def append_to(self, circuit: stim.Circuit, code: StabilizerCode) -> None:
        circuit.append("DEPOLARIZE1", list(range(len(code.data_coords))), self.p)

    def channels(self, code: StabilizerCode) -> list[Channel]:
        table = _depolarizing_table(self.p, 1)
        return [Channel((q,), table) for q in range(len(code.data_coords))]


@dataclass(frozen=True)
class BiasedPauli(NoiseModel):
    """``PAULI_CHANNEL_1`` on every data qubit with total probability ``p`` and bias ``eta``.

    ``p_Z = p eta / (1 + eta)`` and ``p_X = p_Y = p / (2 (1 + eta))``, so ``eta = 0.5`` is
    depolarizing noise and ``eta -> inf`` is pure dephasing. A detector error model needs
    the channel as independent X, Y and Z errors; Stim refuses the circuit when no such form
    exists (e.g. ``eta = 0``, where X and Y together would make Z) rather than approximate.
    """

    name: ClassVar[str] = "biased"
    structure: ClassVar[str | None] = "bias"
    p: float
    bias: float

    def __post_init__(self) -> None:
        _check_range("biased-noise probability", self.p, 1.0)
        if not (self.bias >= 0 and math.isfinite(self.bias)):
            raise ValueError(f"bias must be finite and non-negative, got {self.bias}")

    @property
    def pauli_probabilities(self) -> tuple[float, float, float]:
        """``(p_X, p_Y, p_Z)``."""
        p_xy = self.p / (2 * (1 + self.bias))
        return p_xy, p_xy, self.p * self.bias / (1 + self.bias)

    def append_to(self, circuit: stim.Circuit, code: StabilizerCode) -> None:
        circuit.append(
            "PAULI_CHANNEL_1", list(range(len(code.data_coords))), self.pauli_probabilities
        )

    def channels(self, code: StabilizerCode) -> list[Channel]:
        p_x, p_y, p_z = self.pauli_probabilities
        table = np.array([1 - p_x - p_y - p_z, p_x, p_z, p_y])
        return [Channel((q,), table) for q in range(len(code.data_coords))]


@dataclass(frozen=True)
class CorrelatedPairs(NoiseModel):
    """Single-qubit plus nearest-neighbour two-qubit depolarizing noise.

    Every data qubit gets ``DEPOLARIZE1((1 - c) p)`` and every pair of data qubits at unit
    distance gets ``DEPOLARIZE2(5 c p / 16)``, with ``c = correlation``. A bulk qubit (four
    neighbours) is then hit by a pair event with probability ``c p`` to first order, so its
    total error probability stays about ``p`` while ``c`` moves it from independent
    (``c = 0``, the :class:`Depolarizing` circuit) to fully correlated (``c = 1``) errors.
    Boundary qubits have fewer neighbours and slightly less noise.
    """

    name: ClassVar[str] = "correlated"
    structure: ClassVar[str | None] = "correlation"
    p: float
    correlation: float

    def __post_init__(self) -> None:
        _check_range("correlated-noise probability", self.p, 0.75)
        _check_range("correlation", self.correlation, 1.0)

    @property
    def single_probability(self) -> float:
        return (1 - self.correlation) * self.p

    @property
    def pair_probability(self) -> float:
        return self.correlation * self.p / (_PAIR_WEIGHT_ON_QUBIT * _BULK_NEIGHBOURS)

    def append_to(self, circuit: stim.Circuit, code: StabilizerCode) -> None:
        circuit.append("DEPOLARIZE1", list(range(len(code.data_coords))), self.single_probability)
        if self.correlation > 0:
            pairs = [q for pair in nearest_neighbour_pairs(code) for q in pair]
            circuit.append("DEPOLARIZE2", pairs, self.pair_probability)

    def channels(self, code: StabilizerCode) -> list[Channel]:
        single = _depolarizing_table(self.single_probability, 1)
        out = [Channel((q,), single) for q in range(len(code.data_coords))]
        if self.correlation > 0:
            pair = _depolarizing_table(self.pair_probability, 2)
            out += [Channel(qubits, pair) for qubits in nearest_neighbour_pairs(code)]
        return out


NOISE_MODELS: dict[str, type[NoiseModel]] = {
    cls.name: cls for cls in (Depolarizing, BiasedPauli, CorrelatedPairs)
}


def make_noise(model: str, p: float, structure: float | None = None) -> NoiseModel:
    """Build a noise model by name; ``structure`` is its structure parameter, if it has one."""
    try:
        cls = NOISE_MODELS[model]
    except KeyError:
        raise ValueError(
            f"unknown noise model {model!r}; choose from {sorted(NOISE_MODELS)}"
        ) from None
    if cls.structure is None:
        if structure is not None:
            raise ValueError(f"noise model {model!r} has no structure parameter")
        return cls(p)
    if structure is None:
        raise ValueError(f"noise model {model!r} needs its {cls.structure!r} parameter")
    return cls(p, structure)


def noise_grid(config: dict) -> list[NoiseModel]:
    """All noise models of a config's ``noise`` section: every ``p`` (outer) times every value
    of the structure parameter (inner). Scalars count as one-element lists."""
    model = config["model"]
    if model not in NOISE_MODELS:
        raise ValueError(f"unknown noise model {model!r}; choose from {sorted(NOISE_MODELS)}")
    structure = NOISE_MODELS[model].structure
    if structure is not None and structure not in config:
        raise ValueError(f"noise model {model!r} needs its {structure!r} parameter")
    values = [None] if structure is None else _as_list(config[structure])
    return [make_noise(model, float(p), v) for p in _as_list(config["p"]) for v in values]


def as_noise_model(noise: float | NoiseModel) -> NoiseModel:
    """A noise model, with a bare probability meaning :class:`Depolarizing`."""
    return noise if isinstance(noise, NoiseModel) else Depolarizing(float(noise))


def nearest_neighbour_pairs(code: StabilizerCode) -> tuple[tuple[int, int], ...]:
    """Pairs ``(a, b)``, ``a < b``, of data qubits at unit distance (grid neighbours)."""
    coords = np.array(code.data_coords, dtype=float)
    dist = np.abs(coords[:, None, :] - coords[None, :, :]).sum(axis=2)
    a, b = np.nonzero(np.triu(np.isclose(dist, 1.0), k=1))
    return tuple(zip(a.tolist(), b.tolist(), strict=True))


def _depolarizing_table(p: float, num_qubits: int) -> np.ndarray:
    size = 4**num_qubits
    table = np.full(size, p / (size - 1))
    table[0] = 1 - p
    return table


def _check_range(what: str, value: float, high: float) -> None:
    if not 0.0 <= value <= high:
        raise ValueError(f"{what} must be in [0, {high:g}], got {value}")


def _as_list(value):
    return value if isinstance(value, list) else [value]
