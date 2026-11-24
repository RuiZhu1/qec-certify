"""Circuit-level noise for Stim-generated memory experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import stim

_CHANNELS = (
    "after_clifford_depolarization",
    "before_round_data_depolarization",
    "before_measure_flip_probability",
    "after_reset_flip_probability",
)


@dataclass(frozen=True)
class CircuitLevelNoise:
    """Circuit noise with optional qubit heterogeneity and drift over measurement rounds.

    ``p`` sets all four base channels unless individually overridden. ``heterogeneity`` is
    the standard deviation of a mean-one lognormal qubit strength factor. ``drift`` is the
    relative change from the first to the last round, centred on one. A two-qubit gate uses
    the mean of its qubits' factors. All factors are deterministic for ``noise_seed``.
    """

    name: ClassVar[str] = "circuit_level"
    p: float
    after_clifford_depolarization: float | None = None
    before_round_data_depolarization: float | None = None
    before_measure_flip_probability: float | None = None
    after_reset_flip_probability: float | None = None
    heterogeneity: float = 0.0
    drift: float = 0.0
    noise_seed: int = 0

    def __post_init__(self) -> None:
        for value in (self.p, *(self.stim_arguments().values())):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"circuit-level probability must be in [0, 1], got {value}")
        if not np.isfinite(self.heterogeneity) or self.heterogeneity < 0:
            raise ValueError(
                f"heterogeneity must be finite and non-negative, got {self.heterogeneity}"
            )
        if not np.isfinite(self.drift) or not 0.0 <= self.drift <= 2.0:
            raise ValueError(f"drift must be in [0, 2], got {self.drift}")
        if self.noise_seed < 0:
            raise ValueError(f"noise_seed must be non-negative, got {self.noise_seed}")

    def parameters(self) -> dict[str, float | int | str]:
        parameters: dict[str, float | int | str] = {"model": self.name, "p": self.p}
        for channel in _CHANNELS:
            value = getattr(self, channel)
            if value is not None:
                parameters[channel] = value
        if self.heterogeneity:
            parameters["heterogeneity"] = self.heterogeneity
        if self.drift:
            parameters["drift"] = self.drift
        if self.heterogeneity or self.drift:
            parameters["noise_seed"] = self.noise_seed
        return parameters

    def stim_arguments(self) -> dict[str, float]:
        return {
            channel: self.p if (value := getattr(self, channel)) is None else value
            for channel in _CHANNELS
        }

    def apply_to(self, circuit: stim.Circuit, rounds: int) -> stim.Circuit:
        """Apply qubit factors and drift to an existing Stim-generated noisy circuit."""
        if not self.heterogeneity and not self.drift:
            return circuit
        rng = np.random.default_rng(self.noise_seed)
        sigma = self.heterogeneity
        factors = rng.lognormal(mean=-(sigma**2) / 2, sigma=sigma, size=circuit.num_qubits)
        transformed = stim.Circuit()
        round_index = 0

        def append_instruction(inst: stim.CircuitInstruction) -> None:
            if inst.name not in {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR"}:
                transformed.append(inst)
                return
            qubits = [target.qubit_value for target in inst.targets_copy()]
            factor = float(factors[qubits].mean())
            if rounds > 1:
                factor *= 1.0 + self.drift * (round_index / (rounds - 1) - 0.5)
            args = inst.gate_args_copy()
            probability = args[0] * factor
            if probability > 1.0:
                raise ValueError(
                    f"scaled {inst.name} probability {probability} exceeds 1 at "
                    f"round {round_index}, qubits {qubits}; reduce p, heterogeneity or drift"
                )
            transformed.append(inst.name, inst.targets_copy(), [probability])

        for inst in circuit:
            if isinstance(inst, stim.CircuitRepeatBlock):
                for _ in range(inst.repeat_count):
                    round_index += 1
                    for body_inst in inst.body_copy():
                        append_instruction(body_inst)
            else:
                append_instruction(inst)
        return transformed
