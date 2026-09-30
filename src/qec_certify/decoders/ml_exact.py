"""Exact maximum-likelihood decoding for small codes.

A detector error model (DEM) is a list of independent error mechanisms; mechanism ``i``
fires with probability ``p_i`` and flips a fixed set of detectors and observables (its
*symptom*). Encode a symptom as a bitmask ``s + (l << D)``, where ``s`` holds the ``D``
detector bits and ``l`` the ``L`` observable bits. The exact joint distribution
``P(s, l)`` is the sum, over all ``2^M`` error configurations, of the configuration's
probability, grouped by the XOR of the fired symptoms.

:func:`enumerate_syndrome_class_distribution` performs that sum literally.
:func:`syndrome_class_distribution` computes the same numbers mechanism by mechanism:
``P <- (1 - p_i) P + p_i P[· XOR mask_i]``, which costs ``O(M 2^(D+L))`` instead of
``O(M 2^M)``. The ML decoder then predicts ``argmax_l P(s, l)`` for every syndrome ``s``.

:func:`pauli_bruteforce_distribution` recomputes ``P(s, l)`` from first principles for
code-capacity depolarizing noise (all ``4^n`` Pauli errors), independently of Stim.
"""

from __future__ import annotations

import numpy as np
import stim

from qec_certify.codes import StabilizerCode
from qec_certify.decoders.base import Decoder
from qec_certify.utils.bits import pack_bits, unpack_bits

DEFAULT_MAX_BITS = 26


def error_mechanisms(dem: stim.DetectorErrorModel) -> tuple[np.ndarray, list[int]]:
    """Independent error mechanisms of ``dem`` as ``(probabilities, symptom bitmasks)``.

    Decomposition separators (``^``) are ignored: a mechanism's symptom is the XOR of all
    its components. Mechanisms with zero probability or an empty symptom are dropped.
    """
    num_det = dem.num_detectors
    probs: list[float] = []
    masks: list[int] = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        mask = 0
        for target in inst.targets_copy():
            if target.is_relative_detector_id():
                mask ^= 1 << target.val
            elif target.is_logical_observable_id():
                mask ^= 1 << (num_det + target.val)
        if p > 0 and mask:
            probs.append(p)
            masks.append(mask)
    return np.array(probs, dtype=np.float64), masks


def syndrome_class_distribution(
    dem: stim.DetectorErrorModel, *, max_bits: int = DEFAULT_MAX_BITS
) -> np.ndarray:
    """Exact joint distribution of detection events and observable flips.

    Returns:
        Array ``P`` of shape ``(2^L, 2^D)`` with ``P[l, s] = Pr[observables = l, detectors = s]``,
        both packed little-endian (detector ``j`` is bit ``j`` of ``s``).
    """
    num_det, num_obs = dem.num_detectors, dem.num_observables
    n = num_det + num_obs
    _check_bits(n, max_bits)
    probs, masks = error_mechanisms(dem)

    # dist has one axis per bit; flat C-order index bit b lives on axis n - 1 - b, so
    # XOR-ing the index with a mask is a flip along the mask's axes.
    dist = np.zeros((2,) * n)
    dist[(0,) * n] = 1.0
    scratch = np.empty_like(dist)
    for p, mask in zip(probs, masks, strict=True):
        axes = tuple(n - 1 - b for b in range(n) if (mask >> b) & 1)
        np.multiply(np.flip(dist, axis=axes), p, out=scratch)
        dist *= 1.0 - p
        dist += scratch
    return dist.reshape(2**num_obs, 2**num_det)


def enumerate_syndrome_class_distribution(
    dem: stim.DetectorErrorModel, *, max_mechanisms: int = 22
) -> np.ndarray:
    """Same as :func:`syndrome_class_distribution`, by summing over all ``2^M`` configurations."""
    num_det, num_obs = dem.num_detectors, dem.num_observables
    n = num_det + num_obs
    _check_bits(n, DEFAULT_MAX_BITS)
    probs, masks = error_mechanisms(dem)
    num_mech = len(probs)
    if num_mech > max_mechanisms:
        raise ValueError(f"{num_mech} mechanisms exceed max_mechanisms={max_mechanisms}")
    mask_arr = np.array(masks, dtype=np.int64)

    out = np.zeros(2**n)
    chunk = 1 << 16
    for start in range(0, 2**num_mech, chunk):
        configs = np.arange(start, min(start + chunk, 2**num_mech), dtype=np.int64)
        fired = ((configs[:, None] >> np.arange(num_mech)) & 1).astype(bool)
        symptoms = np.bitwise_xor.reduce(np.where(fired, mask_arr, 0), axis=1)
        weights = np.prod(np.where(fired, probs, 1.0 - probs), axis=1)
        out += np.bincount(symptoms, weights=weights, minlength=2**n)
    return out.reshape(2**num_obs, 2**num_det)


def pauli_bruteforce_distribution(
    code: StabilizerCode, p: float, basis: str = "Z", *, max_qubits: int = 10
) -> np.ndarray:
    """``P(s, l)`` for :func:`~qec_certify.codes.code_capacity_circuit`, from Pauli errors.

    Enumerates all ``4^n`` Pauli errors with weights ``(1 - p, p/3, p/3, p/3)`` per qubit.
    Detector ``k`` fires iff the error anticommutes with stabilizer ``k``; the observable
    flips iff it anticommutes with the logical operator of ``basis``.
    """
    n = code.num_data_qubits
    if n > max_qubits:
        raise ValueError(f"{n} qubits exceed max_qubits={max_qubits}")
    num_det = len(code.stabilizers)

    # Per-qubit Pauli code e: x-component = e & 1, z-component = e >> 1 (0=I, 1=X, 2=Z, 3=Y).
    configs = np.arange(4**n, dtype=np.int64)
    paulis = (configs[:, None] >> (2 * np.arange(n))) & 3
    x_part, z_part = paulis & 1, paulis >> 1
    weights = np.prod(np.where(paulis == 0, 1.0 - p, p / 3.0), axis=1)

    symptoms = np.zeros(len(configs), dtype=np.int64)
    for k, stab in enumerate(code.stabilizers):
        anti = z_part if stab.basis == "X" else x_part
        symptoms |= (anti[:, list(stab.qubits)].sum(axis=1) & 1) << k
    anti = x_part if basis == "Z" else z_part
    symptoms |= (anti[:, list(code.logical_support(basis))].sum(axis=1) & 1) << num_det

    out = np.bincount(symptoms, weights=weights, minlength=2 ** (num_det + 1))
    return out.reshape(2, 2**num_det)


class ExactMLDecoder(Decoder):
    """Exact maximum-likelihood decoder over logical classes, as a syndrome lookup table.

    For every syndrome ``s`` it predicts ``argmax_l P(s, l)``. Ties, including syndromes of
    probability zero, go to the smallest class index (no flip first).

    Attributes:
        lookup_table: Predicted class (packed observable flips) for each packed syndrome.
        optimal_logical_error_rate: Exact LER of this decoder, the minimum achievable by any
            decoder under the DEM's noise model.
    """

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        *,
        method: str = "dp",
        max_bits: int = DEFAULT_MAX_BITS,
    ) -> None:
        if method == "dp":
            joint = syndrome_class_distribution(dem, max_bits=max_bits)
        elif method == "enumerate":
            joint = enumerate_syndrome_class_distribution(dem)
        else:
            raise ValueError(f"method must be 'dp' or 'enumerate', got {method!r}")
        self.num_detectors = dem.num_detectors
        self.num_observables = dem.num_observables
        table = np.argmax(joint, axis=0)
        self.lookup_table = table.astype(np.min_scalar_type(joint.shape[0] - 1))
        self.optimal_logical_error_rate = float(
            (joint.sum(axis=0) - joint[table, np.arange(joint.shape[1])]).sum()
        )

    @classmethod
    def from_circuit(cls, circuit: stim.Circuit, **kwargs) -> ExactMLDecoder:
        return cls(circuit.detector_error_model(), **kwargs)

    def predict(self, detection_events: np.ndarray) -> np.ndarray:
        dets = self._check_detection_events(detection_events)
        classes = self.lookup_table[pack_bits(dets)]
        return unpack_bits(classes, self.num_observables)


def _check_bits(n: int, max_bits: int) -> None:
    if n > max_bits:
        raise ValueError(
            f"{n} detector + observable bits exceed max_bits={max_bits}; "
            "exact ML is only tractable for small codes"
        )
