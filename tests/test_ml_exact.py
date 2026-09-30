import numpy as np
import pytest
import stim

from qec_certify.codes import code_capacity_circuit, repetition_code, rotated_surface_code
from qec_certify.decoders import Decoder, ExactMLDecoder, MWPMDecoder
from qec_certify.decoders.ml_exact import (
    enumerate_syndrome_class_distribution,
    error_mechanisms,
    pauli_bruteforce_distribution,
    syndrome_class_distribution,
)
from qec_certify.eval import exact_logical_error_rate, logical_error_rate
from qec_certify.sim import sample_syndromes
from qec_certify.utils import unpack_bits

P_GRID = [0.01, 0.05, 0.1, 0.2]


def random_dem(seed: int, num_det: int = 5, num_obs: int = 2, num_mech: int = 14):
    """Random DEM with hyperedges, several observables and repeated symptoms."""
    rng = np.random.default_rng(seed)
    lines = []
    for _ in range(num_mech):
        dets = rng.choice(num_det, size=rng.integers(1, 4), replace=False)
        targets = [f"D{d}" for d in sorted(dets)]
        if rng.random() < 0.5:
            targets.append(f"L{rng.integers(num_obs)}")
        lines.append(f"error({rng.uniform(0.01, 0.3)}) {' '.join(targets)}")
    lines += [f"detector D{d}" for d in range(num_det)]
    lines += [f"logical_observable L{k}" for k in range(num_obs)]
    return stim.DetectorErrorModel("\n".join(lines))


# --- ground truth: the exact distribution P(syndrome, logical class) ---------------------


def test_error_mechanisms_of_repetition_code():
    p = 0.12
    dem = code_capacity_circuit(repetition_code(3), p).detector_error_model()
    probs, masks = error_mechanisms(dem)
    # Z errors are invisible and dropped; X and Y on the same qubit merge into one mechanism
    # with probability 2p/3. Bits: D0 = 1, D1 = 2, L0 = 4.
    assert sorted(masks) == [0b010, 0b011, 0b101]
    np.testing.assert_allclose(probs, 2 * p / 3)


@pytest.mark.parametrize("d", [3, 5, 7])
@pytest.mark.parametrize("p", P_GRID)
def test_dp_matches_enumeration_repetition(d, p):
    dem = code_capacity_circuit(repetition_code(d), p).detector_error_model()
    np.testing.assert_allclose(
        syndrome_class_distribution(dem), enumerate_syndrome_class_distribution(dem), atol=1e-15
    )


@pytest.mark.parametrize("seed", range(5))
def test_dp_matches_enumeration_random_dem(seed):
    dem = random_dem(seed)
    dp = syndrome_class_distribution(dem)
    assert dp.shape == (4, 32)
    np.testing.assert_allclose(dp, enumerate_syndrome_class_distribution(dem), atol=1e-15)


@pytest.mark.parametrize(
    ("code", "basis"),
    [
        (repetition_code(3), "Z"),
        (repetition_code(5), "Z"),
        (rotated_surface_code(3), "Z"),
        (rotated_surface_code(3), "X"),
    ],
    ids=["rep3", "rep5", "surf3-Z", "surf3-X"],
)
@pytest.mark.parametrize("p", P_GRID)
def test_dp_matches_pauli_bruteforce(code, basis, p):
    """The DEM-based distribution equals the one computed from all 4^n Pauli errors."""
    dem = code_capacity_circuit(code, p, basis).detector_error_model()
    dp = syndrome_class_distribution(dem)
    np.testing.assert_allclose(dp, pauli_bruteforce_distribution(code, p, basis), atol=1e-14)
    assert dp.min() >= 0
    assert dp.sum() == pytest.approx(1.0)


def test_decomposed_dem_gives_same_distribution():
    circuit = code_capacity_circuit(rotated_surface_code(3), 0.1)
    np.testing.assert_allclose(
        syndrome_class_distribution(circuit.detector_error_model(decompose_errors=True)),
        syndrome_class_distribution(circuit.detector_error_model()),
        atol=1e-15,
    )


# --- the decoder ------------------------------------------------------------------------


@pytest.mark.parametrize("d", [3, 5])
def test_enumerate_and_dp_decoders_agree(d):
    dem = code_capacity_circuit(repetition_code(d), 0.1).detector_error_model()
    dp = ExactMLDecoder(dem, method="dp")
    enum = ExactMLDecoder(dem, method="enumerate")
    np.testing.assert_array_equal(dp.lookup_table, enum.lookup_table)
    assert dp.optimal_logical_error_rate == pytest.approx(enum.optimal_logical_error_rate)


def test_repetition_code_analytic_ler():
    """d=3 repetition code: ML fails iff >= 2 of 3 bits flip, each with probability 2p/3."""
    p = 0.15
    r = 2 * p / 3
    decoder = ExactMLDecoder.from_circuit(code_capacity_circuit(repetition_code(3), p))
    assert decoder.optimal_logical_error_rate == pytest.approx(3 * r**2 * (1 - r) + r**3)


@pytest.mark.parametrize("d", [3, 5, 7])
@pytest.mark.parametrize("p", P_GRID)
def test_ml_equals_mwpm_on_repetition_code(d, p):
    """For i.i.d. noise on an odd-distance repetition code, minimum weight is ML."""
    circuit = code_capacity_circuit(repetition_code(d), p)
    syndromes = unpack_bits(np.arange(2 ** (d - 1)), d - 1)
    np.testing.assert_array_equal(
        ExactMLDecoder.from_circuit(circuit).predict(syndromes),
        MWPMDecoder.from_circuit(circuit).predict(syndromes),
    )


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("p", P_GRID)
def test_exact_ml_ler_strictly_below_mwpm_on_surface_code(basis, p):
    """Exact (not sampled) LERs: ML beats MWPM, which ignores the X/Z correlation of Y errors."""
    circuit = code_capacity_circuit(rotated_surface_code(3), p, basis)
    dem = circuit.detector_error_model()
    ml = ExactMLDecoder(dem)
    ml_ler = exact_logical_error_rate(ml, dem)
    mwpm_ler = exact_logical_error_rate(MWPMDecoder.from_circuit(circuit), dem)
    assert ml_ler == pytest.approx(ml.optimal_logical_error_rate, rel=1e-12)
    assert ml_ler < mwpm_ler


class _TableDecoder(Decoder):
    def __init__(self, table: np.ndarray, num_detectors: int) -> None:
        self.table = table
        self.num_detectors = num_detectors
        self.num_observables = 1

    def predict(self, detection_events):
        dets = self._check_detection_events(detection_events)
        index = dets.astype(np.int64) @ (1 << np.arange(self.num_detectors))
        return self.table[index][:, None]


def test_no_lookup_table_beats_ml():
    circuit = code_capacity_circuit(rotated_surface_code(3), 0.1)
    dem = circuit.detector_error_model()
    optimum = ExactMLDecoder(dem).optimal_logical_error_rate
    rng = np.random.default_rng(0)
    ml_table = ExactMLDecoder(dem).lookup_table.astype(bool)
    for _ in range(20):
        perturbed = ml_table ^ (rng.random(ml_table.shape) < 0.05)
        decoder = _TableDecoder(perturbed, dem.num_detectors)
        assert exact_logical_error_rate(decoder, dem) >= optimum - 1e-15


def test_zero_probability_syndromes_decode_to_no_flip():
    dem = stim.DetectorErrorModel("error(0.2) D0 L0\ndetector D1")
    decoder = ExactMLDecoder(dem)
    np.testing.assert_array_equal(decoder.lookup_table, [0, 1, 0, 0])


def test_multiple_observables():
    dem = random_dem(0)
    decoder = ExactMLDecoder(dem)
    predictions = decoder.predict(unpack_bits(np.arange(32), 5))
    assert predictions.shape == (32, 2)
    classes = predictions.astype(np.int64) @ np.array([1, 2])
    np.testing.assert_array_equal(classes, syndrome_class_distribution(dem).argmax(axis=0))


def test_size_guards():
    dem = code_capacity_circuit(rotated_surface_code(3), 0.1).detector_error_model()
    with pytest.raises(ValueError, match="max_bits"):
        ExactMLDecoder(dem, max_bits=5)
    with pytest.raises(ValueError, match="max_mechanisms"):
        enumerate_syndrome_class_distribution(dem, max_mechanisms=10)
    with pytest.raises(ValueError, match="method"):
        ExactMLDecoder(dem, method="bp")
    with pytest.raises(ValueError, match="max_qubits"):
        pauli_bruteforce_distribution(rotated_surface_code(5), 0.1)


# --- sampled comparison: ML <= MWPM within statistical error -----------------------------


def _assert_ml_not_worse_than_mwpm(circuit, ml, shots, seed):
    dem = circuit.detector_error_model()
    mwpm = MWPMDecoder.from_circuit(circuit)
    data = sample_syndromes(circuit, shots, seed)
    ml_pred = ml.predict(data.detection_events)
    mwpm_pred = mwpm.predict(data.detection_events)

    ml_wrong = np.any(ml_pred != data.observable_flips, axis=1)
    mwpm_wrong = np.any(mwpm_pred != data.observable_flips, axis=1)
    # Paired comparison on the same shots: only discordant shots carry information.
    discordant = int(np.sum(ml_wrong != mwpm_wrong))
    assert ml_wrong.sum() - mwpm_wrong.sum() <= 3 * np.sqrt(max(discordant, 1))

    # Sampled LERs agree with the exact ones.
    ml_result = logical_error_rate(ml_pred, data.observable_flips, confidence=0.999)
    assert ml_result.ci_low <= ml.optimal_logical_error_rate <= ml_result.ci_high
    if dem.num_detectors <= 16:
        mwpm_result = logical_error_rate(mwpm_pred, data.observable_flips, confidence=0.999)
        assert mwpm_result.ci_low <= exact_logical_error_rate(mwpm, dem) <= mwpm_result.ci_high


@pytest.mark.parametrize("p", [0.02, 0.05, 0.1])
def test_sampled_ml_not_worse_than_mwpm_d3(p):
    circuit = code_capacity_circuit(rotated_surface_code(3), p)
    _assert_ml_not_worse_than_mwpm(circuit, ExactMLDecoder.from_circuit(circuit), 200_000, 2026)


@pytest.mark.slow
def test_sampled_ml_not_worse_than_mwpm_d5():
    circuit = code_capacity_circuit(rotated_surface_code(5), 0.1)
    _assert_ml_not_worse_than_mwpm(circuit, ExactMLDecoder.from_circuit(circuit), 50_000, 2026)
