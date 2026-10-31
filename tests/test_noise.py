import itertools
import math

import numpy as np
import pytest

from qec_certify.codes import code_capacity_circuit, repetition_code, rotated_surface_code
from qec_certify.decoders.ml_exact import pauli_bruteforce_distribution, syndrome_class_distribution
from qec_certify.noise import (
    BiasedPauli,
    CorrelatedPairs,
    Depolarizing,
    as_noise_model,
    describe,
    make_noise,
    nearest_neighbour_pairs,
    noise_grid,
)
from qec_certify.noise.descriptors import (
    bias,
    bulk_pair,
    entropy_bits,
    marginal,
    mutual_information_bits,
    xor_convolve,
)

STRUCTURED = [
    BiasedPauli(0.08, 0.5),
    BiasedPauli(0.08, 10),
    BiasedPauli(0.15, 100),
    BiasedPauli(0.3, 3),
    CorrelatedPairs(0.08, 0.5),
    CorrelatedPairs(0.15, 1.0),
]


# --- circuits ------------------------------------------------------------------------------


def test_bare_probability_and_uncorrelated_pairs_give_the_depolarizing_circuit():
    code = rotated_surface_code(3)
    expected = code_capacity_circuit(code, Depolarizing(0.1))
    assert code_capacity_circuit(code, 0.1) == expected
    assert code_capacity_circuit(code, CorrelatedPairs(0.1, 0.0)) == expected
    assert as_noise_model(0.1) == Depolarizing(0.1)


def test_bias_one_half_is_depolarizing():
    code = rotated_surface_code(3)
    np.testing.assert_allclose(
        syndrome_class_distribution(
            code_capacity_circuit(code, BiasedPauli(0.12, 0.5)).detector_error_model()
        ),
        syndrome_class_distribution(code_capacity_circuit(code, 0.12).detector_error_model()),
        atol=1e-15,
    )


@pytest.mark.parametrize("noise", STRUCTURED, ids=str)
@pytest.mark.parametrize(
    ("code", "basis"),
    [(rotated_surface_code(3), "Z"), (rotated_surface_code(3), "X"), (repetition_code(5), "Z")],
    ids=["surf3-Z", "surf3-X", "rep5"],
)
def test_dem_matches_bruteforce(noise, code, basis):
    """Stim's DEM of the circuit equals the composition of the model's own channel tables."""
    dem = code_capacity_circuit(code, noise, basis).detector_error_model()
    np.testing.assert_allclose(
        syndrome_class_distribution(dem),
        pauli_bruteforce_distribution(code, noise, basis),
        atol=1e-14,
    )


def test_channel_without_independent_form_is_rejected():
    """X and Y errors alone (eta = 0) cannot be independent mechanisms, since X and Y together
    make Z; Stim refuses to approximate instead of returning a wrong DEM."""
    circuit = code_capacity_circuit(rotated_surface_code(3), BiasedPauli(0.3, 0))
    with pytest.raises(ValueError, match="approximate_disjoint_errors"):
        circuit.detector_error_model()


def test_nearest_neighbour_pairs():
    assert nearest_neighbour_pairs(repetition_code(5)) == ((0, 1), (1, 2), (2, 3), (3, 4))
    pairs = nearest_neighbour_pairs(rotated_surface_code(5))
    assert len(pairs) == 2 * 5 * 4
    degree = np.bincount(np.ravel(pairs), minlength=25)
    assert degree[12] == 4 and degree[0] == 2


# --- parameters ----------------------------------------------------------------------------


def test_biased_probabilities():
    p_x, p_y, p_z = BiasedPauli(0.09, 2).pauli_probabilities
    assert p_x == pytest.approx(0.015) and p_y == pytest.approx(0.015)
    assert p_z == pytest.approx(0.06)
    assert p_z / (p_x + p_y) == pytest.approx(2)


@pytest.mark.parametrize(
    "make",
    [
        lambda: Depolarizing(0.8),
        lambda: BiasedPauli(-0.1, 1),
        lambda: BiasedPauli(0.1, -1),
        lambda: BiasedPauli(0.1, math.inf),
        lambda: CorrelatedPairs(0.1, 1.5),
        lambda: CorrelatedPairs(0.9, 0.5),
    ],
)
def test_invalid_parameters(make):
    with pytest.raises(ValueError):
        make()


def test_make_noise_and_grid():
    assert make_noise("biased", 0.1, 3) == BiasedPauli(0.1, 3)
    assert make_noise("depolarizing", 0.1) == Depolarizing(0.1)
    with pytest.raises(ValueError, match="unknown noise model"):
        make_noise("coherent", 0.1)
    with pytest.raises(ValueError, match="no structure parameter"):
        make_noise("depolarizing", 0.1, 3)
    with pytest.raises(ValueError, match="needs its 'correlation' parameter"):
        make_noise("correlated", 0.1)
    grid = noise_grid({"model": "correlated", "p": [0.05, 0.1], "correlation": [0, 0.5]})
    assert grid == [
        CorrelatedPairs(0.05, 0),
        CorrelatedPairs(0.05, 0.5),
        CorrelatedPairs(0.1, 0),
        CorrelatedPairs(0.1, 0.5),
    ]
    assert noise_grid({"model": "depolarizing", "p": 0.1}) == [Depolarizing(0.1)]
    for noise in grid:  # parameters() round-trips through make_noise
        params = noise.parameters()
        assert make_noise(params["model"], params["p"], params[noise.structure]) == noise
    assert BiasedPauli(0.1, 3).parameters() == {"model": "biased", "p": 0.1, "bias": 3}


# --- descriptors ---------------------------------------------------------------------------


def test_descriptor_helpers():
    a = np.array([0.7, 0.1, 0.1, 0.1])
    b = np.array([0.9, 0.0, 0.1, 0.0])
    ab = xor_convolve(a, b)
    assert ab.sum() == pytest.approx(1.0)
    assert ab[2] == pytest.approx(0.7 * 0.1 + 0.1 * 0.9)  # Z from (I, Z) or (Z, I)
    assert entropy_bits(np.array([0.5, 0.5, 0, 0])) == pytest.approx(1.0)
    independent = np.outer([0.9, 0.1, 0, 0], [0.8, 0, 0.2, 0]).reshape(-1)
    assert mutual_information_bits(independent) == pytest.approx(0.0, abs=1e-12)
    perfectly_correlated = np.zeros(16)
    perfectly_correlated[0] = perfectly_correlated[1 + 4 * 1] = 0.5  # II or XX
    assert mutual_information_bits(perfectly_correlated) == pytest.approx(1.0)
    assert bias(np.array([0.9, 0.0, 0.1, 0.0])) == math.inf
    assert math.isnan(bias(np.array([1.0, 0.0, 0.0, 0.0])))


def test_marginal_orders_qubits_as_requested():
    noise = CorrelatedPairs(0.1, 1.0)
    code = repetition_code(3)
    channels = noise.channels(code)
    forward = marginal(channels, (0, 1)).reshape(4, 4)  # [e_1, e_0]
    backward = marginal(channels, (1, 0)).reshape(4, 4)  # [e_0, e_1]
    np.testing.assert_allclose(forward, backward.T)
    # Qubit 1 sits in two pairs and qubit 0 in one, so qubit 1 is noisier.
    assert forward.sum(axis=1)[0] < forward.sum(axis=0)[0]


def test_bulk_pair_is_central():
    assert bulk_pair(rotated_surface_code(5)) == (12, 7)
    assert bulk_pair(repetition_code(5)) == (2, 1)


@pytest.mark.parametrize("eta", [0.5, 3, 100])
def test_biased_descriptors(eta):
    described = describe(BiasedPauli(0.08, eta), rotated_surface_code(5))
    assert described["error_rate"] == pytest.approx(0.08)
    assert described["bias"] == pytest.approx(eta)
    assert described["neighbour_mutual_information_bits"] == pytest.approx(0.0, abs=1e-12)


def test_correlated_descriptors():
    code = rotated_surface_code(7)
    described = [describe(CorrelatedPairs(0.08, c), code) for c in (0.0, 0.25, 0.5, 1.0)]
    information = [x["neighbour_mutual_information_bits"] for x in described]
    assert information[0] == pytest.approx(0.0, abs=1e-12)
    assert all(a < b for a, b in itertools.pairwise(information))
    for x in described:
        # A bulk qubit's error probability is p to first order; pair events can cancel.
        assert x["error_rate"] == pytest.approx(0.08, abs=0.08**2)
        assert x["bias"] == pytest.approx(0.5)
