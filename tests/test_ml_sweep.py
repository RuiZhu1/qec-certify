import numpy as np
import pytest
import stim
from test_ml_exact import P_GRID, random_dem

from qec_certify.codes import code_capacity_circuit, repetition_code, rotated_surface_code
from qec_certify.decoders import ExactMLDecoder, MWPMDecoder, SweepMLDecoder
from qec_certify.decoders.ml_exact import error_mechanisms, syndrome_class_distribution
from qec_certify.decoders.ml_sweep import candidate_orders, frontier_width
from qec_certify.sim import sample_syndromes
from qec_certify.utils import pack_bits, unpack_bits


def assert_matches_table(dem, syndromes):
    """``log P(s, l)`` and the decisions, ties included, equal the full table's."""
    joint = syndrome_class_distribution(dem)[:, syndromes].T
    with np.errstate(divide="ignore"):
        expected = np.log(joint)
    decoder = SweepMLDecoder(dem)
    dets = unpack_bits(syndromes, dem.num_detectors)
    got = decoder.log_joint(dets)
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(np.isfinite(got), finite)
    np.testing.assert_allclose(got[finite], expected[finite], rtol=0, atol=1e-11)
    np.testing.assert_array_equal(decoder.predict(dets), ExactMLDecoder(dem).predict(dets))


# --- agreement with the full table -------------------------------------------------------


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("p", P_GRID)
def test_matches_table_on_every_d3_syndrome(basis, p):
    dem = code_capacity_circuit(rotated_surface_code(3), p, basis).detector_error_model()
    syndromes = np.arange(2**dem.num_detectors)
    assert_matches_table(dem, syndromes)


@pytest.mark.parametrize("d", [3, 5, 7])
def test_matches_table_on_repetition_codes(d):
    dem = code_capacity_circuit(repetition_code(d), 0.1).detector_error_model()
    assert_matches_table(dem, np.arange(2**dem.num_detectors))


@pytest.mark.parametrize("seed", range(5))
def test_matches_table_on_random_dems(seed):
    """Random DEMs have no detector coordinates, several observables and hyperedges."""
    dem = random_dem(seed)
    assert_matches_table(dem, np.arange(2**dem.num_detectors))


def test_exact_ties_go_to_no_flip():
    """Symmetric DEM: both classes are exactly equally likely given D0, whatever the rounding."""
    dem = stim.DetectorErrorModel("error(0.1) D0\nerror(0.1) D0 L0")
    dets = np.array([[True]])
    assert not SweepMLDecoder(dem).predict(dets).any()
    assert not ExactMLDecoder(dem).predict(dets).any()


@pytest.mark.slow
def test_matches_table_on_sampled_d5_syndromes():
    circuit = code_capacity_circuit(rotated_surface_code(5), 0.1)
    dem = circuit.detector_error_model()
    syndromes = np.unique(pack_bits(sample_syndromes(circuit, 3000, seed=5).detection_events))
    assert_matches_table(dem, syndromes)


def test_impossible_and_untouched_detectors():
    # D1 is never flipped: any syndrome with D1 = 1 is impossible and decodes to no flip.
    dem = stim.DetectorErrorModel("error(0.2) D0 L0\nerror(0.1) D0 D2\ndetector D1")
    decoder = SweepMLDecoder(dem)
    syndromes = unpack_bits(np.arange(8), 3)
    log_joint = decoder.log_joint(syndromes)
    assert np.isneginf(log_joint[syndromes[:, 1]]).all()
    assert not decoder.predict(syndromes[syndromes[:, 1]]).any()
    np.testing.assert_allclose(decoder.predict_proba(syndromes).sum(axis=1), 1.0)
    np.testing.assert_array_equal(
        decoder.predict(syndromes), ExactMLDecoder(dem).predict(syndromes)
    )


def test_observable_only_mechanism():
    dem = stim.DetectorErrorModel("error(0.3) L0\nerror(0.1) D0\ndetector D0")
    decoder = SweepMLDecoder(dem)
    proba = decoder.predict_proba(np.array([[False], [True]]))
    np.testing.assert_allclose(proba, [[0.7, 0.3], [0.7, 0.3]])


def test_chunking_does_not_change_results():
    """Chunk size changes the summation order only, so results agree to rounding."""
    circuit = code_capacity_circuit(rotated_surface_code(5), 0.08)
    dets = sample_syndromes(circuit, 500, seed=2).detection_events
    a = SweepMLDecoder.from_circuit(circuit)
    b = SweepMLDecoder.from_circuit(circuit, chunk_elements=1)
    np.testing.assert_allclose(a.log_joint(dets), b.log_joint(dets), rtol=1e-13)
    np.testing.assert_array_equal(a.predict(dets), b.predict(dets))


# --- frontier and ordering ----------------------------------------------------------------


def test_frontier_width_counts_open_detectors():
    # Mechanisms over detectors {0, 1}, {1, 2}, {2, 3}: at most two detectors open at once.
    masks = [0b0011, 0b0110, 0b1100]
    assert frontier_width(masks, [0, 1, 2], 4) == 2
    # Processing the two ends first: detector 0 closes at once, but 1, 2 and 3 are then open
    # together until the middle mechanism.
    assert frontier_width(masks, [0, 2, 1], 4) == 3
    assert frontier_width([], [], 0) == 0


@pytest.mark.parametrize("d", [3, 5, 7, 9])
def test_surface_code_frontier_grows_linearly(d):
    dem = code_capacity_circuit(rotated_surface_code(d), 0.1).detector_error_model()
    decoder = SweepMLDecoder(dem)
    assert decoder.frontier <= d + 3
    _, masks = error_mechanisms(dem)
    assert sorted(decoder.order) == list(range(len(masks)))
    assert decoder.frontier == min(
        frontier_width(masks, order, dem.num_detectors) for order in candidate_orders(dem, masks)
    )


def test_frontier_guard():
    dem = code_capacity_circuit(rotated_surface_code(5), 0.1).detector_error_model()
    with pytest.raises(ValueError, match="max_frontier"):
        SweepMLDecoder(dem, max_frontier=4)


# --- beyond the table: d=7 ----------------------------------------------------------------


@pytest.mark.parametrize("p", [0.05, 0.1])
def test_d7_ml_not_worse_than_mwpm(p):
    """Beyond the full table: paired comparison with MWPM on the same shots."""
    circuit = code_capacity_circuit(rotated_surface_code(7), p)
    data = sample_syndromes(circuit, 20_000, seed=2026)
    ml_wrong = np.any(
        SweepMLDecoder.from_circuit(circuit).predict(data.detection_events)
        != data.observable_flips,
        axis=1,
    )
    mwpm_wrong = np.any(
        MWPMDecoder.from_circuit(circuit).predict(data.detection_events) != data.observable_flips,
        axis=1,
    )
    discordant = int(np.sum(ml_wrong != mwpm_wrong))
    assert ml_wrong.sum() - mwpm_wrong.sum() <= 3 * np.sqrt(max(discordant, 1))
