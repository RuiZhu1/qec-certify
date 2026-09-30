import random
import re

import numpy as np
import pytest
import torch

from qec_certify.utils import (
    git_commit_hash,
    load_config,
    pack_bits,
    set_global_seed,
    spawn_seeds,
    split_seeds,
    unpack_bits,
)


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(0)
    bits = rng.random((100, 9)) < 0.5
    packed = pack_bits(bits)
    assert packed.dtype == np.int64
    np.testing.assert_array_equal(unpack_bits(packed, 9), bits)
    np.testing.assert_array_equal(pack_bits(np.array([[1, 0, 1], [0, 1, 1]])), [5, 6])
    with pytest.raises(ValueError):
        pack_bits(np.zeros((2, 63), bool))


def test_set_global_seed_is_reproducible():
    def draw():
        return random.random(), np.random.rand(), torch.rand(1).item()  # noqa: NPY002

    set_global_seed(123)
    first = draw()
    set_global_seed(123)
    assert draw() == first
    set_global_seed(124)
    assert draw() != first


def test_split_seeds_are_distinct_and_deterministic():
    seeds = split_seeds(42)
    assert set(seeds) == {"train", "val", "test"}
    assert len(set(seeds.values())) == 3
    assert split_seeds(42) == seeds
    assert split_seeds(43) != seeds
    assert spawn_seeds(42, 3) == list(seeds.values())


def test_load_config(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("seed: 3\ncode:\n  distance: 3\n")
    assert load_config(path) == {"seed": 3, "code": {"distance": 3}}
    path.write_text("- a\n- b\n")
    with pytest.raises(ValueError):
        load_config(path)


def test_git_commit_hash(tmp_path):
    assert git_commit_hash(tmp_path) is None
    commit = git_commit_hash()
    assert commit is None or re.fullmatch(r"[0-9a-f]{40}(-dirty)?", commit)
