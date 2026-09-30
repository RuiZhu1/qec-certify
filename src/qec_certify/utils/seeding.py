"""Deterministic seeding."""

from __future__ import annotations

import random

import numpy as np

SPLITS = ("train", "val", "test")


def set_global_seed(seed: int) -> None:
    """Seed Python's ``random``, NumPy's legacy global RNG and PyTorch."""
    random.seed(seed)
    np.random.seed(seed % 2**32)  # noqa: NPY002 - seeds the legacy global RNG on purpose
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def spawn_seeds(seed: int, n: int) -> list[int]:
    """``n`` statistically independent 32-bit child seeds derived from ``seed``."""
    children = np.random.SeedSequence(seed).spawn(n)
    return [int(child.generate_state(1)[0]) for child in children]


def split_seeds(seed: int) -> dict[str, int]:
    """Distinct seeds for the train, validation and test splits, so no shots are shared."""
    return dict(zip(SPLITS, spawn_seeds(seed, len(SPLITS)), strict=True))
