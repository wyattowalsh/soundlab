"""Tests for SOUNDLAB_SEED."""

from __future__ import annotations

import numpy as np

from soundlab.utils.seed import apply_configured_seed


def test_unset_seed_leaves_runtime_unseeded(monkeypatch) -> None:
    monkeypatch.delenv("SOUNDLAB_SEED", raising=False)
    assert apply_configured_seed() is None


def test_configured_seed_is_applied(monkeypatch) -> None:
    monkeypatch.setenv("SOUNDLAB_SEED", "7")
    assert apply_configured_seed() == 7
    first = np.random.rand(3)
    apply_configured_seed()
    second = np.random.rand(3)
    assert np.allclose(first, second)
