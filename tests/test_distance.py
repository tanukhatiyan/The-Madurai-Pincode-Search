"""Unit tests for vector distance computation and deterministic top-k selection."""

import numpy as np
import pytest

from src.distance import (
    euclidean_distance,
    vectorized_euclidean_distance,
    vectorized_squared_euclidean_distance,
    exact_top_k,
)


def test_euclidean_distance_basic():
    u = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    v = np.array([4.0, 6.0, 3.0], dtype=np.float32)
    # diff = [-3, -4, 0] -> sqrt(9 + 16) = 5.0
    dist = euclidean_distance(u, v)
    assert pytest.approx(dist, 1e-5) == 5.0


def test_euclidean_distance_dimension_mismatch():
    u = np.array([1.0, 2.0], dtype=np.float32)
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(ValueError, match="Shape mismatch"):
        euclidean_distance(u, v)


def test_euclidean_distance_nan_inf():
    u = np.array([1.0, np.nan], dtype=np.float32)
    v = np.array([1.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError, match="finite numeric values"):
        euclidean_distance(u, v)

    u2 = np.array([1.0, np.inf], dtype=np.float32)
    with pytest.raises(ValueError, match="finite numeric values"):
        euclidean_distance(u2, v)


def test_vectorized_euclidean_distance():
    q = np.array([0.0, 0.0], dtype=np.float32)
    X = np.array([
        [3.0, 4.0],  # dist 5
        [1.0, 0.0],  # dist 1
        [0.0, 0.0],  # dist 0
    ], dtype=np.float32)

    dists = vectorized_euclidean_distance(q, X)
    assert len(dists) == 3
    assert pytest.approx(dists[0], 1e-5) == 5.0
    assert pytest.approx(dists[1], 1e-5) == 1.0
    assert pytest.approx(dists[2], 1e-5) == 0.0


def test_vectorized_empty_input():
    q = np.array([1.0, 2.0], dtype=np.float32)
    X = np.empty((0, 2), dtype=np.float32)
    dists = vectorized_euclidean_distance(q, X)
    assert len(dists) == 0


def test_exact_top_k_deterministic_tie_breaking():
    # Two candidates with identical distance: id 40 and id 12
    # Deterministic rule: smaller ID ranks first
    distances = np.array([2.5, 1.0, 1.0, 3.0], dtype=np.float32)
    record_ids = np.array([100, 40, 12, 50], dtype=np.int32)

    top2 = exact_top_k(distances, record_ids, k=2)
    assert len(top2) == 2
    # 12 should precede 40 despite same distance 1.0
    assert top2[0] == (12, 1.0)
    assert top2[1] == (40, 1.0)


def test_exact_top_k_fewer_than_k():
    distances = np.array([2.0, 1.5], dtype=np.float32)
    record_ids = np.array([5, 2], dtype=np.int32)
    res = exact_top_k(distances, record_ids, k=10)
    assert len(res) == 2
    assert res[0] == (2, 1.5)
    assert res[1] == (5, 2.0)


def test_exact_top_k_empty():
    res = exact_top_k(np.array([], dtype=np.float32), np.array([], dtype=np.int32), k=5)
    assert res == []
