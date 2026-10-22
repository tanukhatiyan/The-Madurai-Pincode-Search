"""Vector distance computation and exact top-k selection with deterministic tie-breaking."""

from typing import List, Tuple
import numpy as np


def euclidean_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Compute the Euclidean distance between two 1D vectors.

    Definition: d(u, v) = sqrt(sum_{i=1}^D (u_i - v_i)^2)

    Args:
        u: First vector of shape (D,).
        v: Second vector of shape (D,).

    Returns:
        Euclidean distance as float.

    Raises:
        ValueError: If vector shapes do not match or values are not finite.
    """
    if u.shape != v.shape:
        raise ValueError(f"Shape mismatch: {u.shape} vs {v.shape}")
    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
        raise ValueError("Vectors must contain only finite numeric values (no NaN or Inf).")
    diff = u - v
    return float(np.sqrt(np.dot(diff, diff)))


def vectorized_euclidean_distance(q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance from query vector q to a matrix of vectors X.

    Args:
        q: Query vector of shape (D,).
        X: Matrix of candidate vectors of shape (N, D).

    Returns:
        1D array of distances of shape (N,).

    Raises:
        ValueError: If dimensions mismatch or inputs contain non-finite numbers.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array, got ndim={X.ndim}")
    if q.ndim != 1:
        raise ValueError(f"q must be 1D array, got ndim={q.ndim}")
    if X.shape[1] != q.shape[0]:
        raise ValueError(f"Dimension mismatch: query dim {q.shape[0]} != matrix dim {X.shape[1]}")
    if not np.all(np.isfinite(q)):
        raise ValueError("Query vector must contain finite values.")
    if X.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)

    diff = X - q
    # Using np.einsum for fast row-wise squared sum without allocating intermediate square array
    sq_dist = np.einsum('ij,ij->i', diff, diff)
    # Numerical safeguard against tiny negative values
    np.maximum(sq_dist, 0.0, out=sq_dist)
    return np.sqrt(sq_dist)


def vectorized_squared_euclidean_distance(q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Compute squared Euclidean distance from query vector q to matrix X."""
    if X.ndim != 2 or q.ndim != 1 or X.shape[1] != q.shape[0]:
        raise ValueError("Dimension mismatch between query and vector matrix.")
    if X.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    diff = X - q
    sq_dist = np.einsum('ij,ij->i', diff, diff)
    np.maximum(sq_dist, 0.0, out=sq_dist)
    return sq_dist


def exact_top_k(
    distances: np.ndarray,
    record_ids: np.ndarray,
    k: int
) -> List[Tuple[int, float]]:
    """Select the top-k nearest records with deterministic tie-breaking by record ID.

    Tie-breaking rule:
        If distance(a) == distance(b), the record with smaller record_id ranks first.

    Args:
        distances: 1D array of distances of length N.
        record_ids: 1D array of record IDs of length N.
        k: Maximum number of results to return.

    Returns:
        List of tuples (record_id, distance) sorted ascending by distance, then record_id.
    """
    n = len(distances)
    if n == 0 or k <= 0:
        return []

    actual_k = min(k, n)

    if n <= actual_k:
        # Sort all elements
        indices = np.lexsort((record_ids, distances))
        return [(int(record_ids[idx]), float(distances[idx])) for idx in indices[:actual_k]]

    # For larger N, use argpartition to avoid full O(N log N) sort
    # argpartition gets the smallest actual_k elements in arbitrary order
    candidate_indices = np.argpartition(distances, actual_k - 1)[:actual_k]

    cand_dist = distances[candidate_indices]
    cand_ids = record_ids[candidate_indices]

    # Deterministic sort using lexsort (secondary key first: cand_ids, primary key: cand_dist)
    sort_order = np.lexsort((cand_ids, cand_dist))
    final_indices = candidate_indices[sort_order]

    return [(int(record_ids[idx]), float(distances[idx])) for idx in final_indices]
