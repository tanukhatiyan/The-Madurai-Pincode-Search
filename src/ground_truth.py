"""Exact brute-force ground truth calculation with metadata filtering and deterministic tie-breaking."""

from typing import List, Optional, Tuple
import numpy as np

from src.distance import vectorized_euclidean_distance, exact_top_k
from src.metadata_index import MetadataIndex, Predicate
from src.vector_index import CustomVectorIndex


def compute_ground_truth(
    query_vector: np.ndarray,
    predicates: Optional[List[Predicate]],
    k: int,
    metadata_index: MetadataIndex,
    vector_index: CustomVectorIndex,
) -> List[Tuple[int, float]]:
    """Compute exact ground-truth nearest neighbors among all records satisfying metadata predicates.

    Important:
        Ground truth is computed against the active dataset without using any index partitions
        or candidate pruning. Deterministic tie-breaking by ascending record ID is guaranteed.

    Args:
        query_vector: Query vector of shape (D,).
        predicates: Metadata predicates (up to 3 AND filters).
        k: Maximum number of neighbors (typically 10).
        metadata_index: Active metadata index.
        vector_index: Active vector store.

    Returns:
        List of (record_id, distance) tuples sorted ascending by distance, then record_id.
    """
    # 1. Retrieve all eligible records strictly matching metadata predicates
    eligible_ids = metadata_index.get_eligible_ids(predicates)
    n_eligible = len(eligible_ids)

    if n_eligible == 0:
        return []

    # 2. Extract vectors for all eligible records from the active vector store
    eligible_vectors = vector_index.get_vectors_batch(eligible_ids)

    # 3. Compute exact Euclidean distances
    distances = vectorized_euclidean_distance(query_vector, eligible_vectors)

    # 4. Rank with deterministic tie-breaking
    top_results = exact_top_k(distances, eligible_ids, k=min(k, n_eligible))

    return top_results


def compute_recall_at_k(
    retrieved_results: List[Tuple[int, float]],
    ground_truth_results: List[Tuple[int, float]],
    k: int = 10,
) -> float:
    """Calculate Recall@k against exact ground truth.

    Definition:
        Recall@k = |retrieved_ids & ground_truth_ids| / |ground_truth_ids|

    When fewer than k records match the predicates in the entire dataset,
    the denominator is the total number of available ground-truth records.
    If ground truth is empty, recall is defined as 1.0 if retrieved is also empty,
    else 0.0.
    """
    gt_ids = set(r[0] for r in ground_truth_results[:k])
    if not gt_ids:
        # If there are no eligible records in the entire dataset, returning 0 records is 100% correct
        return 1.0 if len(retrieved_results) == 0 else 0.0

    ret_ids = set(r[0] for r in retrieved_results[:k])
    overlap = len(gt_ids.intersection(ret_ids))
    return float(overlap / len(gt_ids))
