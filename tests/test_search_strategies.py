"""Unit tests for the three search strategies and correctness against ground truth."""

import numpy as np
import pytest

from src.dataset import generate_synthetic_dataset
from src.ground_truth import compute_ground_truth, compute_recall_at_k
from src.metadata_index import MetadataIndex, Predicate
from src.search_strategies import SearchEngine
from src.vector_index import CustomVectorIndex


@pytest.fixture(scope="module")
def setup_engine():
    bundle = generate_synthetic_dataset(num_records=1200, embedding_dim=16, seed=42)
    meta_idx = MetadataIndex(bundle.metadata)
    vec_idx = CustomVectorIndex(num_clusters=16, nprobe=4, random_state=42)
    vec_idx.build(bundle.metadata["id"].to_numpy(), bundle.embeddings, bundle.metadata)
    engine = SearchEngine(meta_idx, vec_idx)
    return engine, bundle


def test_strategy_a_matches_ground_truth(setup_engine):
    engine, bundle = setup_engine
    q_vec = bundle.embeddings[0]
    preds = [Predicate.category_eq("jasmine_flowers")]

    gt = compute_ground_truth(q_vec, preds, k=10, metadata_index=engine.metadata_index, vector_index=engine.vector_index)
    res_a = engine.search_strategy_a_pre_filter(q_vec, preds, k=10)

    # Strategy A must match ground truth exactly
    assert res_a.results == gt
    assert compute_recall_at_k(res_a.results, gt, k=10) == 1.0
    assert res_a.exact_fallback is False


def test_strategy_b_post_filter(setup_engine):
    engine, bundle = setup_engine
    q_vec = bundle.embeddings[5]
    preds = [Predicate.price_range(100.0, 5000.0)]

    gt = compute_ground_truth(q_vec, preds, k=10, metadata_index=engine.metadata_index, vector_index=engine.vector_index)
    res_b = engine.search_strategy_b_post_filter(q_vec, preds, k=10)

    # All returned results must satisfy the predicates
    for r_id, _ in res_b.results:
        assert engine.metadata_index.matches_predicates(r_id, preds)

    recall = compute_recall_at_k(res_b.results, gt, k=10)
    assert recall >= 0.70  # Post-filtering achieves reasonable recall on moderate/broad queries


def test_strategy_c_partition_restricted(setup_engine):
    engine, bundle = setup_engine
    q_vec = bundle.embeddings[10]
    preds = [Predicate.category_eq("handloom_sarees")]

    gt = compute_ground_truth(q_vec, preds, k=10, metadata_index=engine.metadata_index, vector_index=engine.vector_index)
    res_c = engine.search_strategy_c_partition_restricted(q_vec, preds, k=10)

    for r_id, _ in res_c.results:
        assert engine.metadata_index.matches_predicates(r_id, preds)

    recall = compute_recall_at_k(res_c.results, gt, k=10)
    assert recall >= 0.85


def test_empty_results_when_predicates_unsatisfiable(setup_engine):
    engine, bundle = setup_engine
    q_vec = bundle.embeddings[0]
    # Price range with min > max
    preds = [Predicate.price_range(9999999.0, 100.0)]

    res_a = engine.search_strategy_a_pre_filter(q_vec, preds, k=10)
    assert len(res_a.results) == 0

    res_b = engine.search_strategy_b_post_filter(q_vec, preds, k=10, config={"allow_expansion": False})
    assert len(res_b.results) == 0

    res_c = engine.search_strategy_c_partition_restricted(q_vec, preds, k=10)
    assert len(res_c.results) == 0
