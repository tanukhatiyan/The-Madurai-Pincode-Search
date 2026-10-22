"""End-to-end integration tests for the Madurai Pincode Search engine."""

import numpy as np
import pytest

from src.benchmark import WorkloadGenerator
from src.dataset import generate_synthetic_dataset
from src.distance import euclidean_distance
from src.evaluation import BenchmarkEvaluator
from src.ground_truth import compute_ground_truth, compute_recall_at_k
from src.metadata_index import MetadataIndex, Predicate
from src.query_planner import AdaptiveQueryPlanner
from src.search_strategies import SearchEngine
from src.vector_index import CustomVectorIndex


def test_full_engine_integration():
    # 1. Dataset Generation
    bundle = generate_synthetic_dataset(num_records=1000, embedding_dim=16, seed=42)

    # 2. Metadata Indexing
    meta_idx = MetadataIndex(bundle.metadata)

    # 3. Custom Vector Indexing
    vec_idx = CustomVectorIndex(num_clusters=16, nprobe=4, random_state=42)
    vec_idx.build(bundle.metadata["id"].to_numpy(), bundle.embeddings, bundle.metadata)

    # 4. Engine & Planner
    engine = SearchEngine(meta_idx, vec_idx)
    planner = AdaptiveQueryPlanner(engine)
    evaluator = BenchmarkEvaluator(meta_idx, vec_idx, planner, k=10)

    # 5. Generate test workloads for all 5 selectivity buckets
    generator = WorkloadGenerator(bundle, meta_idx, seed=123)
    suite = generator.generate_full_benchmark_suite(queries_per_bucket=5)

    assert len(suite) == 5
    for b_name, q_list in suite.items():
        assert len(q_list) > 0

    # 6. Run evaluation suite
    detailed_df, summary_df = evaluator.run_benchmark_suite(suite, systems=["Adaptive_Planner", "Baseline_PreFilter"])
    assert len(detailed_df) > 0
    assert len(summary_df) > 0

    # Verify Adaptive Planner meets recall >= 0.95 across buckets
    planner_summary = summary_df[summary_df["system"] == "Adaptive_Planner"]
    assert len(planner_summary) == 5
    for _, row in planner_summary.iterrows():
        assert row["mean_recall"] >= 0.95, f"Bucket {row['bucket']} failed recall target: {row['mean_recall']}"


def test_deterministic_ties_and_fewer_than_k():
    bundle = generate_synthetic_dataset(num_records=200, embedding_dim=8, seed=10)
    meta_idx = MetadataIndex(bundle.metadata)
    vec_idx = CustomVectorIndex(num_clusters=4, nprobe=2, random_state=10)
    vec_idx.build(bundle.metadata["id"].to_numpy(), bundle.embeddings, bundle.metadata)
    engine = SearchEngine(meta_idx, vec_idx)

    # Construct predicate with only 2 matching records
    first_two_ids = bundle.metadata["id"].iloc[:2].tolist()
    p1 = bundle.metadata.loc[bundle.metadata["id"] == first_two_ids[0], "price"].iloc[0]
    p2 = bundle.metadata.loc[bundle.metadata["id"] == first_two_ids[1], "price"].iloc[0]
    min_p, max_p = min(p1, p2), max(p1, p2)

    cat = bundle.metadata.loc[bundle.metadata["id"] == first_two_ids[0], "category"].iloc[0]
    preds = [Predicate.category_eq(cat), Predicate.price_range(min_p, max_p)]

    q_vec = bundle.embeddings[first_two_ids[0]]
    gt = compute_ground_truth(q_vec, preds, k=10, metadata_index=meta_idx, vector_index=vec_idx)

    res_a = engine.search_strategy_a_pre_filter(q_vec, preds, k=10)
    assert len(res_a.results) == len(gt)
    assert compute_recall_at_k(res_a.results, gt, k=10) == 1.0
