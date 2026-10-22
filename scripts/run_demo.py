"""Self-contained interactive demo for the Madurai Pincode Search engine."""

from pathlib import Path
import sys
import time

# Ensure repository root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.dataset import generate_synthetic_dataset
from src.ground_truth import compute_ground_truth, compute_recall_at_k
from src.metadata_index import MetadataIndex, Predicate
from src.mutations import MutationManager
from src.query_planner import AdaptiveQueryPlanner
from src.search_strategies import SearchEngine
from src.vector_index import CustomVectorIndex


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 80)
    print("DEMO: THE MADURAI PINCODE SEARCH - ADAPTIVE HYBRID VECTOR SEARCH ENGINE")
    print("=" * 80)

    # 1. Dataset Generation
    print("\n[STEP 1] Generating synthetic dataset (10,000 records, 32-dim embeddings)...")
    bundle = generate_synthetic_dataset(num_records=10000, embedding_dim=32, seed=42)
    stats = bundle.stats
    print(f"  Records generated:    {bundle.num_records:,}")
    print(f"  Categories:           {stats['categories_count']} commercial categories")
    print(f"  Zones:                {stats['zones_count']} administrative & commercial zones")
    print(f"  Busiest zone:         {stats['busiest_zone']} ({stats['busiest_zone_count']:,} records)")
    print(f"  Price range:          INR {stats['price_stats']['min']:,.2f} to INR {stats['price_stats']['max']:,.2f}")
    print("  * Synthetic data disclaimer: Synthetic records for research purposes only.")

    # 2. Build Inverted Metadata Index and Custom Vector Index
    print("\n[STEP 2] Building Inverted Metadata Index & Custom Partitioned Vector Index...")
    meta_idx = MetadataIndex(bundle.metadata)
    vec_idx = CustomVectorIndex(num_clusters=32, nprobe=8, random_state=42)
    vec_idx.build(bundle.metadata["id"].to_numpy(), bundle.embeddings, bundle.metadata)
    print(f"  Metadata postings:    Indexed {meta_idx.num_active_records:,} records")
    print(f"  Vector partitions:    {len(vec_idx.partitions)} clusters constructed via KMeans")

    engine = SearchEngine(meta_idx, vec_idx)
    planner = AdaptiveQueryPlanner(engine)

    # 3. Demonstration Queries
    print("\n[STEP 3] Demonstrating Adaptive Query Planner across 3 Distinct Workloads:")

    # Query 1: Highly Selective Query -> Strategy A
    print("\n--- Demo Query 1: Highly Selective Query ---")
    q_vec1 = bundle.embeddings[10]
    preds1 = [
        Predicate.category_eq("jasmine_flowers"),
        Predicate.zone_eq("Zone_Sellur"),
        Predicate.price_range(150.0, 350.0),
    ]
    gt1 = compute_ground_truth(q_vec1, preds1, k=10, metadata_index=meta_idx, vector_index=vec_idx)
    res1 = planner.execute_query(q_vec1, preds1, k=10)
    diag1 = res1.diagnostics["planner_decision"]
    recall1 = compute_recall_at_k(res1.results, gt1, k=10)

    print(f"  Filter:               Category=='jasmine_flowers' AND Zone=='Zone_Sellur' AND Price in [150, 350]")
    print(f"  Estimated Sel:        {diag1['estimated_selectivity']*100:.3f}% (~{diag1['estimated_eligible_records']} records)")
    print(f"  Strategy Chosen:      {diag1['chosen_strategy']}")
    print(f"  Planner Rationale:    {diag1['reason']}")
    print(f"  Execution Time:       {res1.latency_ms:.2f} ms")
    print(f"  Distances Computed:   {res1.distances_computed}")
    print(f"  Recall@10:            {recall1:.4f}")

    # Query 2: Broad / Unconstrained Query -> Strategy B
    print("\n--- Demo Query 2: Broad / Unfiltered Query ---")
    q_vec2 = bundle.embeddings[200]
    preds2 = []
    gt2 = compute_ground_truth(q_vec2, preds2, k=10, metadata_index=meta_idx, vector_index=vec_idx)
    res2 = planner.execute_query(q_vec2, preds2, k=10)
    diag2 = res2.diagnostics["planner_decision"]
    recall2 = compute_recall_at_k(res2.results, gt2, k=10)

    print(f"  Filter:               None (Pure vector search over 10,000 items)")
    print(f"  Estimated Sel:        {diag2['estimated_selectivity']*100:.1f}%")
    print(f"  Strategy Chosen:      {diag2['chosen_strategy']}")
    print(f"  Planner Rationale:    {diag2['reason']}")
    print(f"  Execution Time:       {res2.latency_ms:.2f} ms")
    print(f"  Partitions Probed:    {res2.partitions_visited} clusters")
    print(f"  Recall@10:            {recall2:.4f}")

    # Query 3: Moderate Query with Prunable Partitions -> Strategy C
    print("\n--- Demo Query 3: Moderate Query with Cluster Pruning ---")
    q_vec3 = bundle.embeddings[500]
    preds3 = [Predicate.category_eq("heavy_machinery")]
    gt3 = compute_ground_truth(q_vec3, preds3, k=10, metadata_index=meta_idx, vector_index=vec_idx)
    res3 = planner.execute_query(q_vec3, preds3, k=10)
    diag3 = res3.diagnostics["planner_decision"]
    recall3 = compute_recall_at_k(res3.results, gt3, k=10)

    print(f"  Filter:               Category=='heavy_machinery'")
    print(f"  Estimated Sel:        {diag3['estimated_selectivity']*100:.2f}% (~{diag3['estimated_eligible_records']} records)")
    print(f"  Strategy Chosen:      {diag3['chosen_strategy']}")
    print(f"  Planner Rationale:    {diag3['reason']}")
    print(f"  Execution Time:       {res3.latency_ms:.2f} ms")
    print(f"  Distances Computed:   {res3.distances_computed}")
    print(f"  Recall@10:            {recall3:.4f}")

    # 4. Mutation Testing Demonstration
    print("\n[STEP 4] Dynamic Dataset Mutation Testing:")
    mutation_mgr = MutationManager(bundle, meta_idx, vec_idx, planner, seed=999)
    mut_res = mutation_mgr.apply_mutations()
    print(f"  Busiest zone:         {mut_res.busiest_zone}")
    print(f"  Deleted:              {mut_res.num_deleted:,} records (10% of busiest zone)")
    print(f"  Inserted:             {mut_res.num_inserted:,} records (5% shifted distribution)")
    print(f"  Updated Active Count: {mut_res.new_total_records:,} records")
    print(f"  Deleted verified:     {mutation_mgr.verify_deleted_records_excluded(mut_res.deleted_ids)}")

    print("\n" + "=" * 80)
    print("DEMO COMPLETE: All algorithms, strategies, and verifications executed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
