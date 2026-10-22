"""Run mutation experiments, diagnose a poor planner decision, and evaluate post-fix performance."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

# Ensure repository root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.benchmark import WorkloadGenerator
from src.dataset import generate_synthetic_dataset, load_dataset, save_dataset
from src.evaluation import BenchmarkEvaluator, plot_benchmark_results
from src.metadata_index import MetadataIndex
from src.mutations import MutationManager
from src.query_planner import AdaptiveQueryPlanner
from src.search_strategies import SearchEngine
from src.vector_index import CustomVectorIndex


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Execute mutation testing and planner diagnosis.")
    parser.add_argument("--queries-per-bucket", type=int, default=30, help="Queries per bucket for post-mutation benchmark")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--outdir", type=str, default="results", help="Directory for output reports")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data"
    results_dir = root_dir / args.outdir
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MADURAI PINCODE SEARCH: MUTATION TESTING & PLANNER DIAGNOSIS")
    print("=" * 80)

    # 1. Load active dataset
    csv_file = data_dir / "madurai_records.csv"
    if not csv_file.exists():
        print("Generating fresh dataset...")
        bundle = generate_synthetic_dataset(num_records=50000, embedding_dim=32, seed=args.seed)
        save_dataset(bundle, data_dir)
    else:
        bundle = load_dataset(data_dir)

    print(f"Active baseline dataset: {bundle.num_records:,} records.")

    # 2. Build initial indexes
    metadata_idx = MetadataIndex(bundle.metadata)
    vector_idx = CustomVectorIndex(num_clusters=64, nprobe=28, random_state=args.seed)
    vector_idx.build(bundle.metadata["id"].to_numpy(), bundle.embeddings, bundle.metadata)

    engine = SearchEngine(metadata_idx, vector_idx)
    planner = AdaptiveQueryPlanner(engine)

    # 3. Apply Mutations via MutationManager
    print("\nApplying mutations:")
    print("  Mutation 1: Delete 10% of items in the busiest zone.")
    print("  Mutation 2: Add 5% new records from a shifted distribution.")
    mutation_mgr = MutationManager(bundle, metadata_idx, vector_idx, planner, seed=args.seed)
    mutation_res = mutation_mgr.apply_mutations()

    print(f"\nMutation summary:")
    print(f"  Busiest zone:          {mutation_res.busiest_zone} (initially {mutation_res.original_busiest_count:,} records)")
    print(f"  Records deleted:       {mutation_res.num_deleted:,} (10%)")
    print(f"  New records inserted:  {mutation_res.num_inserted:,} (5% relative to initial size)")
    print(f"  New active total:      {mutation_res.new_total_records:,}")
    print(f"  Shifted categories:    {', '.join(mutation_res.shifted_categories)}")

    # 4. Verify deleted records exclusion
    exclusion_verified = mutation_mgr.verify_deleted_records_excluded(mutation_res.deleted_ids)
    print(f"  Deleted records exclusion verified: {exclusion_verified}")
    if not exclusion_verified:
        print("ERROR: Deleted records were detected in active postings!")
        sys.exit(1)

    # 5. Generate workload to find and diagnose a poor planner decision
    print("\nSearching for a sub-optimal planner decision on mutated distribution...")
    workload_gen = WorkloadGenerator(bundle, metadata_idx, seed=args.seed + 300)
    test_queries = workload_gen.generate_workload_for_bucket("broad", num_queries=30)
    test_queries += workload_gen.generate_workload_for_bucket("moderate", num_queries=30)

    poor_decision_report = mutation_mgr.find_and_diagnose_poor_decision(test_queries)

    if poor_decision_report:
        print("\n" + "=" * 80)
        print("POOR PLANNER DECISION INVESTIGATION REPORT")
        print("=" * 80)
        print(f"Query ID:             {poor_decision_report.query_id}")
        print(f"Predicates:           {poor_decision_report.predicates_description}")
        print(f"Estimated Sel:        {poor_decision_report.estimated_selectivity*100:.2f}%")
        print(f"Actual Sel:           {poor_decision_report.actual_selectivity*100:.2f}%")
        print(f"Chosen Strategy:      {poor_decision_report.chosen_strategy}")
        print(f"Chosen Latency:       {poor_decision_report.chosen_latency_ms:.2f} ms ({poor_decision_report.chosen_distances} distances)")
        print(f"Alternative Strategy: {poor_decision_report.alternative_strategy}")
        print(f"Alternative Latency:  {poor_decision_report.alt_latency_ms:.2f} ms ({poor_decision_report.alt_distances} distances)")
        print(f"Measured Slowdown:    {poor_decision_report.slowdown_factor:.2f}x slower than alternative")
        print(f"\nRoot Cause:\n  {poor_decision_report.root_cause}")
        print(f"\nTargeted Fix:\n  {poor_decision_report.fix_description}")
        print("=" * 80)

    # 6. Apply Planner Fix
    print("\nApplying planner fix (re-tuning thresholds and updating cluster dispersion cost)...")
    planner.low_threshold = 0.15
    planner.c_dist = 0.00015

    # 7. Run Full Post-Mutation Benchmark
    print("\nRunning post-mutation evaluation across all 5 selectivity buckets...")
    eval_suite = workload_gen.generate_full_benchmark_suite(queries_per_bucket=args.queries_per_bucket)
    evaluator = BenchmarkEvaluator(metadata_idx, vector_idx, planner, k=10)
    post_detailed_df, post_summary_df = evaluator.run_benchmark_suite(eval_suite)

    post_summary_csv = results_dir / "benchmark_summary_post_mutation.csv"
    post_summary_df.to_csv(post_summary_csv, index=False)

    plot_benchmark_results(post_summary_df, results_dir, suffix="_post_mutation")

    # Save full mutation report JSON
    mutation_report_path = results_dir / "mutation_report.json"
    report_dict = {
        "mutation": asdict(mutation_res),
        "poor_decision_investigation": asdict(poor_decision_report) if poor_decision_report else None,
        "post_mutation_summary": post_summary_df.to_dict(orient="records"),
    }
    with open(mutation_report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    print("\n" + "=" * 80)
    print("POST-MUTATION BENCHMARK SUMMARY")
    print("=" * 80)
    print(post_summary_df[["system", "bucket", "mean_recall", "p50_latency_ms", "p95_latency_ms", "recall_at_10_target_met"]].to_string(index=False))

    print(f"\nArtifacts saved to:")
    print(f"  Post-mutation CSV:    {post_summary_csv}")
    print(f"  Mutation Report JSON: {mutation_report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
