"""Run standard benchmark evaluation comparing Adaptive Planner and Baselines."""

import argparse
from pathlib import Path
import sys
import time

# Ensure repository root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark import WorkloadGenerator
from src.dataset import generate_synthetic_dataset, load_dataset, save_dataset
from src.evaluation import BenchmarkEvaluator, get_system_environment_info, plot_benchmark_results
from src.metadata_index import MetadataIndex
from src.query_planner import AdaptiveQueryPlanner
from src.search_strategies import SearchEngine
from src.vector_index import CustomVectorIndex


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Run Madurai vector search benchmark suite.")
    parser.add_argument("--records", type=int, default=50000, help="Record count if dataset needs generation")
    parser.add_argument("--queries-per-bucket", type=int, default=50, help="Queries per selectivity bucket (default: 50)")
    parser.add_argument("--clusters", type=int, default=64, help="Number of KMeans partitions (default: 64)")
    parser.add_argument("--nprobe", type=int, default=28, help="Partitions to probe (default: 28)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--outdir", type=str, default="results", help="Directory for benchmark outputs")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data"
    results_dir = root_dir / args.outdir
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MADURAI PINCODE SEARCH: ADAPTIVE BENCHMARK EVALUATOR")
    print("=" * 80)

    # 1. Load or Generate Dataset
    csv_file = data_dir / "madurai_records.csv"
    npz_file = data_dir / "madurai_embeddings.npz"

    if csv_file.exists() and npz_file.exists():
        print(f"Loading existing dataset from {data_dir}...")
        bundle = load_dataset(data_dir)
        print(f"Loaded {bundle.num_records:,} records with {bundle.embedding_dim}-dim embeddings.")
    else:
        print(f"Dataset not found. Generating fresh {args.records:,} records...")
        bundle = generate_synthetic_dataset(num_records=args.records, embedding_dim=32, seed=args.seed)
        save_dataset(bundle, data_dir)
        print("Generated and saved dataset to disk.")

    # 2. Build Metadata Index
    print("\nBuilding metadata postings and price indexes...")
    t0 = time.perf_counter()
    metadata_idx = MetadataIndex(bundle.metadata)
    t_meta = (time.perf_counter() - t0) * 1000
    print(f"Metadata index built in {t_meta:.2f} ms ({metadata_idx.num_active_records:,} records).")

    # 3. Build Custom Vector Index
    print(f"\nBuilding custom partitioned vector index (clusters={args.clusters}, nprobe={args.nprobe})...")
    t0 = time.perf_counter()
    vector_idx = CustomVectorIndex(num_clusters=args.clusters, nprobe=args.nprobe, random_state=args.seed)
    vector_idx.build(bundle.metadata["id"].to_numpy(), bundle.embeddings, bundle.metadata)
    t_vec = (time.perf_counter() - t0) * 1000
    print(f"Custom vector index built in {t_vec:.2f} ms ({len(vector_idx.partitions)} active partitions).")

    # 4. Initialize Search Engine & Query Planner
    search_engine = SearchEngine(metadata_idx, vector_idx)
    planner = AdaptiveQueryPlanner(search_engine)

    # 5. Generate Calibration Queries & Calibrate Thresholds
    print("\nGenerating separate calibration query set to avoid data leakage...")
    calib_generator = WorkloadGenerator(bundle, metadata_idx, seed=args.seed + 100)
    calib_suite = calib_generator.generate_full_benchmark_suite(queries_per_bucket=15)
    calib_list = []
    for q_list in calib_suite.values():
        for q in q_list:
            calib_list.append((q.query_vector, q.predicates))

    calib_report = planner.calibrate_thresholds(calib_list)
    print(f"Planner calibration complete:")
    print(f"  Calibration queries: {calib_report['calibration_queries_count']}")
    print(f"  Mean selectivity estimation error: {calib_report['mean_selectivity_estimation_error']:.4f}")
    print(f"  Calibrated low threshold:  {calib_report['calibrated_low_threshold']}")
    print(f"  Calibrated high threshold: {calib_report['calibrated_high_threshold']}")

    # 6. Generate Independent Evaluation Workload Suite
    print(f"\nGenerating evaluation workload ({args.queries_per_bucket} queries/bucket across 5 buckets)...")
    eval_generator = WorkloadGenerator(bundle, metadata_idx, seed=args.seed + 200)
    eval_suite = eval_generator.generate_full_benchmark_suite(queries_per_bucket=args.queries_per_bucket)

    total_queries = sum(len(q_list) for q_list in eval_suite.values())
    print(f"Generated {total_queries} evaluation queries across 5 selectivity buckets.")

    # 7. Run Benchmark Suite
    print("\nExecuting comparative evaluation against exact brute-force ground truth...")
    evaluator = BenchmarkEvaluator(metadata_idx, vector_idx, planner, k=10)
    detailed_df, summary_df = evaluator.run_benchmark_suite(eval_suite)

    # 8. Save Results & Visualizations
    detailed_csv = results_dir / "benchmark_runs_initial.csv"
    summary_csv = results_dir / "benchmark_summary_initial.csv"
    detailed_df.to_csv(detailed_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    plots = plot_benchmark_results(summary_df, results_dir, suffix="_initial")

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY RESULTS")
    print("=" * 80)
    print(summary_df[["system", "bucket", "mean_recall", "p50_latency_ms", "p95_latency_ms", "mean_distances", "recall_at_10_target_met"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("RECALL TARGET VERIFICATION (Recall@10 >= 0.95)")
    print("=" * 80)
    planner_results = summary_df[summary_df["system"] == "Adaptive_Planner"]
    all_met = bool(planner_results["recall_at_10_target_met"].all())
    for _, row in planner_results.iterrows():
        status = "[PASSED]" if row["recall_at_10_target_met"] else "[FAILED]"
        print(f"  {status} {row['bucket']:20s} Recall: {row['mean_recall']:.4f} | p50: {row['p50_latency_ms']:.2f} ms | p95: {row['p95_latency_ms']:.2f} ms")

    print(f"\nOverall Target Met: {'YES (100% compliant)' if all_met else 'NO'}")
    print(f"\nSaved benchmark outputs to:")
    print(f"  Detailed runs: {detailed_csv}")
    print(f"  Summary table: {summary_csv}")
    for p in plots:
        print(f"  Chart plot:    {p}")
    print("=" * 80)


if __name__ == "__main__":
    main()
