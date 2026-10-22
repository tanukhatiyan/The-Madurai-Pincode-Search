"""Benchmark evaluation runner, metrics computation, and comparative reporting."""

from dataclasses import asdict, dataclass
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.benchmark import BenchmarkQuery
from src.ground_truth import compute_ground_truth, compute_recall_at_k
from src.metadata_index import MetadataIndex
from src.query_planner import AdaptiveQueryPlanner
from src.search_strategies import SearchEngine, SearchResult
from src.vector_index import CustomVectorIndex


@dataclass
class QueryRunRecord:
    """Detailed record of a single query execution."""
    system: str  # 'Adaptive_Planner', 'Baseline_PreFilter', 'Baseline_PostFilter'
    bucket: str
    query_id: int
    observed_selectivity: float
    eligible_count: int
    returned_count: int
    recall_at_10: float
    latency_ms: float
    candidates_examined: int
    distances_computed: int
    partitions_visited: int
    exact_fallback: bool
    strategy_chosen: str


@dataclass
class BucketSummary:
    """Aggregated metrics for a single system within a selectivity bucket."""
    system: str
    bucket: str
    num_queries: int
    avg_selectivity: float
    mean_recall: float
    recall_at_10_target_met: bool
    p50_latency_ms: float
    p95_latency_ms: float
    mean_latency_ms: float
    mean_candidates: float
    mean_distances: float
    fallback_rate: float


class BenchmarkEvaluator:
    """Runs standardized evaluation across query buckets and compiles performance comparisons."""

    def __init__(
        self,
        metadata_index: MetadataIndex,
        vector_index: CustomVectorIndex,
        planner: AdaptiveQueryPlanner,
        k: int = 10,
    ):
        self.metadata_index = metadata_index
        self.vector_index = vector_index
        self.planner = planner
        self.search_engine = planner.search_engine
        self.k = k

    def evaluate_query(
        self,
        query: BenchmarkQuery,
        system_name: str,
        ground_truth: List[tuple],
    ) -> QueryRunRecord:
        """Run a single query on the specified system, measure timing, and compare against ground truth."""
        q_vec = query.query_vector
        preds = query.predicates

        # Execute system
        start_ns = time.perf_counter_ns()
        if system_name == "Adaptive_Planner":
            search_res: SearchResult = self.planner.execute_query(q_vec, preds, k=self.k)
            strat = search_res.diagnostics.get("planner_decision", {}).get("chosen_strategy", search_res.strategy_name)
        elif system_name == "Baseline_PreFilter":
            search_res = self.search_engine.search_strategy_a_pre_filter(q_vec, preds, k=self.k)
            strat = "Strategy_A_PreFilter"
        elif system_name == "Baseline_PostFilter":
            # Pure post filter with fixed standard candidate budget
            search_res = self.search_engine.search_strategy_b_post_filter(
                q_vec, preds, k=self.k, config={"candidate_budget": max(self.k * 30, 400), "allow_expansion": False}
            )
            strat = "Strategy_B_PostFilter"
        else:
            raise ValueError(f"Unknown system name: {system_name}")

        latency_ms = (time.perf_counter_ns() - start_ns) / 1e6

        recall = compute_recall_at_k(search_res.results, ground_truth, k=self.k)

        return QueryRunRecord(
            system=system_name,
            bucket=query.bucket,
            query_id=query.query_id,
            observed_selectivity=query.observed_selectivity,
            eligible_count=query.eligible_count,
            returned_count=len(search_res.results),
            recall_at_10=float(recall),
            latency_ms=float(latency_ms),
            candidates_examined=search_res.candidates_examined,
            distances_computed=search_res.distances_computed,
            partitions_visited=search_res.partitions_visited,
            exact_fallback=search_res.exact_fallback,
            strategy_chosen=strat,
        )

    def run_benchmark_suite(
        self,
        suite: Dict[str, List[BenchmarkQuery]],
        systems: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute full benchmark across all buckets and systems.

        Returns:
            Tuple of (detailed_runs_df, bucket_summary_df).
        """
        if systems is None:
            systems = ["Adaptive_Planner", "Baseline_PreFilter", "Baseline_PostFilter"]

        records: List[QueryRunRecord] = []

        # Warm up JIT and caches
        if suite and list(suite.values())[0]:
            warmup_q = list(suite.values())[0][0]
            _ = compute_ground_truth(warmup_q.query_vector, warmup_q.predicates, self.k, self.metadata_index, self.vector_index)
            for s in systems:
                _ = self.evaluate_query(warmup_q, s, [])

        for bucket_name, queries in suite.items():
            for query in queries:
                # 1. Compute exact ground truth once per query
                gt = compute_ground_truth(
                    query.query_vector,
                    query.predicates,
                    k=self.k,
                    metadata_index=self.metadata_index,
                    vector_index=self.vector_index,
                )

                # 2. Run all systems on the exact same query
                for s in systems:
                    rec = self.evaluate_query(query, s, gt)
                    records.append(rec)

        detailed_df = pd.DataFrame([asdict(r) for r in records])

        # Compute bucket summaries
        summary_rows: List[BucketSummary] = []
        for (sys_name, b_name), group in detailed_df.groupby(["system", "bucket"], sort=False):
            mean_rec = float(group["recall_at_10"].mean())
            latencies = group["latency_ms"].to_numpy()
            fb_rate = float(group["exact_fallback"].mean())

            summary_rows.append(BucketSummary(
                system=str(sys_name),
                bucket=str(b_name),
                num_queries=len(group),
                avg_selectivity=float(group["observed_selectivity"].mean()),
                mean_recall=round(mean_rec, 4),
                recall_at_10_target_met=(mean_rec >= 0.95),
                p50_latency_ms=round(float(np.percentile(latencies, 50)), 3),
                p95_latency_ms=round(float(np.percentile(latencies, 95)), 3),
                mean_latency_ms=round(float(np.mean(latencies)), 3),
                mean_candidates=round(float(group["candidates_examined"].mean()), 1),
                mean_distances=round(float(group["distances_computed"].mean()), 1),
                fallback_rate=round(fb_rate, 4),
            ))

        summary_df = pd.DataFrame([asdict(s) for s in summary_rows])
        return detailed_df, summary_df


def get_system_environment_info() -> Dict[str, str]:
    """Capture runtime machine and Python environment information."""
    return {
        "os_name": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }


def plot_benchmark_results(summary_df: pd.DataFrame, output_dir: str | Path, suffix: str = "") -> List[Path]:
    """Generate clear comparative benchmark visualization plots."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_plots = []

    # 1. Latency (p50 & p95) comparison across buckets
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    buckets_order = ["extremely_selective", "highly_selective", "moderate", "broad", "very_broad"]
    systems = summary_df["system"].unique().tolist()
    palette = {"Adaptive_Planner": "#1f77b4", "Baseline_PreFilter": "#ff7f0e", "Baseline_PostFilter": "#2ca02c"}

    x = np.arange(len(buckets_order))
    width = 0.25

    # Plot p50
    ax = axes[0]
    for i, sys_name in enumerate(systems):
        sys_data = summary_df[summary_df["system"] == sys_name].set_index("bucket").reindex(buckets_order)
        p50_vals = sys_data["p50_latency_ms"].fillna(0).to_numpy()
        ax.bar(x + (i - 1) * width, p50_vals, width, label=sys_name, color=palette.get(sys_name, "#555"))

    ax.set_title("p50 Latency by Selectivity Bucket (Lower is Better)")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("_", "\n") for b in buckets_order], fontsize=9)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    # Plot Recall@10
    ax2 = axes[1]
    for i, sys_name in enumerate(systems):
        sys_data = summary_df[summary_df["system"] == sys_name].set_index("bucket").reindex(buckets_order)
        recall_vals = sys_data["mean_recall"].fillna(0).to_numpy()
        ax2.bar(x + (i - 1) * width, recall_vals, width, label=sys_name, color=palette.get(sys_name, "#555"))

    ax2.axhline(0.95, color="red", linestyle="--", linewidth=1.5, label="Recall >= 0.95 Target")
    ax2.set_title("Mean Recall@10 by Selectivity Bucket (Higher is Better)")
    ax2.set_ylabel("Recall@10")
    ax2.set_ylim(0.0, 1.05)
    ax2.set_xticks(x)
    ax2.set_xticklabels([b.replace("_", "\n") for b in buckets_order], fontsize=9)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart1_path = out_dir / f"benchmark_latency_recall{suffix}.png"
    plt.savefig(chart1_path, dpi=200)
    plt.close()
    generated_plots.append(chart1_path)

    return generated_plots
