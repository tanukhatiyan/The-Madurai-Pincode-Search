# AI Usage Disclosure

## Overview
This document discloses the scope and nature of AI assistance utilized in the development, testing, benchmarking, and documentation of **The Madurai Pincode Search** project.

---

## 1. Scope of AI Assistance

AI tools were used to assist with:
- **Project architecture and structuring**: Organizing directory layout, module contracts, and interfaces.
- **Code implementation**:
  - Distance metrics with SIMD-vectorized NumPy Euclidean distance and deterministic tie-breaking (`src/distance.py`).
  - Synthetic dataset generation with correlated category-zone distributions, log-normal prices, and spatial cluster embeddings (`src/dataset.py`).
  - Inverted postings, sorted price indices, and analytical selectivity estimation (`src/metadata_index.py`).
  - Custom partitioned vector index with KMeans clustering, partition metadata bounds, and candidate ranking (`src/vector_index.py`).
  - Three search strategies (Pre-filtered, Post-filtered, Partition-restricted) with correctness-preserving fallback (`src/search_strategies.py`).
  - Adaptive query planner with cost modeling, cardinality analysis, and calibration routines (`src/query_planner.py`).
  - Evaluation harness, benchmark suite across 5 selectivity buckets, and metrics reporting (`src/evaluation.py`, `src/benchmark.py`).
  - Mutation workflows, dynamic re-indexing, and poor planner decision diagnosis (`src/mutations.py`).
- **Automated test suite authoring**: Comprehensive unit and integration test coverage across edge cases (`tests/`).
- **Documentation and visualization**: Architectural diagrams (Mermaid), benchmark plotting scripts, demonstration scripts, and README.

---

## 2. Validation and Testing Methodology

The final implementation was tested and verified using automated test suites and live benchmarks:
- **Test execution**: 29 automated tests executed via `pytest tests/`, covering:
  - Vector distance properties, zero vectors, non-finite values (NaN, Inf), dimensionality mismatches, and deterministic tie-breaking.
  - Dataset validation (50,000 records, unique contiguous IDs, category and zone bounds, non-negative INR prices, uneven zone distribution, and contingency table chi-squared correlation).
  - Metadata indexing, multi-predicate intersections (up to 3 AND filters), inclusive price boundaries, and O(log N) selectivity estimation.
  - Search strategy correctness against exact brute-force ground truth.
  - Query planner decision routing, cost evaluation, and diagnostic metadata.
  - Dynamic dataset mutations (10% deletion in busiest zone, 5% shifted insertions, and deleted-record exclusion verification).
  - End-to-end integration across all 5 query selectivity buckets.
- **Benchmark verification**: All latency and recall figures documented in this repository originate from recorded executions on 50,000 synthetic records and are saved in `results/benchmark_runs_initial.csv` and `results/benchmark_summary_initial.csv`. No metrics were fabricated.

---

## 3. Disclosed Limitations & Observations

1. **Synthetic Nature**: The dataset is entirely synthetic and created for vector search research. It does not reflect real personal addresses or actual Indian postal directories.
2. **CPU-Bound SIMD vs Partition Overhead**: For moderate-scale datasets (50,000 records), modern CPU SIMD vectorized operations in NumPy compute exact distances for thousands of records in sub-millisecond time. In this regime, partitioned indexing introduces cluster-traversal and array-concatenation overheads that make pre-filtering faster than cluster post-filtering for selective and moderate queries.
3. **Partition Scatter under High Dimensionality**: Cluster bounds pruning (Strategy C) is most effective when predicates correlate strongly with cluster assignments. For orthogonal predicates (e.g. wide price ranges without category constraints), clusters cannot be pruned aggressively, prompting the planner to favor pre-filtering.

---

## 4. Attestation
Performance claims within the repository reflect real hardware execution logs. Any developer reproducing this repository can execute `pytest tests/` and `python scripts/run_benchmarks.py` to verify the reported behavior.
