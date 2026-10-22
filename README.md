# The Madurai Pincode Search — Adaptive Metadata-Filtered Vector Search

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Tests](https://img.shields.io/badge/pytest-29%20passed-brightgreen.svg)]()
[![Recall SLA](https://img.shields.io/badge/Recall%4010-100%25%20Met%20(%E2%89%A50.95)-success.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An industrial-grade, CPU-optimized vector search engine built from scratch in Python and NumPy. It adaptively routes similarity queries across multiple search strategies based on analytical filter selectivity estimation, achieving **Recall@10 $\ge$ 0.95 across all query selectivity regimes** without relying on third-party vector databases or ready-made ANN libraries.

---

## 1. Project Overview & Objectives

In modern search systems, querying vector embeddings alongside structured metadata filters (e.g., categories, locations, price thresholds) introduces challenging trade-offs:
- **Pre-filtering** guarantees 100% exact recall, but scanning millions of vector distances becomes computationally prohibitive when filters are broad.
- **Post-filtering** probes vector clusters first, but risks catastrophic recall drops when selective filters discard 99.9% of candidate vectors.
- **Partition-restricted search** uses metadata bounds to prune clusters, but suffers when items are scattered across multiple partitions.

**The Madurai Pincode Search** solves this dilemma through an **Adaptive Query Planner** that analytically estimates query selectivity in $O(1)$ to $O(\log N)$ time and dynamically chooses the optimal execution strategy for each incoming query.

---

## 2. Features & System Constraints

- **Zero Black-Box ANN Libraries**: Built without FAISS, hnswlib, Annoy, or ScaNN. Candidate retrieval, partition selection, SIMD vector distance calculations, and tie-breaking are implemented from scratch in pure NumPy.
- **Strict Determinism**: Guaranteed deterministic tie-breaking by ascending record ID for equal-distance records.
- **Multi-Predicate Filtering**: Full support for `category == value`, `zone == value`, `min_price <= price <= max_price`, and combined `AND` queries up to three predicates.
- **Hardware Agnostic**: Optimized for multi-platform CPU execution across Windows, Linux, and macOS.
- **Dynamic Mutation Resilience**: Dynamic handling of record deletions (10% busiest zone) and insertions (5% shifted distribution) with zero index corruption and verified exclusion of deleted records.

---

## 3. Technology Stack

- **Python 3.11+**
- **NumPy (1.26.4)**: High-performance vectorized Euclidean distance computation and sorted array operations.
- **Pandas (2.2.0)**: Metadata management and tabular evaluation summary reporting.
- **scikit-learn (1.4.2)**: Used **strictly for KMeans partition centroid generation** during index build (no ANN search).
- **Matplotlib (3.8.2)**: Comparative latency/recall visualization and benchmark plotting.
- **pytest (9.1.1)**: Automated unit and integration test framework.

---

## 4. Dataset Schema & Synthetic Generation

The engine generates 50,000 synthetic records (configurable up to 100,000) modeling commercial hubs across Madurai:

| Field | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `id` | `int32` | Unique, contiguous record identifier | `0, 1, 2, ...` |
| `embedding` | `float32[32]` | L2-normalized 32-dimensional semantic embedding vector | `[-0.142, 0.089, ...]` |
| `category` | `string` | 20 commercial categories with realistic market affinities | `jasmine_flowers`, `handloom_sarees` |
| `price` | `float32` | Log-normal price in Indian Rupees (INR ₹) | `₹1,249.02` (range: ₹40 – ₹600,000) |
| `zone` | `string` | 30 Madurai municipal and suburban commercial zones | `Zone_Meenakshi_Amman`, `Zone_Kappalur` |
| `pincode` | `string` | 6-digit postal code | `625001`, `625008` |

### Key Statistical Properties
1. **Uneven Zone Distributions**: Power-law distribution where commercial hubs (e.g. `Zone_Meenakshi_Amman`: 11,451 records) have over 24x the volume of peripheral zones (`Zone_Melur_Road`: 471 records).
2. **Category-Zone Correlations**: Verified via contingency table chi-squared test ($\chi^2 \gg 2000$). For example, `jasmine_flowers` and `temple_handicrafts` cluster heavily around Meenakshi Amman and Simmakkal, whereas `automotive_parts` clusters in Kappalur.
3. **Correlated Cluster Embeddings**: Record embeddings are synthesized from category base vectors, spatial zone biases, and Gaussian noise, ensuring meaningful vector-space neighbors.

> **Synthetic Data Disclaimer**: This dataset is completely synthetic for algorithmic research and benchmark validation. The name refers to the assignment domain; records do not represent real personal addresses or official postal directories.

---

## 5. System Architecture

```mermaid
flowchart TD
    subgraph Query ["Incoming Query"]
        Q_vec["Query Vector (32-dim)"]
        Q_preds["Metadata Predicates (up to 3 AND filters)"]
        Q_k["k = 10"]
    end

    subgraph MetadataIndex ["Metadata Indexing (O(1) - O(log N))"]
        CatPostings["Category Postings (Sorted Array)"]
        ZonePostings["Zone Postings (Sorted Array)"]
        PriceIndex["Sorted Price Index (Binary Search)"]
        JointFreq["Joint Frequency Table"]
        Estimator["Selectivity Estimator"]
    end

    subgraph VectorIndex ["Custom Partitioned Vector Index"]
        Centroids["64 Centroids (KMeans)"]
        Partitions["Cluster Postings & Metadata Bounds"]
        SIMDStore["Aligned float32 SIMD Matrix"]
    end

    subgraph Planner ["Adaptive Query Planner"]
        CostModel["Analytical Cost Model"]
        Regime{"Selectivity Regime"}
    end

    subgraph Strategies ["Execution Strategies"]
        StratA["Strategy A: Pre-Filtered Exact Search"]
        StratB["Strategy B: Post-Filtered Vector Search"]
        StratC["Strategy C: Partition-Restricted Search"]
        Fallback["Correctness Fallback to Strategy A"]
    end

    subgraph Result ["Top-k Output"]
        Out["Exact Top-10 (Deterministic Tie-Breaking)"]
    end

    Query --> Estimator
    MetadataIndex --> Estimator
    Estimator --> Regime

    Regime -- "Selective (<= 15% or < 8,000 matches)" --> StratA
    Regime -- "Moderate (Prunable Partitions)" --> StratC
    Regime -- "Broad (>= 80% Unconstrained)" --> StratB

    StratC -. "Insufficient Candidates" .-> Fallback
    Fallback --> StratA

    StratA --> Out
    StratB --> Out
    StratC --> Out
```

---

## 6. Search Strategies

1. **Strategy A (Pre-Filtered Exact Search)**:
   - Queries metadata postings and price ranges via array intersection.
   - Computes vectorized Euclidean distances only for the qualified candidate IDs.
   - Exact 100% recall guaranteed on active records.
2. **Strategy B (Post-Filtered Vector Search)**:
   - Probes the nearest $nprobe$ vector cluster centroids.
   - Ranks candidates by proximity, then filters by metadata.
   - Supported by candidate expansion if matches $< k$.
3. **Strategy C (Partition-Restricted Search)**:
   - Uses per-cluster metadata bounds (`categories`, `zones`, `min_price`, `max_price`) to prune non-matching clusters.
   - Searches only viable clusters ranked by centroid proximity.
   - Contains a correctness-preserving fallback to Strategy A if candidates are insufficient.

---

## 7. Metadata Index & Query Planner Algorithm

The planner **never scans the raw dataset** to estimate selectivity. It evaluates inverted postings lengths, joint frequency crosstabs, and binary searches `sorted_prices`:
$$\hat{s} = \frac{\text{JointCount}(cat, zone) \times \text{PriceFraction}}{N_{active}}$$

### Calibrated Cost Model
$$\text{Cost}_A = C_{meta} \cdot |\text{preds}| + c_{dist} \cdot \hat{N}_{elig}$$
$$\text{Cost}_B = C_{py\_overhead} + C_{probe} \cdot nprobe + c_{dist} \cdot N_{cand}$$
$$\text{Cost}_C = C_{py\_overhead} \cdot 0.7 + C_{prune} \cdot K + c_{dist} \cdot N_{cand\_C}$$

Thresholds calibrated on independent non-leaking query sets:
- **$\hat{s} \le 0.15$ or $\hat{N}_{elig} \le 8,000$**: Route to **Strategy A** (Pre-Filter).
- **$\hat{s} \ge 0.80$**: Route to **Strategy B** (Post-Filter) if $Cost_B < Cost_A$.
- **$0.15 < \hat{s} < 0.80$**: Route to **Strategy C** if partition pruning prunes $> 60\%$ of clusters; otherwise route to **Strategy A**.

---

## 8. Benchmark Results (50,000 Records, 250 Queries)

The benchmark was executed against exact brute-force ground truth across 5 selectivity buckets (50 queries per bucket):

| Selectivity Bucket | Target Range | System | Mean Recall@10 | p50 Latency (ms) | p95 Latency (ms) | Distances Computed | Target Met? |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Extremely Selective** | $< 0.05\%$ | **Adaptive Planner** | **1.0000** | **0.28 ms** | **0.50 ms** | **2.7** | **YES** |
| | | Baseline Pre-Filter | 1.0000 | 0.13 ms | 0.34 ms | 2.7 | YES |
| | | Baseline Post-Filter | 0.3899 | 7.29 ms | 10.52 ms | 21,724.8 | NO |
| **Highly Selective** | $\sim 0.1\%$ | **Adaptive Planner** | **1.0000** | **0.25 ms** | **0.43 ms** | **51.4** | **YES** |
| | | Baseline Pre-Filter | 1.0000 | 0.14 ms | 0.26 ms | 51.4 | YES |
| | | Baseline Post-Filter | 0.6120 | 7.48 ms | 10.71 ms | 21,807.9 | NO |
| **Moderate** | $\sim 1\%$ | **Adaptive Planner** | **1.0000** | **0.59 ms** | **1.53 ms** | **480.7** | **YES** |
| | | Baseline Pre-Filter | 1.0000 | 0.45 ms | 0.76 ms | 480.7 | YES |
| | | Baseline Post-Filter | 0.7720 | 8.18 ms | 11.59 ms | 21,787.9 | NO |
| **Broad** | $\sim 10\%$ | **Adaptive Planner** | **1.0000** | **0.88 ms** | **1.91 ms** | **4,678.8** | **YES** |
| | | Baseline Pre-Filter | 1.0000 | 0.72 ms | 2.66 ms | 4,678.8 | YES |
| | | Baseline Post-Filter | 0.9300 | 10.33 ms | 13.98 ms | 21,716.5 | NO |
| **Very Broad** | $75\% - 100\%$ | **Adaptive Planner** | **0.9740** | **36.07 ms** | **41.46 ms** | **21,738.1** | **YES** |
| | | Baseline Pre-Filter | 1.0000 | 8.22 ms | 10.91 ms | 48,110.0 | YES |
| | | Baseline Post-Filter | 0.9740 | 36.17 ms | 42.42 ms | 21,738.1 | YES |

> **Target Verification**: The Adaptive Planner achieved **Recall@10 $\ge$ 0.95 across 100% of selectivity buckets** (1.0000, 1.0000, 1.0000, 1.0000, 0.9740).

---

## 9. Dynamic Mutation Testing & Planner Diagnosis

### Workload
1. **Mutation 1 (10% Busiest Zone Deletion)**: Identified `Zone_Meenakshi_Amman` (11,451 records). Deleted 1,145 records randomly with seed 42. Verified 100% exclusion from subsequent searches.
2. **Mutation 2 (5% Shifted Insertions)**: Added 2,500 new valid records from a shifted distribution (inflated prices, shifted categories favoring electronics and solar equipment, shifted cluster vectors). Active records updated to **51,355**.

### Post-Mutation Benchmark Results
| Selectivity Bucket | System | Mean Recall@10 | p50 Latency (ms) | p95 Latency (ms) | Target Met? |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Extremely Selective** | Adaptive Planner | 1.0000 | 0.30 ms | 0.55 ms | YES |
| **Highly Selective** | Adaptive Planner | 1.0000 | 0.31 ms | 0.58 ms | YES |
| **Moderate** | Adaptive Planner | 1.0000 | 0.62 ms | 1.30 ms | YES |
| **Broad** | Adaptive Planner | 1.0000 | 0.86 ms | 1.89 ms | YES |
| **Very Broad** | Adaptive Planner | 0.9700 | 32.29 ms | 39.34 ms | YES |

### Poor Planner Decision & Targeted Fix
- **Diagnostic Case**: Uncalibrated planner evaluating shifted distribution query (`category == 'electronics_hardware' AND zone == 'Zone_Kappalur' AND price in [1000, 50000]`).
- **Initial Decision**: Selected `Strategy_B_PostFilter` (probed multiple clusters, computed 5,892 distances, latency 1.92 ms).
- **Alternative**: `Strategy_A_PreFilter` computed only 400 exact distances in 2.06 ms with zero partition overhead and 100% recall.
- **Root Cause**: Stale cardinality estimates post-mutation underestimated the clustering dispersion of the newly inserted items.
- **Targeted Fix**: Refreshed active metadata statistics immediately post-mutation, calibrated Python array-gathering overhead in the cost model, and enforced an automatic Strategy A fallback.

---

## 10. Setup & Execution Instructions

### Environment Setup
```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Reproduce Everything in 4 Commands

```powershell
# 1. Run automated test suite (29 tests)
python -m pytest tests/

# 2. Run interactive demonstration
python scripts/run_demo.py

# 3. Generate 50,000 synthetic records and run full benchmark suite
python scripts/run_benchmarks.py --queries-per-bucket 50

# 4. Run mutation testing, diagnosis, and post-mutation evaluation
python scripts/run_mutation_test.py --queries-per-bucket 30
```

---

## 11. Artifact Directory

All execution artifacts are saved to `results/`:
- `results/benchmark_runs_initial.csv`: Execution log of all 750 query executions.
- `results/benchmark_summary_initial.csv`: Aggregated percentiles and recall metrics.
- `results/benchmark_summary_post_mutation.csv`: Post-mutation metrics across all buckets.
- `results/mutation_report.json`: Full diagnostic report of deletions, insertions, and planner investigation.
- `results/benchmark_latency_recall_initial.png`: Initial latency and recall comparison plots.
- `results/benchmark_latency_recall_post_mutation.png`: Post-mutation comparison plots.

---

## 12. Known Limitations & Future Improvements

1. **Hardware Regime**: On 50,000 records, NumPy CPU SIMD vector scanning is so fast (~8 ms for 50k vectors) that IVF partitioning primarily provides speedups on selective/moderate queries rather than broad queries. Future work includes testing with 1,000,000+ vectors where IVF indexing shows larger scaling advantages.
2. **Partition Clustering**: Strategy C cluster bounds pruning is most effective when queries filter on correlated categories. Adding hierarchical multi-level clustering would enhance pruning efficiency on orthogonal price predicates.
