# Demonstration Video Script & Walkthrough

**Project**: The Madurai Pincode Search — Adaptive Metadata-Filtered Vector Search  
**Target Duration**: 6–8 minutes  
**Format**: Screen recording with terminal output and browser/markdown views.

---

## Timeline & Scene Overview

| Timestamp | Section / Topic | Command / View | Key Takeaway |
| :--- | :--- | :--- | :--- |
| **00:00 – 00:45** | 1. Introduction & AI Usage Disclosure | `code AI_USAGE.md` | Disclose AI assistance, scope, and validation methodology. |
| **00:45 – 01:30** | 2. Synthetic Dataset Generation | `python scripts/generate_dataset.py` | 50,000 records, 20 categories, 30 zones, uneven distribution. |
| **01:30 – 02:30** | 3. Architecture & Custom Vector Index | `code docs/architecture.md` | No FAISS/HNSW; custom partitioned index + analytical selectivity. |
| **02:30 – 03:15** | 4. Automated Test Suite Execution | `pytest tests/` | 29 passing unit & integration tests across edge cases. |
| **03:15 – 04:30** | 5. Live Demonstration of 3 Strategies | `python scripts/run_demo.py` | Highly selective (A), Broad (B), Moderate/pruned (C). |
| **04:30 – 05:30** | 6. Benchmark Suite & Recall@10 Target | `python scripts/run_benchmarks.py` | Recall >= 0.95 across all 5 selectivity buckets. |
| **05:30 – 06:45** | 7. Mutations, Poor Decision & Targeted Fix | `python scripts/run_mutation_test.py` | 10% busiest zone deletion, 5% shifted addition, poor decision diagnosis & fix. |
| **06:45 – 07:30** | 8. Summary & Conclusion | `git status` / `ls results/` | Clean, reproducible, fully verified project. |

---

## Detailed Scene-by-Scene Script

### Scene 1: Introduction & AI Usage Disclosure (00:00 - 00:45)
- **Visual**: Show `AI_USAGE.md` in the editor.
- **Spoken Script**:
  > "Welcome to the Madurai Pincode Search demonstration. Before examining the code, we begin with our AI Usage Disclosure in `AI_USAGE.md`. AI tools were used for code generation, architectural scaffolding, and test suite design. All performance claims are grounded strictly in recorded executions on our test suite, with no external black-box vector search libraries permitted."

### Scene 2: Synthetic Dataset Generation (00:45 - 01:30)
- **Command**:
  ```powershell
  python scripts/generate_dataset.py --records 50000 --dim 32 --seed 42
  ```
- **Spoken Script**:
  > "We generate 50,000 synthetic records with 32-dimensional embeddings. The dataset models Madurai's commercial hubs—such as Meenakshi Amman, Simmakkal, and Kappalur—with realistic uneven zone sizes and category-zone affinities. Notice the validation checks: unique IDs, finite embeddings, non-negative INR prices, and contingency chi-square correlations. A clear synthetic disclaimer emphasizes this is algorithmic benchmark data."

### Scene 3: Architecture & Custom Index Implementation (01:30 - 02:30)
- **Visual**: Show `docs/architecture.md` and `src/vector_index.py`.
- **Spoken Script**:
  > "Our system features a custom partitioned vector index with 64 clusters using NumPy, with KMeans used only for initial centroid generation. Each cluster stores member record IDs and metadata bounds. The MetadataIndex provides inverted postings for categories and zones, and binary search for price ranges, calculating estimated selectivity in $O(1)$ to $O(\log N)$ time without full-table scans."

### Scene 4: Automated Test Execution (02:30 - 03:15)
- **Command**:
  ```powershell
  pytest tests/
  ```
- **Spoken Script**:
  > "We run our full test suite. 29 automated tests execute in under 7 seconds, verifying distance functions, deterministic tie-breaking by record ID, three-predicate AND combinations, empty result sets, deleted-record exclusion, and end-to-end integration."

### Scene 5: Strategy Demonstration via Demo Script (03:15 - 04:30)
- **Command**:
  ```powershell
  python scripts/run_demo.py
  ```
- **Spoken Script**:
  > "Let's watch the Adaptive Query Planner route three distinct query workloads:
  > - Query 1 is highly selective (0.02% selectivity): The planner routes to Strategy A (Pre-filter), computing distances only for qualified records in 0.36 ms with 1.0000 recall.
  > - Query 2 is unconstrained: The planner routes to Strategy B (Post-filter), searching top vector clusters directly.
  > - Query 3 has moderate selectivity: The planner evaluates partition bounds, prunes non-matching clusters, and runs Strategy C (Partition-Restricted)."

### Scene 6: Benchmark Execution & Recall Verification (04:30 - 05:30)
- **Command**:
  ```powershell
  python scripts/run_benchmarks.py --queries-per-bucket 50
  ```
- **Spoken Script**:
  > "We now run the complete benchmark across all 5 selectivity buckets against exact ground truth. As shown in the generated summary table, the Adaptive Planner meets the Recall@10 >= 0.95 SLA across every single bucket, while achieving sub-millisecond p50 latencies on selective queries."

### Scene 7: Dataset Mutations, Poor Decision, & Targeted Fix (05:30 - 06:45)
- **Command**:
  ```powershell
  python scripts/run_mutation_test.py
  ```
- **Spoken Script**:
  > "Next, we stress-test the system with dataset mutations:
  > - We identify the busiest zone (`Zone_Meenakshi_Amman`) and delete 10% of its records.
  > - We insert 5% new records from a shifted distribution favoring electronics and solar equipment.
  > - We verify deleted records are excluded.
  > - We diagnose a sub-optimal planner decision where cluster dispersion caused unnecessary candidate visits, apply our calibrated cost fix, and demonstrate improved latency."

### Scene 8: Conclusion & Reproducibility (06:45 - 07:30)
- **Visual**: Show `results/` folder with generated CSVs and PNG charts.
- **Spoken Script**:
  > "All benchmark figures, plots, and mutation reports are persisted in the `results/` folder. The codebase is clean, CPU-compatible across Windows, Linux, and macOS, and fully reproducible with standard commands. Thank you."
