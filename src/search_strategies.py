"""Implementation of the three search strategies and standardized search interface."""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.distance import vectorized_euclidean_distance, exact_top_k
from src.metadata_index import MetadataIndex, Predicate
from src.vector_index import CustomVectorIndex


@dataclass
class SearchResult:
    """Standardized search output across all strategies."""
    results: List[Tuple[int, float]]  # [(record_id, distance), ...]
    strategy_name: str
    candidates_examined: int
    partitions_visited: int
    eligible_records: int
    distances_computed: int
    exact_fallback: bool
    latency_ms: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class SearchEngine:
    """Coordinates metadata index and vector index to execute search strategies."""

    def __init__(self, metadata_index: MetadataIndex, vector_index: CustomVectorIndex):
        self.metadata_index = metadata_index
        self.vector_index = vector_index

    # -------------------------------------------------------------------------
    # Strategy A: Pre-Filtered Exact Search
    # -------------------------------------------------------------------------
    def search_strategy_a_pre_filter(
        self,
        query_vector: np.ndarray,
        predicates: Optional[List[Predicate]],
        k: int = 10,
        config: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """Strategy A: Pre-filtered exact search.

        1. Retrieve eligible record IDs from metadata indexes.
        2. Compute exact distances only for those eligible records.
        3. Return nearest k eligible records.

        Guarantees exact recall (1.0) on the active dataset.
        """
        start_ns = time.perf_counter_ns()

        eligible_ids = self.metadata_index.get_eligible_ids(predicates)
        n_eligible = len(eligible_ids)

        if n_eligible == 0:
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
            return SearchResult(
                results=[],
                strategy_name="Strategy_A_PreFilter",
                candidates_examined=0,
                partitions_visited=0,
                eligible_records=0,
                distances_computed=0,
                exact_fallback=False,
                latency_ms=elapsed_ms,
                diagnostics={"reason": "No records matched metadata predicates"},
            )

        # Batch retrieve vectors for eligible candidates
        vectors = self.vector_index.get_vectors_batch(eligible_ids)
        distances = vectorized_euclidean_distance(query_vector, vectors)
        top_k_results = exact_top_k(distances, eligible_ids, k=min(k, n_eligible))

        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
        return SearchResult(
            results=top_k_results,
            strategy_name="Strategy_A_PreFilter",
            candidates_examined=n_eligible,
            partitions_visited=0,
            eligible_records=n_eligible,
            distances_computed=n_eligible,
            exact_fallback=False,
            latency_ms=elapsed_ms,
            diagnostics={"eligible_count": n_eligible},
        )

    # -------------------------------------------------------------------------
    # Strategy B: Post-Filtered Vector Search
    # -------------------------------------------------------------------------
    def search_strategy_b_post_filter(
        self,
        query_vector: np.ndarray,
        predicates: Optional[List[Predicate]],
        k: int = 10,
        config: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """Strategy B: Post-filtered vector search.

        1. Retrieve candidate records using the vector index without applying filters.
        2. Probes top nprobe clusters to gather a comprehensive candidate pool.
        3. Apply metadata predicates to filter candidates down to k results.
        4. If insufficient matches and expansion is enabled, probe additional partitions.
        """
        start_ns = time.perf_counter_ns()
        cfg = config or {}

        initial_nprobe = cfg.get("nprobe", max(28, self.vector_index.nprobe))
        candidate_budget = cfg.get("candidate_budget", max(k * 40, 500))
        allow_expansion = cfg.get("allow_expansion", True)
        max_probe = len(self.vector_index.partitions)

        ranked_clusters = self.vector_index.rank_clusters(query_vector)

        partitions_visited = 0
        current_probe_idx = 0
        matching_results: List[Tuple[int, float]] = []
        total_candidates_examined = 0
        total_distances_computed = 0
        seen_ids = set()

        # Phase 1: Probe initial_nprobe clusters to build high-recall candidate pool
        probes_initial = min(initial_nprobe, max_probe)
        initial_clusters = ranked_clusters[:probes_initial]
        current_probe_idx = probes_initial
        partitions_visited = probes_initial

        new_cands = []
        for c_id in initial_clusters:
            part = self.vector_index.partitions[c_id]
            if len(part.member_ids) > 0:
                new_cands.append(part.member_ids)

        if new_cands:
            batch_arr = np.concatenate(new_cands)
            total_candidates_examined += len(batch_arr)
            seen_ids.update(batch_arr.tolist())

            batch_vecs = self.vector_index.get_vectors_batch(batch_arr)
            batch_dists = vectorized_euclidean_distance(query_vector, batch_vecs)
            total_distances_computed += len(batch_dists)

            valid_mask = self.metadata_index.matches_batch(batch_arr, predicates)
            filtered_ids = batch_arr[valid_mask]
            filtered_dists = batch_dists[valid_mask]

            for r_id, d in zip(filtered_ids, filtered_dists):
                matching_results.append((int(r_id), float(d)))

        # Phase 2: If insufficient matches and expansion is permitted, probe further
        while len(matching_results) < k and current_probe_idx < max_probe and allow_expansion:
            next_batch_size = min(4, max_probe - current_probe_idx)
            next_clusters = ranked_clusters[current_probe_idx : current_probe_idx + next_batch_size]
            current_probe_idx += next_batch_size
            partitions_visited += next_batch_size

            exp_cands = []
            for c_id in next_clusters:
                part = self.vector_index.partitions[c_id]
                if len(part.member_ids) > 0:
                    exp_cands.append(part.member_ids)

            if exp_cands:
                batch_arr = np.concatenate(exp_cands)
                unseen_mask = [r not in seen_ids for r in batch_arr]
                batch_arr = batch_arr[unseen_mask]
                seen_ids.update(batch_arr.tolist())

                if len(batch_arr) > 0:
                    total_candidates_examined += len(batch_arr)
                    batch_vecs = self.vector_index.get_vectors_batch(batch_arr)
                    batch_dists = vectorized_euclidean_distance(query_vector, batch_vecs)
                    total_distances_computed += len(batch_dists)

                    valid_mask = self.metadata_index.matches_batch(batch_arr, predicates)
                    for r_id, d in zip(batch_arr[valid_mask], batch_dists[valid_mask]):
                        matching_results.append((int(r_id), float(d)))

        # Sort all matching results by distance, then record ID
        matching_results.sort(key=lambda item: (item[1], item[0]))
        final_results = matching_results[:k]

        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
        return SearchResult(
            results=final_results,
            strategy_name="Strategy_B_PostFilter",
            candidates_examined=total_candidates_examined,
            partitions_visited=partitions_visited,
            eligible_records=len(matching_results),
            distances_computed=total_distances_computed,
            exact_fallback=False,
            latency_ms=elapsed_ms,
            diagnostics={
                "candidate_budget": candidate_budget,
                "allow_expansion": allow_expansion,
                "matches_found": len(matching_results),
            },
        )

    # -------------------------------------------------------------------------
    # Strategy C: Partition-Restricted Search
    # -------------------------------------------------------------------------
    def search_strategy_c_partition_restricted(
        self,
        query_vector: np.ndarray,
        predicates: Optional[List[Predicate]],
        k: int = 10,
        config: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """Strategy C: Partition-restricted search.

        1. Use metadata postings and partition statistics to prune partitions that
           cannot contain eligible records.
        2. Restrict vector search to eligible partitions, ranked by centroid proximity.
        3. Filter candidates within partitions and compute distances.
        4. If candidate set is insufficient, expand partitions or trigger correctness-preserving
           fallback to pre-filtered exact search.
        """
        start_ns = time.perf_counter_ns()
        cfg = config or {}

        max_probe = cfg.get("nprobe", max(8, self.vector_index.nprobe))
        allow_fallback = cfg.get("allow_fallback", True)

        # 1. Prune partitions using member metadata statistics
        eligible_cluster_ids = []
        for c_id, part in self.vector_index.partitions.items():
            if part.size > 0 and part.can_contain_predicates(predicates):
                eligible_cluster_ids.append(c_id)

        # If zero partitions can contain matches, or fallback condition
        if not eligible_cluster_ids:
            if allow_fallback:
                sub_res = self.search_strategy_a_pre_filter(query_vector, predicates, k, config)
                elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
                sub_res.strategy_name = "Strategy_C_PartitionRestricted"
                sub_res.exact_fallback = True
                sub_res.latency_ms = elapsed_ms
                sub_res.diagnostics["fallback_reason"] = "No candidate partitions satisfied metadata bounds"
                return sub_res
            else:
                elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
                return SearchResult(
                    results=[],
                    strategy_name="Strategy_C_PartitionRestricted",
                    candidates_examined=0,
                    partitions_visited=0,
                    eligible_records=0,
                    distances_computed=0,
                    exact_fallback=False,
                    latency_ms=elapsed_ms,
                    diagnostics={"reason": "No candidate partitions matched bounds"},
                )

        # 2. Rank eligible partitions by centroid distance to query vector
        eligible_cluster_arr = np.array(eligible_cluster_ids, dtype=np.int32)
        eligible_centroids = self.vector_index.centroids[eligible_cluster_arr]
        c_dists = vectorized_euclidean_distance(query_vector, eligible_centroids)
        ranked_order = np.argsort(c_dists)
        sorted_eligible_clusters = eligible_cluster_arr[ranked_order]

        # 3. Probe top clusters among eligible ones
        probes_to_visit = min(max_probe, len(sorted_eligible_clusters))
        visited_clusters = sorted_eligible_clusters[:probes_to_visit]

        total_candidates_examined = 0
        total_distances_computed = 0
        matching_results: List[Tuple[int, float]] = []

        for c_id in visited_clusters:
            part = self.vector_index.partitions[int(c_id)]
            members = part.member_ids
            total_candidates_examined += len(members)

            valid_mask = self.metadata_index.matches_batch(members, predicates)
            eligible_in_part = members[valid_mask]

            if len(eligible_in_part) > 0:
                vecs = self.vector_index.get_vectors_batch(eligible_in_part)
                dists = vectorized_euclidean_distance(query_vector, vecs)
                total_distances_computed += len(dists)

                for r_id, d in zip(eligible_in_part, dists):
                    matching_results.append((int(r_id), float(d)))

        # 4. Check sufficiency and fallback
        # If fewer than k results were found, but more eligible clusters remain, expand
        if len(matching_results) < k and probes_to_visit < len(sorted_eligible_clusters):
            remaining_clusters = sorted_eligible_clusters[probes_to_visit:]
            for c_id in remaining_clusters:
                part = self.vector_index.partitions[int(c_id)]
                members = part.member_ids
                total_candidates_examined += len(members)
                valid_mask = self.metadata_index.matches_batch(members, predicates)
                eligible_in_part = members[valid_mask]
                if len(eligible_in_part) > 0:
                    vecs = self.vector_index.get_vectors_batch(eligible_in_part)
                    dists = vectorized_euclidean_distance(query_vector, vecs)
                    total_distances_computed += len(dists)
                    for r_id, d in zip(eligible_in_part, dists):
                        matching_results.append((int(r_id), float(d)))
                if len(matching_results) >= k:
                    break

        # If still insufficient results found and fallback is permitted:
        total_actual_eligible = len(self.metadata_index.get_eligible_ids(predicates))
        if len(matching_results) < min(k, total_actual_eligible) and allow_fallback:
            fallback_res = self.search_strategy_a_pre_filter(query_vector, predicates, k, config)
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
            fallback_res.strategy_name = "Strategy_C_PartitionRestricted"
            fallback_res.exact_fallback = True
            fallback_res.latency_ms = elapsed_ms
            fallback_res.diagnostics["fallback_reason"] = (
                f"Found only {len(matching_results)} matches out of {total_actual_eligible} available"
            )
            return fallback_res

        # Rank final results
        matching_results.sort(key=lambda item: (item[1], item[0]))
        final_results = matching_results[:k]

        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
        return SearchResult(
            results=final_results,
            strategy_name="Strategy_C_PartitionRestricted",
            candidates_examined=total_candidates_examined,
            partitions_visited=len(visited_clusters),
            eligible_records=len(matching_results),
            distances_computed=total_distances_computed,
            exact_fallback=False,
            latency_ms=elapsed_ms,
            diagnostics={
                "eligible_partitions_count": len(eligible_cluster_ids),
                "probes_visited": len(visited_clusters),
            },
        )
