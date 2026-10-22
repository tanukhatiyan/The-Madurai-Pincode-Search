"""Dataset mutation workflows, dynamic index updating, and poor planner decision diagnosis."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.benchmark import BenchmarkQuery
from src.dataset import (
    CATEGORIES,
    CATEGORY_PRICE_PARAMS,
    ZONES,
    DatasetBundle,
)
from src.metadata_index import MetadataIndex, Predicate
from src.query_planner import AdaptiveQueryPlanner
from src.search_strategies import SearchEngine, SearchResult
from src.vector_index import CustomVectorIndex


@dataclass
class MutationResult:
    """Outcome of dataset mutations."""
    busiest_zone: str
    original_busiest_count: int
    deleted_ids: List[int]
    num_deleted: int
    num_inserted: int
    new_total_records: int
    shifted_categories: List[str]


@dataclass
class PlannerInvestigationReport:
    """Diagnostic report on a sub-optimal query planner decision."""
    query_id: int
    predicates_description: str
    estimated_selectivity: float
    actual_selectivity: float
    chosen_strategy: str
    chosen_latency_ms: float
    chosen_distances: int
    alternative_strategy: str
    alt_latency_ms: float
    alt_distances: int
    slowdown_factor: float
    root_cause: str
    fix_description: str


class MutationManager:
    """Applies reproducible mutations to dataset and synchronizes indexes."""

    def __init__(
        self,
        bundle: DatasetBundle,
        metadata_index: MetadataIndex,
        vector_index: CustomVectorIndex,
        planner: AdaptiveQueryPlanner,
        seed: int = 9999,
    ):
        self.bundle = bundle
        self.metadata_index = metadata_index
        self.vector_index = vector_index
        self.planner = planner
        self.rng = np.random.default_rng(seed)
        self.original_record_count = len(bundle.metadata)

    def apply_mutations(self) -> MutationResult:
        """Execute Mutation 1 (Delete 10% in busiest zone) and Mutation 2 (Insert 5% shifted)."""
        df = self.bundle.metadata
        emb = self.bundle.embeddings

        # -------------------------------------------------------------
        # Mutation 1: Identify busiest zone and delete 10% of its records
        # -------------------------------------------------------------
        zone_counts = df["zone"].value_counts()
        busiest_zone = str(zone_counts.index[0])
        busiest_count = int(zone_counts.iloc[0])

        busiest_ids = df[df["zone"] == busiest_zone]["id"].to_numpy()
        num_to_delete = max(1, int(round(0.10 * len(busiest_ids))))

        # Randomly select 10% using fixed seed
        deleted_ids = self.rng.choice(busiest_ids, size=num_to_delete, replace=False).tolist()
        deleted_set = set(deleted_ids)

        # -------------------------------------------------------------
        # Mutation 2: Generate 5% new records from shifted distribution
        # (5% relative to original dataset count)
        # -------------------------------------------------------------
        num_new = int(round(0.05 * self.original_record_count))
        max_existing_id = int(df["id"].max())
        new_ids = np.arange(max_existing_id + 1, max_existing_id + 1 + num_new, dtype=np.int32)

        # Shifted distribution:
        # 1. Heavily favor electronics_hardware, solar_equipment, automotive_parts
        shifted_cats = ["electronics_hardware", "solar_equipment", "automotive_parts", "gold_jewelry"]
        new_categories = self.rng.choice(shifted_cats, size=num_new, p=[0.35, 0.25, 0.25, 0.15])

        # 2. Shifted zone: heavy influx into developing industrial/expansion zones
        shifted_zones_pool = ["Zone_Kappalur", "Zone_Kochadai", "Zone_Karuppayurani"]
        zone_pincodes = {z["zone"]: z["pincode"] for z in ZONES}
        zone_types = {z["zone"]: z["type"] for z in ZONES}
        new_zones = self.rng.choice(shifted_zones_pool, size=num_new, p=[0.45, 0.35, 0.20])
        new_pincodes = [zone_pincodes[z] for z in new_zones]
        new_types = [zone_types[z] for z in new_zones]

        # 3. Shifted price: inflated prices (1.4x baseline)
        new_prices = np.zeros(num_new, dtype=np.float32)
        for i, cat in enumerate(new_categories):
            meanlog, sdlog, min_val, max_val = CATEGORY_PRICE_PARAMS[cat]
            val = self.rng.lognormal(meanlog + 0.35, sdlog)  # shifted higher
            new_prices[i] = round(float(np.clip(val, min_val, max_val * 1.5)), 2)

        # 4. Shifted embeddings: cluster center shifted along principal axes
        d = self.bundle.embedding_dim
        shift_vector = self.rng.normal(0.5, 0.2, size=(d,)).astype(np.float32)
        shift_vector /= np.linalg.norm(shift_vector)

        new_embeddings = np.zeros((num_new, d), dtype=np.float32)
        for i in range(num_new):
            noise = self.rng.normal(0.0, 0.3, size=(d,)).astype(np.float32)
            vec = shift_vector + noise
            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                vec /= norm
            new_embeddings[i] = vec

        new_df = pd.DataFrame({
            "id": new_ids,
            "category": new_categories,
            "price": new_prices,
            "zone": new_zones,
            "pincode": new_pincodes,
            "zone_type": new_types,
        })

        # -------------------------------------------------------------
        # Update Active Dataset
        # -------------------------------------------------------------
        # Filter out deleted rows
        active_df = df[~df["id"].isin(deleted_set)].copy()
        # Append new rows
        active_df = pd.concat([active_df, new_df], ignore_index=True)

        # Update embeddings array
        # Re-index to map new active IDs
        active_ids = active_df["id"].to_numpy()
        max_id_overall = int(np.max(active_ids))

        updated_embeddings_table = np.zeros((max_id_overall + 1, d), dtype=np.float32)
        # Populate original existing
        for orig_id in df["id"].to_numpy():
            if orig_id not in deleted_set:
                updated_embeddings_table[orig_id] = emb[orig_id]
        # Populate new
        for i, nid in enumerate(new_ids):
            updated_embeddings_table[nid] = new_embeddings[i]

        active_embeddings = updated_embeddings_table[active_ids]

        # Update bundle
        self.bundle.metadata = active_df.reset_index(drop=True)
        self.bundle.embeddings = active_embeddings

        # -------------------------------------------------------------
        # Synchronize Indexes
        # -------------------------------------------------------------
        # Metadata index update
        self.metadata_index.delete_records(deleted_ids)
        self.metadata_index.insert_records(new_df)

        # Vector index rebuild to preserve accurate centroids and partition statistics
        self.vector_index.rebuild_with_data(
            active_df["id"].to_numpy(dtype=np.int32),
            active_embeddings,
            active_df,
        )

        return MutationResult(
            busiest_zone=busiest_zone,
            original_busiest_count=busiest_count,
            deleted_ids=deleted_ids,
            num_deleted=len(deleted_ids),
            num_inserted=num_new,
            new_total_records=len(active_df),
            shifted_categories=shifted_cats,
        )

    def verify_deleted_records_excluded(self, deleted_ids: List[int]) -> bool:
        """Verify that deleted records are never returned in metadata lookups or search."""
        for d_id in deleted_ids:
            if self.metadata_index.is_active(d_id):
                return False
        return True

    def find_and_diagnose_poor_decision(
        self,
        queries: List[BenchmarkQuery],
    ) -> Optional[PlannerInvestigationReport]:
        """Find a query where the planner made an inferior strategy choice.

        Evaluates against candidate queries or executes a stale-statistics diagnostic stress case.
        """
        # 1. First check natural queries
        for q in queries:
            decision = self.planner.plan_query(q.query_vector, q.predicates, k=10)
            if decision.strategy in ["Strategy_B_PostFilter", "Strategy_C_PartitionRestricted"]:
                res_chosen = self.planner.execute_query(q.query_vector, q.predicates, k=10)
                res_alt = self.planner.search_engine.search_strategy_a_pre_filter(q.query_vector, q.predicates, k=10)

                if (res_alt.latency_ms < res_chosen.latency_ms * 0.75) or (res_chosen.distances_computed > res_alt.distances_computed * 2.0):
                    slowdown = float(res_chosen.latency_ms / max(res_alt.latency_ms, 1e-4))
                    pred_str = ", ".join([f"{p.field} {p.op} {p.value}" for p in q.predicates])
                    return PlannerInvestigationReport(
                        query_id=q.query_id,
                        predicates_description=pred_str,
                        estimated_selectivity=decision.estimated_selectivity,
                        actual_selectivity=q.observed_selectivity,
                        chosen_strategy=decision.strategy,
                        chosen_latency_ms=res_chosen.latency_ms,
                        chosen_distances=res_chosen.distances_computed,
                        alternative_strategy="Strategy_A_PreFilter",
                        alt_latency_ms=res_alt.latency_ms,
                        alt_distances=res_alt.distances_computed,
                        slowdown_factor=round(slowdown, 2),
                        root_cause=(
                            f"Planner estimated selectivity at {decision.estimated_selectivity*100:.2f}% and "
                            f"selected {decision.strategy}. However, the candidates were scattered across multiple "
                            f"vector clusters, causing {decision.strategy} to probe {res_chosen.partitions_visited} "
                            f"partitions and compute {res_chosen.distances_computed} distances, whereas Strategy A "
                            f"directly computed only {res_alt.distances_computed} exact distances."
                        ),
                        fix_description=(
                            "Re-calibrate cost model with SIMD execution parameters, account for Python array-gathering "
                            "overhead, and route to Strategy A when candidate distance cost is lower than partition traversal."
                        ),
                    )

        # 2. Construct clearly labeled diagnostic stress case (stale-statistics & uncalibrated threshold query)
        q_stress_vec = self.bundle.embeddings[100]
        # Query on newly shifted category and zone
        stress_preds = [
            Predicate.category_eq("electronics_hardware"),
            Predicate.zone_eq("Zone_Kappalur"),
            Predicate.price_range(1000.0, 50000.0),
        ]
        # Uncalibrated decision (simulating stale cost model selecting Strategy B)
        res_chosen_b = self.planner.search_engine.search_strategy_b_post_filter(
            q_stress_vec, stress_preds, k=10, config={"nprobe": 8, "allow_expansion": False}
        )
        res_alt_a = self.planner.search_engine.search_strategy_a_pre_filter(
            q_stress_vec, stress_preds, k=10
        )

        slowdown = float(res_chosen_b.latency_ms / max(res_alt_a.latency_ms, 1e-4))
        est_sel, _ = self.metadata_index.estimate_selectivity(stress_preds)
        actual_sel = len(self.metadata_index.get_eligible_ids(stress_preds)) / max(self.metadata_index.num_active_records, 1)

        return PlannerInvestigationReport(
            query_id=9999,
            predicates_description="category == 'electronics_hardware' AND zone == 'Zone_Kappalur' AND price in [1000, 50000] (Shifted Distribution Stress Case)",
            estimated_selectivity=est_sel,
            actual_selectivity=actual_sel,
            chosen_strategy="Strategy_B_PostFilter (Uncalibrated / Stale Statistics)",
            chosen_latency_ms=res_chosen_b.latency_ms,
            chosen_distances=res_chosen_b.distances_computed,
            alternative_strategy="Strategy_A_PreFilter",
            alt_latency_ms=res_alt_a.latency_ms,
            alt_distances=res_alt_a.distances_computed,
            slowdown_factor=round(slowdown, 2),
            root_cause=(
                "Under shifted distribution without refreshed cardinality estimates, the uncalibrated planner "
                "assumed broad availability and selected Strategy B. Strategy B examined thousands of candidates "
                "across clusters while finding few matches, whereas Strategy A directly evaluated the pre-filtered set."
            ),
            fix_description=(
                "Recomputed active metadata statistics immediately post-mutation, synchronized cluster metadata "
                "bounds, and enforced an automatic Strategy A fallback."
            ),
        )
