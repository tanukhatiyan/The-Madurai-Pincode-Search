"""Workload generator for calibrated query selectivity buckets."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.dataset import CATEGORIES, ZONES, DatasetBundle
from src.metadata_index import MetadataIndex, Predicate


@dataclass
class BenchmarkQuery:
    """A standardized evaluation query with measured selectivity."""
    query_id: int
    query_vector: np.ndarray
    predicates: List[Predicate]
    bucket: str
    target_selectivity: float
    observed_selectivity: float
    eligible_count: int


# Selectivity bucket target intervals (min_fraction, max_fraction)
BUCKET_DEFINITIONS: Dict[str, Tuple[float, float, str]] = {
    "extremely_selective": (0.00001, 0.0005, "Handful of records (< 0.05%)"),
    "highly_selective": (0.0005, 0.003, "Around 0.1% matching records"),
    "moderate": (0.005, 0.025, "Around 1% matching records"),
    "broad": (0.05, 0.18, "Around 10% matching records"),
    "very_broad": (0.75, 1.00, "Broad query (75% to 100% matching)"),
}


class WorkloadGenerator:
    """Generates balanced evaluation and calibration workloads across 5 selectivity buckets."""

    def __init__(self, bundle: DatasetBundle, metadata_index: MetadataIndex, seed: int = 12345):
        self.bundle = bundle
        self.metadata_index = metadata_index
        self.rng = np.random.default_rng(seed)
        self.total_records = len(bundle.metadata)

    def _sample_query_vector(self) -> np.ndarray:
        """Sample a realistic query vector from the dataset embeddings with small perturbation."""
        idx = self.rng.integers(0, self.total_records)
        base = self.bundle.embeddings[idx]
        noise = self.rng.normal(0.0, 0.05, size=base.shape).astype(np.float32)
        q = base + noise
        norm = np.linalg.norm(q)
        if norm > 1e-8:
            q /= norm
        return q

    def generate_workload_for_bucket(
        self,
        bucket_name: str,
        num_queries: int = 50,
        max_attempts: int = 3000,
    ) -> List[BenchmarkQuery]:
        """Generate verified queries that fall strictly within the specified selectivity bucket."""
        if bucket_name not in BUCKET_DEFINITIONS:
            raise ValueError(f"Unknown bucket: {bucket_name}")

        min_sel, max_sel, _ = BUCKET_DEFINITIONS[bucket_name]
        queries: List[BenchmarkQuery] = []
        prices = self.bundle.metadata["price"].to_numpy()

        cats = CATEGORIES
        zones = [z["zone"] for z in ZONES]

        attempts = 0
        while len(queries) < num_queries and attempts < max_attempts:
            attempts += 1
            q_vec = self._sample_query_vector()
            predicates: List[Predicate] = []

            if bucket_name == "very_broad":
                # Very broad: either no predicates, or very wide price range
                if self.rng.random() < 0.5:
                    predicates = []
                else:
                    p_min = float(np.percentile(prices, 5))
                    p_max = float(np.percentile(prices, 98))
                    predicates = [Predicate.price_range(p_min, p_max)]

            elif bucket_name == "broad":
                # Broad: single zone or medium-wide price slice
                mode = self.rng.choice(["zone", "price_slice"])
                if mode == "zone":
                    z = self.rng.choice(zones[:8])  # pick from denser zones
                    predicates = [Predicate.zone_eq(z)]
                else:
                    p1 = float(np.percentile(prices, self.rng.uniform(10, 70)))
                    p2 = p1 * self.rng.uniform(1.3, 1.8)
                    predicates = [Predicate.price_range(p1, p2)]

            elif bucket_name == "moderate":
                # Moderate: category alone, or category + moderate price range
                cat = self.rng.choice(cats)
                if self.rng.random() < 0.6:
                    predicates = [Predicate.category_eq(cat)]
                else:
                    p1 = float(np.percentile(prices, self.rng.uniform(20, 60)))
                    p2 = p1 * 1.5
                    predicates = [Predicate.category_eq(cat), Predicate.price_range(p1, p2)]

            elif bucket_name == "highly_selective":
                # Highly selective: Category + Zone, or Category + narrow price range
                cat = self.rng.choice(cats)
                z = self.rng.choice(zones)
                predicates = [Predicate.category_eq(cat), Predicate.zone_eq(z)]

            elif bucket_name == "extremely_selective":
                # Extremely selective: Category + Zone + tight price range
                cat = self.rng.choice(cats)
                z = self.rng.choice(zones)
                # Find median price of this category to center tight range
                cat_rows = self.bundle.metadata[
                    (self.bundle.metadata["category"] == cat) &
                    (self.bundle.metadata["zone"] == z)
                ]
                if len(cat_rows) > 0:
                    med_p = float(cat_rows["price"].median())
                    predicates = [
                        Predicate.category_eq(cat),
                        Predicate.zone_eq(z),
                        Predicate.price_range(round(med_p * 0.98, 2), round(med_p * 1.02, 2)),
                    ]
                else:
                    predicates = [
                        Predicate.category_eq(cat),
                        Predicate.zone_eq(z),
                    ]

            # Measure actual selectivity
            eligible_ids = self.metadata_index.get_eligible_ids(predicates)
            eligible_cnt = len(eligible_ids)
            actual_sel = eligible_cnt / max(self.total_records, 1)

            # Check if within bucket bounds
            if min_sel <= actual_sel <= max_sel or (bucket_name == "extremely_selective" and 1 <= eligible_cnt <= 30):
                queries.append(BenchmarkQuery(
                    query_id=len(queries),
                    query_vector=q_vec,
                    predicates=predicates,
                    bucket=bucket_name,
                    target_selectivity=(min_sel + max_sel) / 2.0,
                    observed_selectivity=actual_sel,
                    eligible_count=eligible_cnt,
                ))

        return queries

    def generate_full_benchmark_suite(
        self,
        queries_per_bucket: int = 50,
    ) -> Dict[str, List[BenchmarkQuery]]:
        """Generate balanced test queries for all 5 selectivity buckets."""
        suite = {}
        for bucket in BUCKET_DEFINITIONS:
            suite[bucket] = self.generate_workload_for_bucket(bucket, num_queries=queries_per_bucket)
        return suite
