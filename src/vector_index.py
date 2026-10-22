"""Custom partitioned vector index (IVF structure) using NumPy with KMeans clustering for centroids."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from sklearn.cluster import KMeans

from src.distance import (
    vectorized_euclidean_distance,
    vectorized_squared_euclidean_distance,
    exact_top_k,
)


@dataclass
class ClusterPartition:
    """Represents a single partition in the custom vector index."""
    cluster_id: int
    centroid: np.ndarray
    member_ids: np.ndarray  # 1D array of record IDs in this partition
    categories: Set[str] = field(default_factory=set)
    zones: Set[str] = field(default_factory=set)
    min_price: float = 0.0
    max_price: float = 0.0

    @property
    def size(self) -> int:
        return len(self.member_ids)

    def can_contain_predicates(self, predicates: Optional[List[Any]]) -> bool:
        """Check if any record in this cluster could possibly satisfy the predicates.

        Prevents false assumptions by checking exact metadata bounds of member records.
        """
        if not predicates:
            return True
        for p in predicates:
            if p.field == "category" and p.value not in self.categories:
                return False
            if p.field == "zone" and p.value not in self.zones:
                return False
            if p.field == "price":
                min_p, max_p = p.value
                if self.max_price < min_p or self.min_price > max_p:
                    return False
        return True


class CustomVectorIndex:
    """Custom partitioned vector index without external ANN library dependencies.

    Stores cluster centroids, postings lists of record IDs, and per-partition metadata bounds.
    Candidate retrieval and ranking are performed manually via NumPy.
    """

    def __init__(
        self,
        num_clusters: int = 64,
        nprobe: int = 28,
        random_state: int = 42,
    ):
        self.num_clusters = num_clusters
        self.nprobe = nprobe
        self.random_state = random_state

        self.centroids: Optional[np.ndarray] = None  # shape (num_clusters, D)
        self.partitions: Dict[int, ClusterPartition] = {}
        self.record_to_cluster: Dict[int, int] = {}
        self.vector_store: Optional[np.ndarray] = None  # direct vector storage indexed by record_id
        self.active_id_mask: Optional[np.ndarray] = None
        self.embedding_dim: int = 0
        self.total_records: int = 0

    def build(
        self,
        record_ids: np.ndarray,
        embeddings: np.ndarray,
        metadata_df: Any,
    ) -> None:
        """Build the partitioned index by clustering embeddings and partitioning records."""
        n_records, d = embeddings.shape
        self.embedding_dim = d
        self.total_records = n_records

        # Adjust num_clusters if n_records is small
        actual_clusters = min(self.num_clusters, max(1, n_records // 10))

        # 1. Use scikit-learn KMeans exclusively for centroid computation
        kmeans = KMeans(
            n_clusters=actual_clusters,
            init="k-means++",
            n_init=1,
            max_iter=50,
            random_state=self.random_state,
        )
        labels = kmeans.fit_predict(embeddings)
        self.centroids = kmeans.cluster_centers_.astype(np.float32)

        # 2. Setup direct vector store indexed by record ID
        max_id = int(np.max(record_ids)) if n_records > 0 else 0
        self.vector_store = np.zeros((max_id + 1, d), dtype=np.float32)
        self.active_id_mask = np.zeros(max_id + 1, dtype=bool)

        self.vector_store[record_ids] = embeddings
        self.active_id_mask[record_ids] = True

        # Extract metadata columns for partition summary statistics
        id_to_meta: Dict[int, Tuple[str, str, float]] = {}
        for _, row in metadata_df.iterrows():
            r_id = int(row["id"])
            id_to_meta[r_id] = (str(row["category"]), str(row["zone"]), float(row["price"]))

        # 3. Build cluster partitions
        self.partitions.clear()
        self.record_to_cluster.clear()

        for c_id in range(actual_clusters):
            c_mask = (labels == c_id)
            c_ids = record_ids[c_mask]

            if len(c_ids) == 0:
                self.partitions[c_id] = ClusterPartition(
                    cluster_id=c_id,
                    centroid=self.centroids[c_id],
                    member_ids=np.empty(0, dtype=np.int32),
                )
                continue

            cats: Set[str] = set()
            zones: Set[str] = set()
            prices: List[float] = []

            for r_id in c_ids:
                self.record_to_cluster[int(r_id)] = c_id
                cat, zone, price = id_to_meta[r_id]
                cats.add(cat)
                zones.add(zone)
                prices.append(price)

            self.partitions[c_id] = ClusterPartition(
                cluster_id=c_id,
                centroid=self.centroids[c_id],
                member_ids=np.sort(c_ids),
                categories=cats,
                zones=zones,
                min_price=min(prices),
                max_price=max(prices),
            )

    def get_vector(self, record_id: int) -> Optional[np.ndarray]:
        """Retrieve the embedding vector for a record ID."""
        if self.active_id_mask is not None and 0 <= record_id < len(self.active_id_mask):
            if self.active_id_mask[record_id]:
                return self.vector_store[record_id]
        return None

    def get_vectors_batch(self, record_ids: np.ndarray) -> np.ndarray:
        """Retrieve embedding matrix for an array of record IDs."""
        if len(record_ids) == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        return self.vector_store[record_ids]

    def rank_clusters(self, query_vector: np.ndarray) -> np.ndarray:
        """Rank all cluster centroids by Euclidean distance to query vector.

        Returns:
            1D array of cluster indices sorted nearest to farthest.
        """
        if self.centroids is None or len(self.centroids) == 0:
            return np.empty(0, dtype=np.int32)
        distances = vectorized_euclidean_distance(query_vector, self.centroids)
        return np.argsort(distances)

    def retrieve_candidates(
        self,
        query_vector: np.ndarray,
        nprobe: Optional[int] = None,
        candidate_budget: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """Retrieve candidate record IDs from the nearest nprobe clusters.

        Args:
            query_vector: Query vector of shape (D,).
            nprobe: Number of closest centroids to probe (defaults to self.nprobe).
            candidate_budget: Optional max count of candidates to collect.

        Returns:
            Tuple of (candidate_record_ids, partitions_visited).
        """
        probe_count = nprobe if nprobe is not None else self.nprobe
        probe_count = min(probe_count, len(self.partitions))

        ranked_clusters = self.rank_clusters(query_vector)
        chosen_clusters = ranked_clusters[:probe_count]

        candidates_list = []
        collected = 0
        partitions_visited = 0

        for c_id in chosen_clusters:
            part = self.partitions[c_id]
            partitions_visited += 1
            if len(part.member_ids) > 0:
                candidates_list.append(part.member_ids)
                collected += len(part.member_ids)
                if candidate_budget is not None and collected >= candidate_budget:
                    break

        if not candidates_list:
            return np.empty(0, dtype=np.int32), partitions_visited

        candidates = np.concatenate(candidates_list)
        if candidate_budget is not None and len(candidates) > candidate_budget:
            candidates = candidates[:candidate_budget]

        return candidates, partitions_visited

    def delete_records(self, record_ids: List[int]) -> None:
        """Mark deleted records and update cluster postings."""
        del_set = set(record_ids)
        if not del_set:
            return

        for r_id in del_set:
            if 0 <= r_id < len(self.active_id_mask):
                self.active_id_mask[r_id] = False
            c_id = self.record_to_cluster.pop(r_id, None)
            if c_id is not None and c_id in self.partitions:
                part = self.partitions[c_id]
                part.member_ids = part.member_ids[~np.isin(part.member_ids, list(del_set))]

    def rebuild_with_data(
        self,
        record_ids: np.ndarray,
        embeddings: np.ndarray,
        metadata_df: Any,
    ) -> None:
        """Completely rebuild index with updated data."""
        self.build(record_ids, embeddings, metadata_df)
