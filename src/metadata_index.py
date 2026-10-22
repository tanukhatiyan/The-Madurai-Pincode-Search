"""Custom metadata indexing, predicate filtering, and analytical selectivity estimation."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Predicate:
    """Represents a single query predicate."""
    field: str  # 'category', 'zone', or 'price'
    op: str     # '==' for categorical, 'range' for price
    value: Any = None  # str for '==', (min_price, max_price) for 'range'

    @classmethod
    def category_eq(cls, val: str) -> "Predicate":
        return cls(field="category", op="==", value=val)

    @classmethod
    def zone_eq(cls, val: str) -> "Predicate":
        return cls(field="zone", op="==", value=val)

    @classmethod
    def price_range(cls, min_price: float, max_price: float) -> "Predicate":
        return cls(field="price", op="range", value=(float(min_price), float(max_price)))


class MetadataIndex:
    """Inverted and sorted index for fast metadata filtering and selectivity estimation."""

    def __init__(self, metadata_df: pd.DataFrame):
        self._build_index(metadata_df)

    def _build_index(self, df: pd.DataFrame) -> None:
        """Construct inverted postings, sorted price arrays, and cardinality histograms."""
        self.df = df.copy()
        self.total_records = len(df)
        self.active_id_set: Set[int] = set(df["id"].tolist())
        self.active_ids_sorted: np.ndarray = np.sort(df["id"].to_numpy(dtype=np.int32))

        # 1. Category postings: category -> sorted np.ndarray of record IDs
        self.category_postings: Dict[str, np.ndarray] = {}
        for cat, group in df.groupby("category", observed=False):
            self.category_postings[str(cat)] = np.sort(group["id"].to_numpy(dtype=np.int32))

        # 2. Zone postings: zone -> sorted np.ndarray of record IDs
        self.zone_postings: Dict[str, np.ndarray] = {}
        for z, group in df.groupby("zone", observed=False):
            self.zone_postings[str(z)] = np.sort(group["id"].to_numpy(dtype=np.int32))

        # 3. Sorted price array for O(log N) range queries
        price_order = np.argsort(df["price"].to_numpy())
        self.sorted_prices: np.ndarray = df["price"].to_numpy()[price_order]
        self.sorted_price_ids: np.ndarray = df["id"].to_numpy(dtype=np.int32)[price_order]

        # 4. Joint category-zone frequency table for accurate multi-predicate selectivity estimation
        self.joint_cat_zone_counts: Dict[Tuple[str, str], int] = {}
        joint_crosstab = pd.crosstab(df["category"], df["zone"])
        for cat in joint_crosstab.index:
            for z in joint_crosstab.columns:
                cnt = int(joint_crosstab.loc[cat, z])
                if cnt > 0:
                    self.joint_cat_zone_counts[(str(cat), str(z))] = cnt

        # 5. Direct array lookups indexed by record ID for O(1) predicate evaluation
        # Determine maximum ID to size direct lookup arrays
        max_id = int(df["id"].max()) if len(df) > 0 else 0
        self._id_to_cat = np.empty(max_id + 1, dtype=object)
        self._id_to_zone = np.empty(max_id + 1, dtype=object)
        self._id_to_price = np.full(max_id + 1, -1.0, dtype=np.float32)
        self._id_active = np.zeros(max_id + 1, dtype=bool)

        ids = df["id"].to_numpy(dtype=np.int32)
        self._id_to_cat[ids] = df["category"].to_numpy()
        self._id_to_zone[ids] = df["zone"].to_numpy()
        self._id_to_price[ids] = df["price"].to_numpy(dtype=np.float32)
        self._id_active[ids] = True

    @property
    def num_active_records(self) -> int:
        return len(self.active_id_set)

    def is_active(self, record_id: int) -> bool:
        if 0 <= record_id < len(self._id_active):
            return bool(self._id_active[record_id])
        return False

    def get_record_metadata(self, record_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a specific record ID."""
        if not self.is_active(record_id):
            return None
        return {
            "id": record_id,
            "category": self._id_to_cat[record_id],
            "zone": self._id_to_zone[record_id],
            "price": float(self._id_to_price[record_id]),
        }

    def evaluate_predicate_ids(self, pred: Predicate) -> np.ndarray:
        """Retrieve sorted record IDs matching a single predicate."""
        if pred.field == "category":
            if pred.op == "==":
                return self.category_postings.get(str(pred.value), np.empty(0, dtype=np.int32))
            raise ValueError(f"Unsupported operator {pred.op} for category")

        elif pred.field == "zone":
            if pred.op == "==":
                return self.zone_postings.get(str(pred.value), np.empty(0, dtype=np.int32))
            raise ValueError(f"Unsupported operator {pred.op} for zone")

        elif pred.field == "price":
            if pred.op == "range":
                min_p, max_p = pred.value
                # Binary search in sorted prices
                l_idx = np.searchsorted(self.sorted_prices, min_p, side="left")
                r_idx = np.searchsorted(self.sorted_prices, max_p, side="right")
                matched_ids = self.sorted_price_ids[l_idx:r_idx]
                return np.sort(matched_ids)
            raise ValueError(f"Unsupported operator {pred.op} for price")

        raise ValueError(f"Unknown predicate field: {pred.field}")

    def get_eligible_ids(self, predicates: Optional[List[Predicate]]) -> np.ndarray:
        """Find all record IDs satisfying all predicates (AND combination).

        Args:
            predicates: List of up to 3 predicates. If empty or None, returns all active IDs.

        Returns:
            Sorted 1D array of matching record IDs.
        """
        if not predicates:
            return self.active_ids_sorted.copy()

        if len(predicates) > 3:
            raise ValueError(f"Maximum 3 AND predicates supported, got {len(predicates)}")

        # Sort predicates by estimated posting size for fast early-exit intersection
        evaluated_lists = [self.evaluate_predicate_ids(p) for p in predicates]
        # Sort by array length ascending
        evaluated_lists.sort(key=lambda arr: len(arr))

        result = evaluated_lists[0]
        for next_arr in evaluated_lists[1:]:
            if len(result) == 0:
                return np.empty(0, dtype=np.int32)
            result = np.intersect1d(result, next_arr, assume_unique=True)

        return result

    def matches_predicates(self, record_id: int, predicates: Optional[List[Predicate]]) -> bool:
        """Check if a specific record ID satisfies all predicates in O(1)."""
        if not self.is_active(record_id):
            return False
        if not predicates:
            return True

        for p in predicates:
            if p.field == "category":
                if self._id_to_cat[record_id] != p.value:
                    return False
            elif p.field == "zone":
                if self._id_to_zone[record_id] != p.value:
                    return False
            elif p.field == "price":
                p_min, p_max = p.value
                p_val = self._id_to_price[record_id]
                if not (p_min <= p_val <= p_max):
                    return False
        return True

    def matches_batch(self, record_ids: np.ndarray, predicates: Optional[List[Predicate]]) -> np.ndarray:
        """Vectorized filter check for an array of candidate record IDs.

        Args:
            record_ids: 1D array of candidate record IDs.
            predicates: List of predicates.

        Returns:
            Boolean array of shape (len(record_ids),) where True means all predicates match.
        """
        n = len(record_ids)
        if n == 0:
            return np.empty(0, dtype=bool)
        if not predicates:
            return np.ones(n, dtype=bool)

        mask = np.ones(n, dtype=bool)
        valid_id_mask = (record_ids >= 0) & (record_ids < len(self._id_active))
        mask &= valid_id_mask
        if not np.any(mask):
            return mask

        active_submask = self._id_active[record_ids[mask]]
        mask[mask] &= active_submask

        for p in predicates:
            if not np.any(mask):
                break
            valid_indices = record_ids[mask]
            if p.field == "category":
                matches = (self._id_to_cat[valid_indices] == p.value)
                mask[mask] &= matches
            elif p.field == "zone":
                matches = (self._id_to_zone[valid_indices] == p.value)
                mask[mask] &= matches
            elif p.field == "price":
                p_min, p_max = p.value
                prices = self._id_to_price[valid_indices]
                matches = (prices >= p_min) & (prices <= p_max)
                mask[mask] &= matches

        return mask

    def estimate_selectivity(self, predicates: Optional[List[Predicate]]) -> Tuple[float, int]:
        """Estimate filter selectivity without scanning the dataset.

        Uses inverted posting cardinalities, joint distribution tables, and price binary search.

        Returns:
            Tuple of (estimated_selectivity, estimated_matching_records).
        """
        total = self.num_active_records
        if total == 0 or not predicates:
            return 1.0, total

        # Check for individual predicate estimations
        cat_pred = next((p for p in predicates if p.field == "category"), None)
        zone_pred = next((p for p in predicates if p.field == "zone"), None)
        price_pred = next((p for p in predicates if p.field == "price"), None)

        # 1. Base count from category and zone
        if cat_pred and zone_pred:
            # Joint lookup
            pair = (str(cat_pred.value), str(zone_pred.value))
            joint_count = self.joint_cat_zone_counts.get(pair, 0)
            base_count = joint_count
        elif cat_pred:
            base_count = len(self.category_postings.get(str(cat_pred.value), []))
        elif zone_pred:
            base_count = len(self.zone_postings.get(str(zone_pred.value), []))
        else:
            base_count = total

        # 2. Price factor (O(log N) lookup in sorted prices, no scan)
        if price_pred:
            min_p, max_p = price_pred.value
            l_idx = np.searchsorted(self.sorted_prices, min_p, side="left")
            r_idx = np.searchsorted(self.sorted_prices, max_p, side="right")
            price_fraction = (r_idx - l_idx) / max(total, 1)
            estimated_count = int(round(base_count * price_fraction))
        else:
            estimated_count = base_count

        estimated_count = max(0, min(total, estimated_count))
        selectivity = estimated_count / total
        return float(selectivity), int(estimated_count)

    def delete_records(self, record_ids_to_delete: Union[List[int], np.ndarray, Set[int]]) -> None:
        """Remove records from active postings and re-index.

        Supports dynamic dataset mutations cleanly.
        """
        del_set = set(record_ids_to_delete)
        if not del_set:
            return

        # Filter the internal DataFrame and rebuild
        remaining_df = self.df[~self.df["id"].isin(del_set)].copy()
        self._build_index(remaining_df)

    def insert_records(self, new_records_df: pd.DataFrame) -> None:
        """Insert new records and rebuild indexes.

        Supports dynamic dataset mutations cleanly.
        """
        if len(new_records_df) == 0:
            return
        combined_df = pd.concat([self.df, new_records_df], ignore_index=True)
        self._build_index(combined_df)
