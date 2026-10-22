"""Unit tests for dataset mutation workflows and deleted-record exclusion."""

import numpy as np
import pytest

from src.dataset import generate_synthetic_dataset
from src.metadata_index import MetadataIndex, Predicate
from src.mutations import MutationManager
from src.query_planner import AdaptiveQueryPlanner
from src.search_strategies import SearchEngine
from src.vector_index import CustomVectorIndex


def test_mutation_workflow():
    bundle = generate_synthetic_dataset(num_records=1000, embedding_dim=16, seed=42)
    meta_idx = MetadataIndex(bundle.metadata)
    vec_idx = CustomVectorIndex(num_clusters=16, nprobe=4, random_state=42)
    vec_idx.build(bundle.metadata["id"].to_numpy(), bundle.embeddings, bundle.metadata)
    engine = SearchEngine(meta_idx, vec_idx)
    planner = AdaptiveQueryPlanner(engine)

    manager = MutationManager(bundle, meta_idx, vec_idx, planner, seed=999)
    res = manager.apply_mutations()

    # 1. Check deletion
    assert res.num_deleted > 0
    assert manager.verify_deleted_records_excluded(res.deleted_ids) is True

    # 2. Check insertion (5% of 1000 = 50)
    assert res.num_inserted == 50
    assert res.new_total_records == 1000 - res.num_deleted + 50

    # 3. Check search query never returns deleted records
    q_vec = bundle.embeddings[0]
    search_res = engine.search_strategy_a_pre_filter(q_vec, predicates=None, k=50)
    returned_ids = set(r[0] for r in search_res.results)
    for d_id in res.deleted_ids:
        assert d_id not in returned_ids
