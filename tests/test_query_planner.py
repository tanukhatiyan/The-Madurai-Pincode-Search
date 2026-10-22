"""Unit tests for adaptive query planner decisions and cost model."""

import pytest

from src.dataset import generate_synthetic_dataset
from src.metadata_index import MetadataIndex, Predicate
from src.query_planner import AdaptiveQueryPlanner
from src.search_strategies import SearchEngine
from src.vector_index import CustomVectorIndex


@pytest.fixture(scope="module")
def setup_planner():
    bundle = generate_synthetic_dataset(num_records=1500, embedding_dim=16, seed=42)
    meta_idx = MetadataIndex(bundle.metadata)
    vec_idx = CustomVectorIndex(num_clusters=16, nprobe=4, random_state=42)
    vec_idx.build(bundle.metadata["id"].to_numpy(), bundle.embeddings, bundle.metadata)
    engine = SearchEngine(meta_idx, vec_idx)
    planner = AdaptiveQueryPlanner(engine)
    return planner, bundle


def test_planner_chooses_strategy_b_for_unfiltered(setup_planner):
    planner, bundle = setup_planner
    q_vec = bundle.embeddings[0]
    decision = planner.plan_query(q_vec, predicates=[])
    assert decision.strategy == "Strategy_B_PostFilter"
    assert decision.estimated_selectivity == 1.0
    assert "No metadata predicates specified" in decision.reason


def test_planner_chooses_strategy_a_for_highly_selective(setup_planner):
    planner, bundle = setup_planner
    q_vec = bundle.embeddings[1]
    # Triple predicate: Category + Zone + tight price range -> < 0.5%
    preds = [
        Predicate.category_eq("jasmine_flowers"),
        Predicate.zone_eq("Zone_Sellur"),
        Predicate.price_range(200.0, 220.0),
    ]
    decision = planner.plan_query(q_vec, predicates=preds)
    assert decision.strategy == "Strategy_A_PreFilter"
    assert decision.estimated_selectivity <= 0.05
    assert "High selectivity" in decision.reason


def test_planner_diagnostics_present(setup_planner):
    planner, bundle = setup_planner
    q_vec = bundle.embeddings[2]
    preds = [Predicate.category_eq("handloom_sarees")]
    res = planner.execute_query(q_vec, predicates=preds, k=10)

    diag = res.diagnostics.get("planner_decision")
    assert diag is not None
    assert "chosen_strategy" in diag
    assert "estimated_selectivity" in diag
    assert "expected_cost" in diag
    assert "reason" in diag
