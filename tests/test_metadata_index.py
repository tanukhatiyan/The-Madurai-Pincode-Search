"""Unit tests for metadata index, predicate evaluation, and selectivity estimation."""

import numpy as np
import pandas as pd
import pytest

from src.dataset import generate_synthetic_dataset
from src.metadata_index import MetadataIndex, Predicate


@pytest.fixture
def sample_index():
    bundle = generate_synthetic_dataset(num_records=1000, embedding_dim=16, seed=42)
    return MetadataIndex(bundle.metadata), bundle.metadata


def test_category_predicate(sample_index):
    idx, df = sample_index
    cat = "jasmine_flowers"
    pred = [Predicate.category_eq(cat)]
    matched_ids = idx.get_eligible_ids(pred)

    expected_ids = np.sort(df[df["category"] == cat]["id"].to_numpy())
    assert np.array_equal(matched_ids, expected_ids)


def test_zone_predicate(sample_index):
    idx, df = sample_index
    z = "Zone_Meenakshi_Amman"
    pred = [Predicate.zone_eq(z)]
    matched_ids = idx.get_eligible_ids(pred)

    expected_ids = np.sort(df[df["zone"] == z]["id"].to_numpy())
    assert np.array_equal(matched_ids, expected_ids)


def test_price_range_inclusive_boundaries(sample_index):
    idx, df = sample_index
    p_min = 200.0
    p_max = 800.0
    pred = [Predicate.price_range(p_min, p_max)]
    matched_ids = idx.get_eligible_ids(pred)

    expected_ids = np.sort(df[(df["price"] >= p_min) & (df["price"] <= p_max)]["id"].to_numpy())
    assert np.array_equal(matched_ids, expected_ids)


def test_three_predicate_and_combination(sample_index):
    idx, df = sample_index
    cat = "handloom_sarees"
    z = "Zone_Sellur"
    p_min = 500.0
    p_max = 10000.0

    preds = [
        Predicate.category_eq(cat),
        Predicate.zone_eq(z),
        Predicate.price_range(p_min, p_max),
    ]
    matched_ids = idx.get_eligible_ids(preds)

    expected = df[
        (df["category"] == cat) &
        (df["zone"] == z) &
        (df["price"] >= p_min) &
        (df["price"] <= p_max)
    ]["id"].to_numpy()
    assert np.array_equal(matched_ids, np.sort(expected))


def test_empty_predicates_returns_all(sample_index):
    idx, df = sample_index
    all_ids = idx.get_eligible_ids([])
    assert len(all_ids) == len(df)


def test_unknown_values_returns_empty(sample_index):
    idx, _ = sample_index
    res = idx.get_eligible_ids([Predicate.category_eq("NON_EXISTENT_CATEGORY")])
    assert len(res) == 0

    res2 = idx.get_eligible_ids([Predicate.zone_eq("NON_EXISTENT_ZONE")])
    assert len(res2) == 0


def test_selectivity_estimation_accuracy(sample_index):
    idx, df = sample_index
    cat = "traditional_sweets"
    preds = [Predicate.category_eq(cat)]
    est_sel, est_cnt = idx.estimate_selectivity(preds)

    actual_cnt = len(df[df["category"] == cat])
    actual_sel = actual_cnt / len(df)

    assert pytest.approx(est_sel, abs=0.01) == actual_sel
    assert abs(est_cnt - actual_cnt) <= 2


def test_matches_batch_consistency(sample_index):
    idx, df = sample_index
    preds = [Predicate.price_range(300.0, 700.0)]
    candidate_ids = np.array([5, 12, 50, 99, 120, 300], dtype=np.int32)

    batch_mask = idx.matches_batch(candidate_ids, preds)
    for c_id, match in zip(candidate_ids, batch_mask):
        price = df.loc[df["id"] == c_id, "price"].iloc[0]
        expected_match = 300.0 <= price <= 700.0
        assert match == expected_match
