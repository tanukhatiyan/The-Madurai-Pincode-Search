"""Unit tests for dataset generation, schema validation, and persistence."""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from src.dataset import (
    CATEGORIES,
    ZONES,
    generate_synthetic_dataset,
    save_dataset,
    load_dataset,
    validate_dataset,
)


def test_generate_synthetic_dataset_properties():
    bundle = generate_synthetic_dataset(num_records=1000, embedding_dim=16, seed=42)
    assert bundle.num_records == 1000
    assert bundle.embedding_dim == 16
    assert len(bundle.metadata) == 1000

    vals = validate_dataset(bundle)
    assert vals["ALL_PASSED"] is True
    assert vals["unique_ids"] is True
    assert vals["contiguous_ids"] is True
    assert vals["finite_embeddings"] is True
    assert vals["non_negative_prices"] is True
    assert vals["uneven_zone_distribution"] is True
    assert vals["category_zone_correlation"] is True


def test_dataset_reproducibility():
    b1 = generate_synthetic_dataset(num_records=300, embedding_dim=8, seed=123)
    b2 = generate_synthetic_dataset(num_records=300, embedding_dim=8, seed=123)

    assert np.allclose(b1.embeddings, b2.embeddings)
    assert (b1.metadata["category"] == b2.metadata["category"]).all()
    assert (b1.metadata["zone"] == b2.metadata["zone"]).all()
    assert (b1.metadata["price"] == b2.metadata["price"]).all()


def test_dataset_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle = generate_synthetic_dataset(num_records=200, embedding_dim=8, seed=77)
        csv_p, npz_p, json_p = save_dataset(bundle, tmpdir)

        assert csv_p.exists()
        assert npz_p.exists()
        assert json_p.exists()

        loaded = load_dataset(tmpdir)
        assert loaded.num_records == 200
        assert loaded.embedding_dim == 8
        assert np.allclose(loaded.embeddings, bundle.embeddings)
        assert (loaded.metadata["id"] == bundle.metadata["id"]).all()
