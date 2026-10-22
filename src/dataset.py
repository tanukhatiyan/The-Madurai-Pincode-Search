"""Synthetic dataset generation and validation for the Madurai Pincode Search project.

Disclaimer:
    The dataset generated here is completely synthetic for research and benchmark
    purposes in vector search with metadata filtering. It does not represent real-world
    personal addresses or actual Indian postal database records.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
import pandas as pd


CATEGORIES: List[str] = [
    "jasmine_flowers",        # Madurai Malli & floral market
    "handloom_sarees",       # Sungudi and traditional sarees
    "temple_handicrafts",     # Brass lamps, puja items
    "bronze_idols",          # Traditional metal casting
    "traditional_sweets",     # Halwa, jigarthanda ingredients
    "spices_condiments",     # Local spice trade
    "cotton_textiles",       # Spinning & weaving
    "pharmaceuticals",        # Medical distribution
    "electronics_hardware",   # Retail & wholesale tech
    "automotive_parts",       # Spares & ancillary manufacturing
    "agricultural_produce",  # Wholesale vegetables, grains
    "dairy_products",        # Milk, ghee, sweets
    "heavy_machinery",       # Agro & industrial equipment
    "ceramic_tiles",         # Building finishes
    "leather_footwear",      # Tannery & footwear trade
    "printing_publishing",   # Book printing & packaging
    "solar_equipment",       # Green energy components
    "construction_materials",# Cement, steel, brick trade
    "gold_jewelry",          # Retail jewelry & bullion
    "packaged_foods",        # Processed agro goods
]

ZONES: List[Dict[str, str]] = [
    {"zone": "Zone_Meenakshi_Amman", "pincode": "625001", "type": "heritage_hub"},
    {"zone": "Zone_Simmakkal", "pincode": "625001", "type": "wholesale_market"},
    {"zone": "Zone_South_Gate", "pincode": "625001", "type": "retail_hub"},
    {"zone": "Zone_Madurai_East", "pincode": "625009", "type": "textile_weaving"},
    {"zone": "Zone_Teppakulam", "pincode": "625009", "type": "commercial_residential"},
    {"zone": "Zone_Sellur", "pincode": "625002", "type": "handloom_cluster"},
    {"zone": "Zone_Goripalayam", "pincode": "625002", "type": "medical_educational"},
    {"zone": "Zone_Anna_Nagar", "pincode": "625020", "type": "prime_commercial"},
    {"zone": "Zone_KK_Nagar", "pincode": "625020", "type": "retail_services"},
    {"zone": "Zone_Mattuthavani", "pincode": "625007", "type": "integrated_market_transit"},
    {"zone": "Zone_Kappalur", "pincode": "625008", "type": "industrial_estate"},
    {"zone": "Zone_Kochadai", "pincode": "625016", "type": "light_manufacturing"},
    {"zone": "Zone_Thiruparankundram", "pincode": "625005", "type": "temple_town_crafts"},
    {"zone": "Zone_Pasumalai", "pincode": "625004", "type": "engineering_education"},
    {"zone": "Zone_Palanganatham", "pincode": "625003", "type": "transit_retail"},
    {"zone": "Zone_Villapuram", "pincode": "625012", "type": "residential_trade"},
    {"zone": "Zone_Avaniyapuram", "pincode": "625012", "type": "agro_suburban"},
    {"zone": "Zone_Arapalayam", "pincode": "625016", "type": "bus_terminal_logistics"},
    {"zone": "Zone_TVS_Nagar", "pincode": "625003", "type": "auto_ancillary_residential"},
    {"zone": "Zone_Alagappan_Nagar", "pincode": "625003", "type": "mixed_residential"},
    {"zone": "Zone_Ellis_Nagar", "pincode": "625016", "type": "warehousing_trade"},
    {"zone": "Zone_Vandiyur", "pincode": "625020", "type": "suburban_residential"},
    {"zone": "Zone_Othakadai", "pincode": "625107", "type": "agro_institutional"},
    {"zone": "Zone_Karuppayurani", "pincode": "625020", "type": "expansion_zone"},
    {"zone": "Zone_Sakkimangalam", "pincode": "625201", "type": "pottery_crafts"},
    {"zone": "Zone_Nagamalai_Pudukottai", "pincode": "625019", "type": "academic_suburb"},
    {"zone": "Zone_Samayanallur", "pincode": "625402", "type": "agro_processing"},
    {"zone": "Zone_Thirunagar", "pincode": "625006", "type": "craft_handloom"},
    {"zone": "Zone_Melur_Road", "pincode": "625106", "type": "granite_logistics"},
    {"zone": "Zone_Vadipatti", "pincode": "625218", "type": "peri_urban_agriculture"},
]

# Baseline price distribution parameters per category (log-normal: meanlog, sdlog, min_val, max_val) in INR (₹)
CATEGORY_PRICE_PARAMS: Dict[str, Tuple[float, float, float, float]] = {
    "jasmine_flowers": (5.3, 0.6, 50.0, 3000.0),         # median ~200
    "handloom_sarees": (7.8, 0.7, 600.0, 25000.0),       # median ~2500
    "temple_handicrafts": (7.2, 0.8, 250.0, 18000.0),    # median ~1350
    "bronze_idols": (8.8, 0.9, 1500.0, 150000.0),        # median ~6600
    "traditional_sweets": (5.8, 0.5, 100.0, 2500.0),     # median ~330
    "spices_condiments": (6.1, 0.6, 120.0, 4500.0),      # median ~450
    "cotton_textiles": (7.5, 0.8, 400.0, 20000.0),       # median ~1800
    "pharmaceuticals": (6.5, 0.9, 80.0, 15000.0),        # median ~660
    "electronics_hardware": (8.5, 1.1, 500.0, 85000.0),  # median ~4900
    "automotive_parts": (8.2, 1.0, 300.0, 65000.0),      # median ~3600
    "agricultural_produce": (5.5, 0.7, 40.0, 5000.0),    # median ~250
    "dairy_products": (5.0, 0.5, 45.0, 1800.0),          # median ~150
    "heavy_machinery": (11.0, 1.0, 15000.0, 600000.0),   # median ~60000
    "ceramic_tiles": (7.6, 0.6, 600.0, 18000.0),        # median ~2000
    "leather_footwear": (7.1, 0.6, 350.0, 9500.0),       # median ~1200
    "printing_publishing": (6.8, 0.7, 150.0, 12000.0),   # median ~900
    "solar_equipment": (9.8, 0.9, 5000.0, 250000.0),     # median ~18000
    "construction_materials": (8.6, 0.8, 1000.0, 75000.0),# median ~5400
    "gold_jewelry": (10.5, 0.9, 8000.0, 450000.0),      # median ~36000
    "packaged_foods": (5.7, 0.6, 80.0, 3500.0),          # median ~300
}


@dataclass
class DatasetBundle:
    """Holds metadata DataFrame, embeddings matrix, and summary statistics."""
    metadata: pd.DataFrame
    embeddings: np.ndarray
    stats: Dict[str, any]

    @property
    def num_records(self) -> int:
        return len(self.metadata)

    @property
    def embedding_dim(self) -> int:
        return self.embeddings.shape[1]


def compute_zone_probabilities() -> np.ndarray:
    """Generate realistic uneven zone probabilities following a power-law distribution.

    Busiest zones (Meenakshi Amman, Mattuthavani, Simmakkal, Kappalur) receive significantly
    higher probability weights than peripheral suburban zones.
    """
    n_zones = len(ZONES)
    # Ranks from 1 to 30, with Zipf-like exponent
    ranks = np.arange(1, n_zones + 1, dtype=np.float64)
    # Add mild domain clustering: top 4 hubs get extra boost
    weights = 1.0 / (ranks ** 0.82)
    weights[:4] *= 1.45
    probs = weights / np.sum(weights)
    return probs


def compute_category_zone_affinity_matrix() -> np.ndarray:
    """Compute affinity weight matrix between categories (rows) and zones (cols).

    Returns:
        Matrix of shape (n_categories, n_zones) representing conditional P(category | zone).
    """
    n_cats = len(CATEGORIES)
    n_zones = len(ZONES)
    affinity = np.ones((n_cats, n_zones), dtype=np.float64) * 0.4

    cat_idx = {cat: i for i, cat in enumerate(CATEGORIES)}
    zone_idx = {z["zone"]: i for i, z in enumerate(ZONES)}

    # Define strong local cluster associations
    correlations = [
        ("jasmine_flowers", ["Zone_Meenakshi_Amman", "Zone_Simmakkal", "Zone_Mattuthavani", "Zone_Thiruparankundram"], 6.0),
        ("handloom_sarees", ["Zone_Sellur", "Zone_Madurai_East", "Zone_South_Gate", "Zone_Thirunagar"], 6.5),
        ("temple_handicrafts", ["Zone_Meenakshi_Amman", "Zone_Thiruparankundram", "Zone_South_Gate"], 7.0),
        ("bronze_idols", ["Zone_Meenakshi_Amman", "Zone_Thiruparankundram", "Zone_Sakkimangalam"], 6.0),
        ("traditional_sweets", ["Zone_Simmakkal", "Zone_South_Gate", "Zone_Meenakshi_Amman", "Zone_Mattuthavani"], 5.5),
        ("spices_condiments", ["Zone_Simmakkal", "Zone_Mattuthavani", "Zone_Madurai_East"], 5.0),
        ("cotton_textiles", ["Zone_Sellur", "Zone_Madurai_East", "Zone_Ellis_Nagar"], 5.5),
        ("pharmaceuticals", ["Zone_Goripalayam", "Zone_Anna_Nagar", "Zone_KK_Nagar"], 5.5),
        ("electronics_hardware", ["Zone_Anna_Nagar", "Zone_KK_Nagar", "Zone_South_Gate"], 5.0),
        ("automotive_parts", ["Zone_Kappalur", "Zone_Kochadai", "Zone_TVS_Nagar", "Zone_Arapalayam"], 7.5),
        ("agricultural_produce", ["Zone_Mattuthavani", "Zone_Samayanallur", "Zone_Vadipatti", "Zone_Othakadai"], 7.0),
        ("heavy_machinery", ["Zone_Kappalur", "Zone_Kochadai", "Zone_Melur_Road"], 8.0),
        ("ceramic_tiles", ["Zone_Melur_Road", "Zone_Villapuram", "Zone_Kappalur"], 5.0),
        ("gold_jewelry", ["Zone_South_Gate", "Zone_Anna_Nagar", "Zone_Meenakshi_Amman"], 6.0),
        ("printing_publishing", ["Zone_Madurai_East", "Zone_Goripalayam", "Zone_Kochadai"], 5.0),
        ("solar_equipment", ["Zone_Kappalur", "Zone_Pasumalai", "Zone_Anna_Nagar"], 5.0),
        ("construction_materials", ["Zone_Melur_Road", "Zone_Kappalur", "Zone_Villapuram"], 5.5),
        ("leather_footwear", ["Zone_South_Gate", "Zone_Villapuram", "Zone_Kappalur"], 5.0),
        ("packaged_foods", ["Zone_Mattuthavani", "Zone_Simmakkal", "Zone_Samayanallur"], 4.5),
        ("dairy_products", ["Zone_Samayanallur", "Zone_Vadipatti", "Zone_Mattuthavani"], 5.5),
    ]

    for cat_name, target_zones, boost in correlations:
        c_i = cat_idx[cat_name]
        for z_name in target_zones:
            if z_name in zone_idx:
                affinity[c_i, zone_idx[z_name]] += boost

    # Normalize column-wise so that for each zone, sum of category probabilities == 1.0
    col_sums = affinity.sum(axis=0, keepdims=True)
    cond_prob = affinity / col_sums
    return cond_prob


def generate_synthetic_dataset(
    num_records: int = 50000,
    embedding_dim: int = 32,
    seed: int = 42,
    noise_level: float = 0.35,
) -> DatasetBundle:
    """Generate a reproducible synthetic dataset with realistic correlations and cluster structure.

    Args:
        num_records: Total records to generate (30,000 to 100,000).
        embedding_dim: Number of vector embedding dimensions (default: 32).
        seed: Random seed for deterministic reproducibility.
        noise_level: Standard deviation of Gaussian noise added to embeddings.

    Returns:
        DatasetBundle containing metadata DataFrame, embedding matrix, and statistics.
    """
    if num_records < 100:
        raise ValueError(f"num_records must be at least 100, got {num_records}")
    if embedding_dim < 4:
        raise ValueError(f"embedding_dim must be at least 4, got {embedding_dim}")

    rng = np.random.default_rng(seed)

    # 1. Zone distribution (uneven)
    zone_names = [z["zone"] for z in ZONES]
    zone_pincodes = {z["zone"]: z["pincode"] for z in ZONES}
    zone_types = {z["zone"]: z["type"] for z in ZONES}
    zone_probs = compute_zone_probabilities()

    zone_indices = rng.choice(len(ZONES), size=num_records, p=zone_probs)
    assigned_zones = [zone_names[idx] for idx in zone_indices]

    # 2. Category-Zone correlation
    cond_category_probs = compute_category_zone_affinity_matrix()  # shape (n_cats, n_zones)
    assigned_categories: List[str] = []

    # Sample category for each record conditioned on its assigned zone
    for z_idx in zone_indices:
        cat_p = cond_category_probs[:, z_idx]
        chosen_cat_idx = rng.choice(len(CATEGORIES), p=cat_p)
        assigned_categories.append(CATEGORIES[chosen_cat_idx])

    # 3. Price generation in INR (₹)
    prices = np.zeros(num_records, dtype=np.float32)
    for i, cat in enumerate(assigned_categories):
        meanlog, sdlog, min_val, max_val = CATEGORY_PRICE_PARAMS[cat]
        # Log-normal distribution
        raw_price = rng.lognormal(meanlog, sdlog)
        clipped_price = np.clip(raw_price, min_val, max_val)
        # Round to 2 decimals for currency precision
        prices[i] = round(float(clipped_price), 2)

    # 4. Correlated vector embeddings with cluster structure
    # Create base semantic cluster centers for categories
    cat_centers = rng.normal(0.0, 1.0, size=(len(CATEGORIES), embedding_dim)).astype(np.float32)
    cat_centers /= np.linalg.norm(cat_centers, axis=1, keepdims=True)

    # Create spatial bias centers for zones
    zone_biases = rng.normal(0.0, 0.8, size=(len(ZONES), embedding_dim)).astype(np.float32)
    zone_biases /= np.linalg.norm(zone_biases, axis=1, keepdims=True)

    cat_map = {cat: i for i, cat in enumerate(CATEGORIES)}
    embeddings = np.zeros((num_records, embedding_dim), dtype=np.float32)

    for i in range(num_records):
        c_i = cat_map[assigned_categories[i]]
        z_i = zone_indices[i]
        # Base vector = 0.65 * category_center + 0.35 * zone_bias + noise
        base = 0.65 * cat_centers[c_i] + 0.35 * zone_biases[z_i]
        noise = rng.normal(0.0, noise_level, size=(embedding_dim,)).astype(np.float32)
        vec = base + noise
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec /= norm
        embeddings[i] = vec

    record_ids = np.arange(num_records, dtype=np.int32)
    pincodes = [zone_pincodes[z] for z in assigned_zones]
    types = [zone_types[z] for z in assigned_zones]

    metadata_df = pd.DataFrame({
        "id": record_ids,
        "category": assigned_categories,
        "price": prices,
        "zone": assigned_zones,
        "pincode": pincodes,
        "zone_type": types,
    })

    # Compute summary statistics
    zone_counts = metadata_df["zone"].value_counts().to_dict()
    category_counts = metadata_df["category"].value_counts().to_dict()
    price_stats = {
        "min": float(prices.min()),
        "max": float(prices.max()),
        "mean": float(prices.mean()),
        "median": float(np.median(prices)),
        "p25": float(np.percentile(prices, 25)),
        "p75": float(np.percentile(prices, 75)),
        "p95": float(np.percentile(prices, 95)),
    }

    stats = {
        "num_records": num_records,
        "embedding_dim": embedding_dim,
        "seed": seed,
        "categories_count": len(category_counts),
        "zones_count": len(zone_counts),
        "busiest_zone": metadata_df["zone"].value_counts().index[0],
        "busiest_zone_count": int(metadata_df["zone"].value_counts().iloc[0]),
        "least_busy_zone": metadata_df["zone"].value_counts().index[-1],
        "least_busy_zone_count": int(metadata_df["zone"].value_counts().iloc[-1]),
        "zone_counts": zone_counts,
        "category_counts": category_counts,
        "price_stats": price_stats,
    }

    return DatasetBundle(metadata=metadata_df, embeddings=embeddings, stats=stats)


def save_dataset(
    bundle: DatasetBundle,
    output_dir: str | Path,
    metadata_filename: str = "madurai_records.csv",
    embeddings_filename: str = "madurai_embeddings.npz",
    stats_filename: str = "metadata_stats.json",
) -> Tuple[Path, Path, Path]:
    """Save dataset files to disk."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_path = out_path / metadata_filename
    npz_path = out_path / embeddings_filename
    json_path = out_path / stats_filename

    bundle.metadata.to_csv(csv_path, index=False)
    np.savez_compressed(npz_path, embeddings=bundle.embeddings, record_ids=bundle.metadata["id"].to_numpy())

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(bundle.stats, f, indent=2)

    return csv_path, npz_path, json_path


def load_dataset(
    data_dir: str | Path,
    metadata_filename: str = "madurai_records.csv",
    embeddings_filename: str = "madurai_embeddings.npz",
    stats_filename: str = "metadata_stats.json",
) -> DatasetBundle:
    """Load dataset files from disk."""
    dir_path = Path(data_dir)
    csv_path = dir_path / metadata_filename
    npz_path = dir_path / embeddings_filename
    json_path = dir_path / stats_filename

    if not csv_path.exists() or not npz_path.exists():
        raise FileNotFoundError(f"Dataset files not found in {data_dir}")

    metadata = pd.read_csv(csv_path)
    npz_data = np.load(npz_path)
    embeddings = npz_data["embeddings"]

    stats = {}
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

    return DatasetBundle(metadata=metadata, embeddings=embeddings, stats=stats)


def validate_dataset(bundle: DatasetBundle) -> Dict[str, bool]:
    """Validate all required properties of the generated dataset.

    Checks:
        1. Record count and unique IDs
        2. Embedding dimensionality and finite values
        3. Valid categories and zones
        4. Nonnegative prices
        5. Uneven zone distribution (max/min ratio >= 3.0)
        6. Category-zone correlation (contingency chi-squared test p-value < 1e-5)
    """
    df = bundle.metadata
    emb = bundle.embeddings

    validations = {}

    # 1. Unique IDs
    validations["count_matches"] = bool(len(df) == emb.shape[0])
    validations["unique_ids"] = bool(df["id"].nunique() == len(df))
    validations["contiguous_ids"] = bool((df["id"].min() == 0) and (df["id"].max() == len(df) - 1))

    # 2. Embeddings
    validations["finite_embeddings"] = bool(np.all(np.isfinite(emb)))
    validations["non_zero_norms"] = bool(np.all(np.linalg.norm(emb, axis=1) > 0.1))

    # 3. Categories and zones
    validations["valid_categories"] = bool(set(df["category"].unique()).issubset(set(CATEGORIES)))
    validations["valid_zones"] = bool(set(df["zone"].unique()).issubset(set(z["zone"] for z in ZONES)))
    validations["category_count_in_range"] = bool(18 <= df["category"].nunique() <= 22)
    validations["zone_count_in_range"] = bool(28 <= df["zone"].nunique() <= 32)

    # 4. Prices
    validations["non_negative_prices"] = bool((df["price"] >= 0).all())

    # 5. Uneven zone distribution (check max zone vs min zone ratio)
    zone_counts = df["zone"].value_counts()
    ratio = float(zone_counts.iloc[0]) / float(zone_counts.iloc[-1])
    validations["uneven_zone_distribution"] = bool(ratio >= 3.0)

    # 6. Category-zone correlation (Contingency table chi-square test)
    contingency = pd.crosstab(df["category"], df["zone"]).to_numpy()
    row_sums = contingency.sum(axis=1, keepdims=True)
    col_sums = contingency.sum(axis=0, keepdims=True)
    total = contingency.sum()
    expected = (row_sums @ col_sums) / total
    # Chi-square statistic
    chi2 = float(np.sum((contingency - expected) ** 2 / np.maximum(expected, 1e-9)))
    # For (20-1)*(30-1) = 551 degrees of freedom, expected chi2 under independence is ~551.
    # Significant correlation produces chi2 >> 2000
    validations["category_zone_correlation"] = bool(chi2 > 2000.0)

    all_passed = bool(all(validations.values()))
    validations["ALL_PASSED"] = all_passed
    return validations
