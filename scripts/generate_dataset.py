"""Script to generate, validate, and save the synthetic Madurai dataset."""

import argparse
from pathlib import Path
import sys

# Ensure repository root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import (
    generate_synthetic_dataset,
    save_dataset,
    validate_dataset,
)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Generate synthetic Madurai dataset for vector search.")
    parser.add_argument("--records", type=int, default=50000, help="Number of records to generate (default: 50,000)")
    parser.add_argument("--dim", type=int, default=32, help="Embedding dimensionality (default: 32)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--outdir", type=str, default="data", help="Output directory path (default: data)")
    args = parser.parse_args()

    print("=" * 70)
    print("MADURAI PINCODE SEARCH: SYNTHETIC DATASET GENERATION")
    print("=" * 70)
    print(f"Generating {args.records:,} records with {args.dim}-dimensional embeddings...")
    print(f"Random seed: {args.seed}")

    bundle = generate_synthetic_dataset(
        num_records=args.records,
        embedding_dim=args.dim,
        seed=args.seed,
    )

    print("\nValidating dataset integrity and statistical properties...")
    validations = validate_dataset(bundle)
    for check_name, passed in validations.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {check_name}")

    if not validations["ALL_PASSED"]:
        print("\nERROR: Dataset validation failed!")
        sys.exit(1)

    print("\nSaving dataset artifacts to disk...")
    out_dir = Path(__file__).resolve().parent.parent / args.outdir
    csv_p, npz_p, json_p = save_dataset(bundle, out_dir)

    stats = bundle.stats
    print(f"\nSuccessfully generated and saved dataset:")
    print(f"  Metadata CSV:       {csv_p.name} ({csv_p.stat().st_size / 1e6:.2f} MB)")
    print(f"  Embeddings NPZ:     {npz_p.name} ({npz_p.stat().st_size / 1e6:.2f} MB)")
    print(f"  Stats JSON:         {json_p.name}")
    print(f"  Total records:      {stats['num_records']:,}")
    print(f"  Categories count:   {stats['categories_count']}")
    print(f"  Zones count:        {stats['zones_count']}")
    print(f"  Busiest zone:       {stats['busiest_zone']} ({stats['busiest_zone_count']:,} records)")
    print(f"  Least busy zone:    {stats['least_busy_zone']} ({stats['least_busy_zone_count']:,} records)")
    print(f"  Price range (INR):  INR {stats['price_stats']['min']:,.2f} - INR {stats['price_stats']['max']:,.2f} (median: INR {stats['price_stats']['median']:,.2f})")
    print("\nDISCLAIMER:")
    print("  This dataset is completely synthetic for algorithmic research and benchmarking.")
    print("  It does NOT represent real personal data or actual Indian postal directories.")
    print("=" * 70)


if __name__ == "__main__":
    main()
