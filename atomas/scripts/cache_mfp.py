#!/usr/bin/env python
"""Pre-compute molecular fingerprints for SMILES and cache to disk.

Supports Morgan, MACCS Keys, and RDKit topological fingerprints.

Usage:
    python scripts/cache_mfp.py --dataset pubchemstm --data-dir data/ --output data/pubchemstm/smiles2mfp.npy
    python scripts/cache_mfp.py --dataset pubchemstm --fp-type maccs --output data/pubchemstm/smiles2maccs.npy
    python scripts/cache_mfp.py --dataset ChEBI-20_data --fp-type rdkit --output data/chebi20/smiles2rdkit.npy
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from utils.fingerprint import (
    FP_DIMENSIONS,
    disable_rdkit_logging,
    smiles_to_fingerprint,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable RDKit warnings to avoid noise
disable_rdkit_logging()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-compute molecular fingerprints and cache them."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["pubchemstm", "ChEBI-20_data"],
        help="Dataset to process.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root directory for raw data files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the cached fingerprint file (e.g., data/pubchemstm/smiles2mfp.npy).",
    )
    parser.add_argument(
        "--fp-type",
        type=str,
        choices=["morgan", "maccs", "rdkit"],
        default="morgan",
        help="Fingerprint type to cache (default: morgan).",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius (default: 2). Only used for morgan type.",
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        default=2048,
        help="Fingerprint bit length (default: 2048). Only used for morgan/rdkit types.",
    )
    parser.add_argument(
        "--min-path",
        type=int,
        default=1,
        help="Minimum path length for RDKit fingerprint (default: 1).",
    )
    parser.add_argument(
        "--max-path",
        type=int,
        default=7,
        help="Maximum path length for RDKit fingerprint (default: 7).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cache file if present.",
    )
    return parser.parse_args()


def get_unique_smiles_pubchemstm(data_dir: str) -> list[str]:
    """Extract unique SMILES from PubChemSTM dataset."""
    import pandas as pd

    cid2smiles_file = Path(data_dir) / "raw" / "CID2SMILES.csv"
    if not cid2smiles_file.exists():
        raise FileNotFoundError(f"CID2SMILES.csv not found at {cid2smiles_file}")

    df = pd.read_csv(cid2smiles_file)
    smiles_list = df["SMILES"].unique().tolist()
    logger.info(f"Found {len(smiles_list)} unique SMILES in PubChemSTM")
    return smiles_list


def get_unique_smiles_chebi20() -> list[str]:
    """Extract unique SMILES from ChEBI-20 dataset."""
    from dataset.chEBI20MMDataset import ChEBI20MMDataset

    all_smiles = set()
    for split in ["train", "validation", "test"]:
        try:
            dataset = ChEBI20MMDataset(split=split)
            for i in range(len(dataset)):
                sample = dataset[i]
                all_smiles.add(sample["smiles"])
        except Exception as e:
            logger.warning(f"Could not load {split} split: {e}")

    smiles_list = list(all_smiles)
    logger.info(f"Found {len(smiles_list)} unique SMILES in ChEBI-20")
    return smiles_list


def build_smiles2fp(
    smiles_list: list[str],
    fp_type: str = "morgan",
    radius: int = 2,
    n_bits: int = 2048,
    min_path: int = 1,
    max_path: int = 7,
) -> dict[str, np.ndarray]:
    """Build SMILES to fingerprint mapping.

    Args:
        smiles_list: List of unique SMILES strings.
        fp_type: Fingerprint type ("morgan", "maccs", or "rdkit").
        radius: Morgan fingerprint radius.
        n_bits: Fingerprint bit length (morgan/rdkit only).
        min_path: Minimum path length for RDKit fingerprint.
        max_path: Maximum path length for RDKit fingerprint.

    Returns:
        Dictionary mapping SMILES to fingerprint arrays.
    """
    smiles2fp = {}
    total = len(smiles_list)

    # Build kwargs based on fingerprint type
    kwargs = {}
    if fp_type == "morgan":
        kwargs = {"radius": radius, "n_bits": n_bits}
    elif fp_type == "rdkit":
        kwargs = {"min_path": min_path, "max_path": max_path, "n_bits": n_bits}
    # maccs has no configurable parameters

    for i, smiles in enumerate(smiles_list):
        fp = smiles_to_fingerprint(smiles, fp_type=fp_type, **kwargs)
        smiles2fp[smiles] = fp.squeeze(0)  # (fp_dim,)

        if (i + 1) % 10000 == 0:
            logger.info(f"Processed {i + 1}/{total} SMILES")

    return smiles2fp


def main() -> None:
    args = parse_args()

    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        logger.info(
            f"Cache file already exists at {output_path}. Use --overwrite to replace."
        )
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get unique SMILES based on dataset
    if args.dataset == "pubchemstm":
        smiles_list = get_unique_smiles_pubchemstm(args.data_dir)
    else:
        smiles_list = get_unique_smiles_chebi20()

    # Build SMILES -> fingerprint mapping
    logger.info(f"Building {args.fp_type} fingerprints...")
    smiles2fp = build_smiles2fp(
        smiles_list,
        fp_type=args.fp_type,
        radius=args.radius,
        n_bits=args.n_bits,
        min_path=args.min_path,
        max_path=args.max_path,
    )

    # Save to disk
    np.save(output_path, smiles2fp, allow_pickle=True)
    logger.info(
        f"Saved {len(smiles2fp)} SMILES->{args.fp_type} mappings to {output_path}"
    )

    # Save metadata
    fp_dim = FP_DIMENSIONS.get(args.fp_type, 2048)
    metadata = {
        "dataset": args.dataset,
        "fp_type": args.fp_type,
        "fp_dim": fp_dim,
        "num_smiles": len(smiles2fp),
    }
    # Add type-specific parameters
    if args.fp_type == "morgan":
        metadata["radius"] = args.radius
        metadata["n_bits"] = args.n_bits
    elif args.fp_type == "rdkit":
        metadata["min_path"] = args.min_path
        metadata["max_path"] = args.max_path
        metadata["n_bits"] = args.n_bits
    # maccs has no configurable parameters

    metadata_path = output_path.with_suffix(".json")
    import json

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
