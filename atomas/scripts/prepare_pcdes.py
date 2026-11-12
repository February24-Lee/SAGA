#!/usr/bin/env python
"""Prepare PCdes dataset with scaffold splitting.

Downloads data from KV-PLM GitHub and creates train/valid/test splits
using Murcko scaffold splitting.
"""

import argparse
import os
import random
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def get_scaffold(smiles: str) -> str:
    """Extract Murcko scaffold from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:
        return ""


def scaffold_split(
    smiles_list: list[str],
    train_ratio: float = 0.7,
    valid_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Split data by scaffold.

    Args:
        smiles_list: List of SMILES strings
        train_ratio: Fraction for training set
        valid_ratio: Fraction for validation set
        seed: Random seed

    Returns:
        Tuple of (train_indices, valid_indices, test_indices)
    """
    random.seed(seed)

    # Group molecules by scaffold
    scaffold_to_indices = defaultdict(list)
    for idx, smiles in enumerate(smiles_list):
        scaffold = get_scaffold(smiles)
        scaffold_to_indices[scaffold].append(idx)

    # Sort scaffolds by size (largest first for more stable splits)
    scaffold_groups = list(scaffold_to_indices.values())
    scaffold_groups.sort(key=lambda x: len(x), reverse=True)

    # Shuffle scaffolds with same size
    random.shuffle(scaffold_groups)

    n_total = len(smiles_list)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)

    train_indices = []
    valid_indices = []
    test_indices = []

    for group in scaffold_groups:
        if len(train_indices) < n_train:
            train_indices.extend(group)
        elif len(valid_indices) < n_valid:
            valid_indices.extend(group)
        else:
            test_indices.extend(group)

    return train_indices, valid_indices, test_indices


def main():
    parser = argparse.ArgumentParser(description="Prepare PCdes dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/PCdes",
        help="Directory containing raw PCdes files",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Training set ratio",
    )
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.1,
        help="Validation set ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    # Read raw data
    smiles_path = os.path.join(args.data_dir, "align_smiles.txt")
    desc_path = os.path.join(args.data_dir, "align_des_filt3.txt")

    with open(smiles_path) as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    with open(desc_path) as f:
        desc_list = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(smiles_list)} SMILES and {len(desc_list)} descriptions")

    # Ensure same length
    n = min(len(smiles_list), len(desc_list))
    smiles_list = smiles_list[:n]
    desc_list = desc_list[:n]
    print(f"Using {n} paired samples")

    # Scaffold split
    train_idx, valid_idx, test_idx = scaffold_split(
        smiles_list,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )

    print(f"Train: {len(train_idx)}, Valid: {len(valid_idx)}, Test: {len(test_idx)}")

    # Write split files
    def write_split(filename: str, indices: list[int]):
        filepath = os.path.join(args.data_dir, filename)
        with open(filepath, "w") as f:
            f.write("smiles\tdesc\n")  # Header
            for idx in indices:
                smiles = smiles_list[idx]
                desc = desc_list[idx]
                # Escape tabs in description
                desc = desc.replace("\t", " ")
                f.write(f"{smiles}\t{desc}\n")
        print(f"Wrote {filepath}")

    write_split("train.txt", train_idx)
    write_split("valid.txt", valid_idx)
    write_split("test.txt", test_idx)

    print("Done!")


if __name__ == "__main__":
    main()
