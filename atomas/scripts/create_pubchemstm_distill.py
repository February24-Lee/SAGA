#!/usr/bin/env python
"""
Create the PubChemSTM-distill corpus, following the data pipeline of Atomas (Appendix E).

Process (counts of the corpus shipped in data/PubchemSTM_distill/):
1. Load the PubChemSTM molecule-text pairs (408,250)
2. Keep one description per molecule, merging by CID and canonical SMILES (219,645)
3. Drop descriptions shorter than 18 words (45,496)
4. Remove molecules that occur in ChEBI-20 or PCdes (34,365)
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Create the PubChemSTM-distill corpus")
    parser.add_argument(
        "--pubchem-dir",
        type=str,
        default="data/PubChemSTM_data/raw",
        help="Directory containing CID2text.json and CID2SMILES.csv",
    )
    parser.add_argument(
        "--chebi20-dir",
        type=str,
        default="data/cache/ChEBI-20_data/base",
        help="Directory containing the cached ChEBI-20 splits (train/validation/test)",
    )
    parser.add_argument(
        "--pcdes-dir",
        type=str,
        default="data/PCdes",
        help="Directory containing PCdes dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/PubchemSTM_distill",
        help="Output directory",
    )
    parser.add_argument(
        "--min-word-count",
        type=int,
        default=18,
        help="Minimum word count in text",
    )
    return parser.parse_args()


def canonicalize_smiles(smiles: str) -> str | None:
    """Convert SMILES to canonical form using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def load_chebi20_smiles(chebi20_dir: Path) -> set:
    """Load all SMILES from ChEBI-20 dataset."""
    smiles_set = set()

    for split in ["train", "validation", "test"]:
        split_dir = chebi20_dir / split
        if not split_dir.exists():
            print(f"  Warning: {split_dir} not found")
            continue

        # Try to load from Arrow dataset
        try:
            from datasets import load_from_disk

            ds = load_from_disk(str(split_dir))
            for sample in ds:
                smiles = sample.get("smiles", "")
                if smiles:
                    canon = canonicalize_smiles(smiles)
                    if canon:
                        smiles_set.add(canon)
        except Exception as e:
            print(f"  Warning: Could not load {split_dir}: {e}")

    return smiles_set


def load_pcdes_smiles(pcdes_dir: Path) -> set:
    """Load all SMILES from PCdes dataset."""
    smiles_set = set()

    for split_file in ["train.txt", "valid.txt", "test.txt"]:
        filepath = pcdes_dir / split_file
        if not filepath.exists():
            print(f"  Warning: {filepath} not found")
            continue

        df = pd.read_csv(filepath, sep="\t")
        for smiles in df["smiles"]:
            if smiles:
                canon = canonicalize_smiles(smiles)
                if canon:
                    smiles_set.add(canon)

    return smiles_set


def main():
    args = parse_args()

    pubchem_dir = Path(args.pubchem_dir)
    chebi20_dir = Path(args.chebi20_dir)
    pcdes_dir = Path(args.pcdes_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load PubChem data
    print("Step 1: Loading PubChem data...")

    cid2text_file = pubchem_dir / "CID2text.json"
    cid2smiles_file = pubchem_dir / "CID2SMILES.csv"

    with open(cid2text_file) as f:
        cid2text = json.load(f)
    print(f"  Loaded {len(cid2text)} CIDs with text")

    df_smiles = pd.read_csv(cid2smiles_file)
    cid2smiles = {
        str(cid): smiles for cid, smiles in zip(df_smiles["CID"], df_smiles["SMILES"])
    }
    print(f"  Loaded {len(cid2smiles)} CIDs with SMILES")

    # Count total pairs
    total_pairs = sum(
        len(texts) if isinstance(texts, list) else 1 for texts in cid2text.values()
    )
    print(f"  Total molecule-text pairs: {total_pairs}")

    # Step 2: Create unique SMILES-text pairs (merge same CID)
    print("\nStep 2: Creating unique SMILES-text pairs (merging same CID)...")

    unique_pairs = {}  # canonical_smiles -> (original_smiles, merged_text)

    for cid, texts in tqdm(cid2text.items(), desc="Processing CIDs"):
        if cid not in cid2smiles:
            continue

        smiles = cid2smiles[cid]
        canon_smiles = canonicalize_smiles(smiles)
        if canon_smiles is None:
            continue

        # Get all texts for this CID
        if isinstance(texts, list):
            text_list = [t for t in texts if t and isinstance(t, str)]
        elif texts:
            text_list = [texts]
        else:
            continue

        if not text_list:
            continue

        # Merge texts: take the longest one (most informative)
        best_text = max(text_list, key=len)

        # If same canonical SMILES exists, keep the longer text
        if canon_smiles in unique_pairs:
            existing_text = unique_pairs[canon_smiles][1]
            if len(best_text) > len(existing_text):
                unique_pairs[canon_smiles] = (smiles, best_text)
        else:
            unique_pairs[canon_smiles] = (smiles, best_text)

    print(f"  Unique SMILES-text pairs: {len(unique_pairs)}")

    # Step 3: Filter by word count
    print(f"\nStep 3: Filtering texts with less than {args.min_word_count} words...")

    filtered_pairs = {
        k: v
        for k, v in unique_pairs.items()
        if len(v[1].split()) >= args.min_word_count
    }
    print(f"  After word count filtering: {len(filtered_pairs)}")

    # Step 4: Load downstream datasets for deduplication
    print("\nStep 4: Loading downstream datasets for deduplication...")

    print("  Loading ChEBI-20...")
    chebi20_smiles = load_chebi20_smiles(chebi20_dir)
    print(f"    ChEBI-20 SMILES: {len(chebi20_smiles)}")

    print("  Loading PCdes...")
    pcdes_smiles = load_pcdes_smiles(pcdes_dir)
    print(f"    PCdes SMILES: {len(pcdes_smiles)}")

    downstream_smiles = chebi20_smiles | pcdes_smiles
    print(f"  Total downstream SMILES: {len(downstream_smiles)}")

    # Step 5: Remove duplicates
    print("\nStep 5: Removing duplicates with downstream datasets...")

    final_pairs = {
        k: v for k, v in filtered_pairs.items() if k not in downstream_smiles
    }

    removed_count = len(filtered_pairs) - len(final_pairs)
    print(f"  Removed {removed_count} duplicates")
    print(f"  Final dataset size: {len(final_pairs)}")

    # Save dataset
    print(f"\nSaving to {output_dir}...")

    output_file = output_dir / "train.txt"
    with open(output_file, "w") as f:
        f.write("smiles\tdesc\n")
        for canon_smiles, (orig_smiles, text) in final_pairs.items():
            # Clean text (remove tabs and newlines)
            clean_text = text.replace("\t", " ").replace("\n", " ")
            f.write(f"{orig_smiles}\t{clean_text}\n")

    print(f"  Saved {len(final_pairs)} pairs to {output_file}")

    # Save statistics
    stats = {
        "total_raw_pairs": total_pairs,
        "unique_pairs_after_cid_merge": len(unique_pairs),
        "pairs_after_length_filter": len(filtered_pairs),
        "chebi20_smiles_count": len(chebi20_smiles),
        "pcdes_smiles_count": len(pcdes_smiles),
        "duplicates_removed": removed_count,
        "final_pairs": len(final_pairs),
    }

    stats_file = output_dir / "stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved statistics to {stats_file}")


if __name__ == "__main__":
    main()
