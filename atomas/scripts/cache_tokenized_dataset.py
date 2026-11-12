#!/usr/bin/env python
import argparse
import json
import shutil
from pathlib import Path

from datasets import Dataset
from tqdm import tqdm
from transformers import T5Tokenizer

from dataset.chEBI20MMDataset import ChEBI20MMDataset
from dataset.pubChemSTMDataset import (
    PubChemSTM_Datasets_SMILES,
    PubChemSTM_Distill_Dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-tokenise datasets and store them as Hugging Face Arrow caches."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["ChEBI-20_data", "pubchemstm", "pubchemstm_distill"],
        help="Dataset to preprocess.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root directory for raw PubChemSTM files (ignored for ChEBI-20).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Output directory where cached splits will be stored.",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="base",
        choices=["base", "large"],
        help="Model size (determines tokenizer and cache subdirectory).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer checkpoint used to produce input ids. Defaults to laituan245/molt5-{model_size}.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenisation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cached splits if present.",
    )
    return parser.parse_args()


def encode_text(tokenizer: T5Tokenizer, text: str, max_length: int) -> dict:
    encoding = tokenizer(
        text,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return {
        "input_ids": encoding["input_ids"].squeeze(0).tolist(),
        "attention_mask": encoding["attention_mask"].squeeze(0).tolist(),
    }


def build_cached_split(
    base_dataset,
    tokenizer: T5Tokenizer,
    max_length: int,
    progress_label: str,
) -> Dataset:
    total = len(base_dataset)

    def generator():
        for idx in tqdm(range(total), desc=progress_label):
            sample = base_dataset[idx]
            description = sample["description"]
            smiles = sample["smiles"]
            description_tokens = encode_text(tokenizer, description, max_length)
            smiles_tokens = encode_text(tokenizer, smiles, max_length)
            yield {
                "description": description,
                "smiles": smiles,
                "description_input_ids": description_tokens["input_ids"],
                "description_attention_mask": description_tokens["attention_mask"],
                "smiles_input_ids": smiles_tokens["input_ids"],
                "smiles_attention_mask": smiles_tokens["attention_mask"],
            }

    return Dataset.from_generator(generator)


def main() -> None:
    args = parse_args()
    if args.tokenizer is None:
        args.tokenizer = f"laituan245/molt5-{args.model_size}"

    cache_root = Path(args.cache_dir) / args.dataset / args.model_size
    cache_root.mkdir(parents=True, exist_ok=True)

    tokenizer = T5Tokenizer.from_pretrained(
        args.tokenizer, model_max_length=args.max_length, legacy=False
    )

    if args.dataset == "ChEBI-20_data":
        split_builders = [
            ("train", lambda: ChEBI20MMDataset("train")),
            ("validation", lambda: ChEBI20MMDataset("validation")),
            ("test", lambda: ChEBI20MMDataset("test")),
        ]
    elif args.dataset == "pubchemstm_distill":
        split_builders = [("train", lambda: PubChemSTM_Distill_Dataset(args.data_dir))]
    else:
        split_builders = [("train", lambda: PubChemSTM_Datasets_SMILES(args.data_dir))]

    metadata = {
        "tokenizer": args.tokenizer,
        "max_length": args.max_length,
    }

    for split, builder in split_builders:
        split_dir = cache_root / split
        if split_dir.exists():
            if args.overwrite:
                shutil.rmtree(split_dir)
            else:
                print(f"Split '{split}' already exists at {split_dir}. Skipping.")
                continue

        dataset_obj = builder()
        cached_split = build_cached_split(
            dataset_obj, tokenizer, args.max_length, progress_label=split
        )
        cached_split.save_to_disk(str(split_dir))
        metadata_path = split_dir / "cache_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))
        print(f"Saved cached split '{split}' to {split_dir}.")


if __name__ == "__main__":
    main()
