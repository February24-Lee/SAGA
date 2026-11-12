import typing as tp
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset

from dataset.pubChemSTMDataset import load_mfp_cache


def load_chEBI20MM(data_dir: Path | str = "data/chEBI-20-MM"):
    ds = load_dataset(data_dir)

    return ds


def load_chEBI20MM_from_df(
    split: tp.Literal["train", "validation", "test"] = "train",
) -> pd.DataFrame:
    splits = {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}
    df = pd.read_csv("hf://datasets/liupf/ChEBI-20-MM/" + splits[split])

    return df


class ChEBI20MMDataset(Dataset):
    def __init__(
        self,
        split: tp.Literal["train", "validation", "test"] = "train",
        cache_path: str | Path | None = None,
        mfp_cache_path: str | Path | None = None,
    ):
        self.split = split
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.mfp_cache = load_mfp_cache(mfp_cache_path)

        if self.cache_path is not None:
            split_dir = self.cache_path / split
            if not split_dir.exists():
                raise FileNotFoundError(
                    f"Cached split '{split}' not found under {self.cache_path}."
                )
            self.cached_dataset = load_from_disk(str(split_dir))
        else:
            self.cached_dataset = None

        if self.cached_dataset is None:
            self.df = load_chEBI20MM_from_df(split)
        else:
            self.df = None

    def __len__(self):
        if self.cached_dataset is not None:
            return len(self.cached_dataset)
        return len(self.df)

    def __getitem__(self, idx):
        if self.cached_dataset is not None:
            sample = self.cached_dataset[int(idx)]
            item = {
                "description": sample["description"],
                "smiles": sample["smiles"],
            }

            if (
                "description_input_ids" in sample
                and "description_attention_mask" in sample
            ):
                item["description_token"] = {
                    "input_ids": torch.tensor(
                        sample["description_input_ids"], dtype=torch.long
                    ),
                    "attention_mask": torch.tensor(
                        sample["description_attention_mask"], dtype=torch.long
                    ),
                }

            if "smiles_input_ids" in sample and "smiles_attention_mask" in sample:
                item["smiles_token"] = {
                    "input_ids": torch.tensor(
                        sample["smiles_input_ids"], dtype=torch.long
                    ),
                    "attention_mask": torch.tensor(
                        sample["smiles_attention_mask"], dtype=torch.long
                    ),
                }

            if self.mfp_cache is not None:
                smiles = sample["smiles"]
                if smiles in self.mfp_cache:
                    item["mfp"] = self.mfp_cache[smiles]

            return item

        smiles = self.df.iloc[idx]["SMILES"]
        description = self.df.iloc[idx]["description"]
        item = {"description": description, "smiles": smiles}

        if self.mfp_cache is not None and smiles in self.mfp_cache:
            item["mfp"] = self.mfp_cache[smiles]

        return item
