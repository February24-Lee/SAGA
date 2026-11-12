import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def load_mfp_cache(mfp_cache_path: str | Path | None) -> dict | None:
    """Load pre-computed MFP cache from disk.

    Args:
        mfp_cache_path: Path to .npy file containing SMILES->MFP mapping.

    Returns:
        Dictionary mapping SMILES to MFP arrays, or None if not provided.
    """
    if mfp_cache_path is None:
        return None

    mfp_cache_path = Path(mfp_cache_path)
    if not mfp_cache_path.exists():
        raise FileNotFoundError(f"MFP cache not found at {mfp_cache_path}")

    logger.info(f"Loading MFP cache from {mfp_cache_path}")
    mfp_cache = np.load(mfp_cache_path, allow_pickle=True).item()
    logger.info(f"Loaded {len(mfp_cache)} SMILES->MFP mappings")
    return mfp_cache


class PubChemSTM_Datasets_SMILES(Dataset):
    def __init__(
        self,
        root,
        cache_path: str | Path | None = None,
        mfp_cache_path: str | Path | None = None,
    ):
        self.root = root
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.mfp_cache = load_mfp_cache(mfp_cache_path)

        if self.cache_path is not None:
            if not self.cache_path.exists():
                raise FileNotFoundError(
                    f"Cached dataset path '{self.cache_path}' does not exist."
                )
            self.cached_dataset = load_from_disk(str(self.cache_path))
        else:
            self.cached_dataset = None

        if self.cached_dataset is None:
            CID2text_file = os.path.join(self.root, "raw/CID2text.json")
            CID2SMILES_file = os.path.join(self.root, "raw/CID2SMILES.csv")
            self.load_CID2SMILES(CID2text_file, CID2SMILES_file)

            self.text_list = []
            missing_count = 0
            for CID, value_list in self.CID2text_data.items():
                if CID not in self.CID2SMILES:
                    logger.debug(f"CID {CID} missing")
                    missing_count += 1
                    continue
                for value in value_list:
                    if value is None or value == "":
                        logger.debug(f"value is None or empty: {value}")
                        continue
                    self.text_list.append([CID, value])
            logger.info(f"missing {missing_count}")
            logger.info(f"len of text_list: {len(self.text_list)}")
        else:
            self.text_list = None

    def load_CID2SMILES(self, CID2text_file, CID2SMILES_file):
        with open(CID2text_file) as f:
            self.CID2text_data = json.load(f)
        logger.info(f"len of CID2text: {len(self.CID2text_data.keys())}")

        df = pd.read_csv(CID2SMILES_file)
        CID_list, SMILES_list = df["CID"].tolist(), df["SMILES"].tolist()
        self.CID2SMILES = {}
        for CID, SMILES in zip(CID_list, SMILES_list):
            CID = str(CID)
            self.CID2SMILES[CID] = SMILES
        logger.info(f"len of CID2SMILES: {len(self.CID2SMILES.keys())}")
        return

    def __getitem__(self, index) -> dict:
        if self.cached_dataset is not None:
            sample = self.cached_dataset[int(index)]
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

            # Add MFP from cache if available
            if self.mfp_cache is not None:
                smiles = sample["smiles"]
                if smiles in self.mfp_cache:
                    item["mfp"] = self.mfp_cache[smiles]

            return item

        CID, description = self.text_list[index]
        smiles = self.CID2SMILES[CID]
        item = {"description": description, "smiles": smiles}

        # Add MFP from cache if available
        if self.mfp_cache is not None and smiles in self.mfp_cache:
            item["mfp"] = self.mfp_cache[smiles]

        return item

    def __len__(self):
        if self.cached_dataset is not None:
            return len(self.cached_dataset)
        return len(self.text_list)


class PubChemSTM_Distill_Dataset(Dataset):
    """Dataset for PubchemSTM-distill (filtered 34k pairs)."""

    def __init__(
        self,
        data_dir: str | Path,
        cache_path: str | Path | None = None,
        mfp_cache_path: str | Path | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.mfp_cache = load_mfp_cache(mfp_cache_path)

        if self.cache_path is not None:
            if not self.cache_path.exists():
                raise FileNotFoundError(
                    f"Cached dataset path '{self.cache_path}' does not exist."
                )
            self.cached_dataset = load_from_disk(str(self.cache_path))
            self.data_list = None
            logger.info(
                f"Loaded cached dataset with {len(self.cached_dataset)} samples"
            )
        else:
            self.cached_dataset = None
            # Load from TSV file
            tsv_file = self.data_dir / "PubchemSTM_distill" / "train.txt"
            if not tsv_file.exists():
                raise FileNotFoundError(f"Distill dataset not found at {tsv_file}")

            df = pd.read_csv(tsv_file, sep="\t")
            self.data_list = list(zip(df["smiles"].tolist(), df["desc"].tolist()))
            logger.info(f"Loaded {len(self.data_list)} samples from {tsv_file}")

    def __getitem__(self, index) -> dict:
        if self.cached_dataset is not None:
            sample = self.cached_dataset[int(index)]
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

        smiles, description = self.data_list[index]
        item = {"description": description, "smiles": smiles}

        if self.mfp_cache is not None and smiles in self.mfp_cache:
            item["mfp"] = self.mfp_cache[smiles]

        return item

    def __len__(self):
        if self.cached_dataset is not None:
            return len(self.cached_dataset)
        return len(self.data_list)


class PubChemSTM_Datasets_Raw_SMILES(PubChemSTM_Datasets_SMILES):
    def __init__(
        self,
        root,
        cache_path: str | Path | None = None,
        mfp_cache_path: str | Path | None = None,
    ):
        self.root = root
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.mfp_cache = load_mfp_cache(mfp_cache_path)

        if self.cache_path is not None:
            if not self.cache_path.exists():
                raise FileNotFoundError(
                    f"Cached dataset path '{self.cache_path}' does not exist."
                )
            self.cached_dataset = load_from_disk(str(self.cache_path))
        else:
            self.cached_dataset = None

        if self.cached_dataset is None:
            CID2text_file = os.path.join(self.root, "raw/CID2text_raw.json")
            CID2SMILES_file = os.path.join(self.root, "raw/CID2SMILES.csv")
            self.load_CID2SMILES(CID2text_file, CID2SMILES_file)

            self.text_list = []
            missing_count = 0
            for CID, value_list in self.CID2text_data.items():
                if CID not in self.CID2SMILES:
                    logger.warning(f"CID {CID} missing")
                    missing_count += 1
                    continue
                for value in value_list:
                    self.text_list.append([CID, value])
            logger.info(f"missing {missing_count}")
            logger.info(f"len of text_list: {len(self.text_list)}")
        else:
            self.text_list = None
