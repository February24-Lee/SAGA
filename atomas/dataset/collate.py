"""Custom collate functions for Atomas DataLoader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from utils.fingerprint import FingerprintType, smiles_to_fingerprint


@dataclass
class AtomasBatch:
    """
    Batch container for Atomas training.

    This dataclass provides a structured way to pass batch data through
    the training pipeline, replacing unstructured dictionaries.

    Attributes:
        description: List of text descriptions.
        smiles: List of SMILES strings.
        description_token: Pre-tokenized descriptions (optional).
        smiles_token: Pre-tokenized SMILES (optional).
        mfp: Molecular fingerprints tensor of shape (B, fp_dim). None if not included.
             Dimension depends on fp_type: morgan=2048, maccs=167, rdkit=2048.

    Example:
        >>> batch = AtomasBatch(
        ...     description=["molecule A desc", "molecule B desc"],
        ...     smiles=["CCO", "C1=CC=CC=C1"],
        ...     mfp=torch.zeros(2, 2048),  # or (2, 167) for MACCS
        ... )
    """

    description: list[str]
    smiles: list[str]
    description_token: dict[str, Tensor] | None = None
    smiles_token: dict[str, Tensor] | None = None
    mfp: Tensor | None = None

    def __len__(self) -> int:
        """Return batch size."""
        return len(self.description)

    def to(self, device: torch.device) -> AtomasBatch:
        """
        Move tensors to specified device.

        Args:
            device: Target device.

        Returns:
            New AtomasBatch with tensors on target device.
        """
        return AtomasBatch(
            description=self.description,
            smiles=self.smiles,
            description_token=(
                {k: v.to(device) for k, v in self.description_token.items()}
                if self.description_token is not None
                else None
            ),
            smiles_token=(
                {k: v.to(device) for k, v in self.smiles_token.items()}
                if self.smiles_token is not None
                else None
            ),
            mfp=self.mfp.to(device) if self.mfp is not None else None,
        )


def collate_atomas_batch(
    batch: list[dict[str, Any]],
    include_mfp: bool = False,
    fp_type: FingerprintType = "morgan",
) -> AtomasBatch:
    """
    Collate function for Atomas dataset.

    Args:
        batch: List of sample dictionaries from dataset __getitem__.
               Each dict should have 'description' and 'smiles' keys.
               Optionally 'description_token' and 'smiles_token'.
        include_mfp: Whether to compute and include fingerprints.
        fp_type: Type of fingerprint to use ("morgan", "maccs", or "rdkit").

    Returns:
        AtomasBatch with collated data.
    """
    descriptions = [sample["description"] for sample in batch]
    smiles_list = [sample["smiles"] for sample in batch]

    # Handle pre-tokenized data if present
    description_token = None
    smiles_token = None

    if "description_token" in batch[0] and batch[0]["description_token"] is not None:
        description_token = _collate_tokens([s["description_token"] for s in batch])

    if "smiles_token" in batch[0] and batch[0]["smiles_token"] is not None:
        smiles_token = _collate_tokens([s["smiles_token"] for s in batch])

    # Get fingerprints if requested
    mfp = None
    if include_mfp:
        if "mfp" in batch[0] and batch[0]["mfp"] is not None:
            # Use pre-computed fingerprint from cache
            mfp = torch.stack([torch.from_numpy(s["mfp"]) for s in batch])
        else:
            # Fallback: compute on-the-fly (slower)
            mfp_np = smiles_to_fingerprint(smiles_list, fp_type=fp_type)
            mfp = torch.from_numpy(mfp_np)

    return AtomasBatch(
        description=descriptions,
        smiles=smiles_list,
        description_token=description_token,
        smiles_token=smiles_token,
        mfp=mfp,
    )


def _collate_tokens(token_list: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """
    Collate list of token dictionaries into batched tensors.

    Args:
        token_list: List of token dictionaries with 'input_ids' and 'attention_mask'.

    Returns:
        Dictionary with batched tensors.
    """
    return {
        "input_ids": torch.stack([t["input_ids"] for t in token_list]),
        "attention_mask": torch.stack([t["attention_mask"] for t in token_list]),
    }


class AtomasCollator:
    """
    Callable collator class for DataLoader.

    This class wraps collate_atomas_batch to be used as collate_fn
    in PyTorch DataLoader.

    Example:
        >>> collator = AtomasCollator(include_mfp=True, fp_type="morgan")
        >>> loader = DataLoader(dataset, collate_fn=collator)
    """

    def __init__(
        self,
        include_mfp: bool = False,
        fp_type: FingerprintType = "morgan",
    ) -> None:
        """
        Args:
            include_mfp: Whether to compute fingerprints for each batch.
            fp_type: Type of fingerprint to use ("morgan", "maccs", or "rdkit").
        """
        self.include_mfp = include_mfp
        self.fp_type = fp_type

    def __call__(self, batch: list[dict[str, Any]]) -> AtomasBatch:
        """
        Collate a batch of samples.

        Args:
            batch: List of sample dictionaries.

        Returns:
            AtomasBatch with collated data.
        """
        return collate_atomas_batch(
            batch,
            include_mfp=self.include_mfp,
            fp_type=self.fp_type,
        )
