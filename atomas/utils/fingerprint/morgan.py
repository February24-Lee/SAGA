"""Molecular fingerprint generation utilities (Morgan, MACCS, RDKit)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

logger = logging.getLogger(__name__)

# Type for fingerprint selection
FingerprintType = Literal["morgan", "maccs", "rdkit"]

# Fingerprint dimensions
FP_DIMENSIONS: dict[FingerprintType, int] = {
    "morgan": 2048,
    "maccs": 167,
    "rdkit": 2048,
}


def disable_rdkit_logging() -> None:
    """Disable RDKit warning messages."""
    RDLogger.DisableLog("rdApp.*")


def smiles_to_morgan_fingerprint(
    smiles: str | Sequence[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> NDArray[np.float32]:
    """
    Convert SMILES string(s) to Morgan fingerprint(s).

    Args:
        smiles: Single SMILES string or sequence of SMILES strings.
        radius: Morgan fingerprint radius (default: 2).
        n_bits: Fingerprint bit length (default: 2048).

    Returns:
        Morgan fingerprint array of shape (N, n_bits) with dtype float32.
        Values are binary {0.0, 1.0}. Invalid SMILES return zero vectors.

    Example:
        >>> fps = smiles_to_morgan_fingerprint(["CCO", "C1=CC=CC=C1"])
        >>> fps.shape
        (2, 2048)
    """
    if isinstance(smiles, str):
        smiles = [smiles]

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
    )

    fingerprints: list[NDArray[np.float32]] = []

    for smi in smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smi}. Using zero vector.")
                fp = np.zeros(n_bits, dtype=np.float32)
            else:
                morgan_fp = generator.GetFingerprint(mol)
                fp = np.zeros(n_bits, dtype=np.float32)
                DataStructs.ConvertToNumpyArray(morgan_fp, fp)
            fingerprints.append(fp)
        except Exception as e:
            logger.warning(f"Error processing SMILES '{smi}': {e}. Using zero vector.")
            fingerprints.append(np.zeros(n_bits, dtype=np.float32))

    return np.stack(fingerprints, axis=0)


def smiles_to_maccs_fingerprint(
    smiles: str | Sequence[str],
) -> NDArray[np.float32]:
    """
    Convert SMILES string(s) to MACCS Keys fingerprint(s).

    MACCS Keys are a set of 166 structural keys (167 bits total, index 0 unused).

    Args:
        smiles: Single SMILES string or sequence of SMILES strings.

    Returns:
        MACCS fingerprint array of shape (N, 167) with dtype float32.
        Values are binary {0.0, 1.0}. Invalid SMILES return zero vectors.

    Example:
        >>> fps = smiles_to_maccs_fingerprint(["CCO", "C1=CC=CC=C1"])
        >>> fps.shape
        (2, 167)
    """
    if isinstance(smiles, str):
        smiles = [smiles]

    n_bits = 167  # MACCS keys are fixed at 167 bits
    fingerprints: list[NDArray[np.float32]] = []

    for smi in smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smi}. Using zero vector.")
                fp = np.zeros(n_bits, dtype=np.float32)
            else:
                maccs_fp = MACCSkeys.GenMACCSKeys(mol)
                fp = np.zeros(n_bits, dtype=np.float32)
                DataStructs.ConvertToNumpyArray(maccs_fp, fp)
            fingerprints.append(fp)
        except Exception as e:
            logger.warning(f"Error processing SMILES '{smi}': {e}. Using zero vector.")
            fingerprints.append(np.zeros(n_bits, dtype=np.float32))

    return np.stack(fingerprints, axis=0)


def smiles_to_rdkit_fingerprint(
    smiles: str | Sequence[str],
    min_path: int = 1,
    max_path: int = 7,
    n_bits: int = 2048,
) -> NDArray[np.float32]:
    """
    Convert SMILES string(s) to RDKit topological fingerprint(s).

    RDKit fingerprints encode paths of atoms in the molecule.

    Args:
        smiles: Single SMILES string or sequence of SMILES strings.
        min_path: Minimum path length (default: 1).
        max_path: Maximum path length (default: 7).
        n_bits: Fingerprint bit length (default: 2048).

    Returns:
        RDKit fingerprint array of shape (N, n_bits) with dtype float32.
        Values are binary {0.0, 1.0}. Invalid SMILES return zero vectors.

    Example:
        >>> fps = smiles_to_rdkit_fingerprint(["CCO", "C1=CC=CC=C1"])
        >>> fps.shape
        (2, 2048)
    """
    if isinstance(smiles, str):
        smiles = [smiles]

    generator = rdFingerprintGenerator.GetRDKitFPGenerator(
        minPath=min_path,
        maxPath=max_path,
        fpSize=n_bits,
    )

    fingerprints: list[NDArray[np.float32]] = []

    for smi in smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smi}. Using zero vector.")
                fp = np.zeros(n_bits, dtype=np.float32)
            else:
                rdkit_fp = generator.GetFingerprint(mol)
                fp = np.zeros(n_bits, dtype=np.float32)
                DataStructs.ConvertToNumpyArray(rdkit_fp, fp)
            fingerprints.append(fp)
        except Exception as e:
            logger.warning(f"Error processing SMILES '{smi}': {e}. Using zero vector.")
            fingerprints.append(np.zeros(n_bits, dtype=np.float32))

    return np.stack(fingerprints, axis=0)


def smiles_to_fingerprint(
    smiles: str | Sequence[str],
    fp_type: FingerprintType = "morgan",
    **kwargs,
) -> NDArray[np.float32]:
    """
    Convert SMILES string(s) to fingerprint(s) of the specified type.

    This is a unified interface for all fingerprint types.

    Args:
        smiles: Single SMILES string or sequence of SMILES strings.
        fp_type: Type of fingerprint ("morgan", "maccs", or "rdkit").
        **kwargs: Additional arguments passed to the specific fingerprint function.
            - morgan: radius (int), n_bits (int)
            - maccs: none (fixed 167 bits)
            - rdkit: min_path (int), max_path (int), n_bits (int)

    Returns:
        Fingerprint array of shape (N, dim) with dtype float32.
        Dimension depends on fp_type: morgan=2048, maccs=167, rdkit=2048.

    Example:
        >>> fps = smiles_to_fingerprint(["CCO"], fp_type="morgan")
        >>> fps.shape
        (1, 2048)
        >>> fps = smiles_to_fingerprint(["CCO"], fp_type="maccs")
        >>> fps.shape
        (1, 167)
    """
    if fp_type == "morgan":
        return smiles_to_morgan_fingerprint(smiles, **kwargs)
    elif fp_type == "maccs":
        return smiles_to_maccs_fingerprint(smiles)
    elif fp_type == "rdkit":
        return smiles_to_rdkit_fingerprint(smiles, **kwargs)
    else:
        raise ValueError(
            f"Unknown fingerprint type: {fp_type}. Choose from: morgan, maccs, rdkit"
        )
