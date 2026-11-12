import logging
from typing import List, Literal, Union

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

logger = logging.getLogger(__name__)

# Fingerprint type alias
FingerprintType = Literal["morgan", "maccs", "rdkit"]


def turn_off_rdkit_warnings():
    RDLogger.DisableLog("rdApp.*")
    logger.info("RDKit warnings are turned off")


def smiles2mfp(
    smiles: Union[str, List[str]], radius: int = 2, n_bits: int = 2048
) -> np.ndarray:
    """
    Convert SMILES string(s) to Morgan fingerprint(s).

    Args:
        smiles: SMILES string or list of SMILES strings
        radius: Radius for Morgan fingerprint (default: 2)
        n_bits: Number of bits for fingerprint (default: 2048)

    Returns:
        Morgan fingerprint numpy array (shape: B x n_bits for batch, 1 x n_bits for single)
    """
    if isinstance(smiles, str):
        smiles = [smiles]

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    fingerprints = []
    for smi in smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                logger.warning(f"Invalid SMILES string: {smi}. Returning zero vector.")
                fp = np.zeros(n_bits, dtype=np.float32)
            else:
                morgan_fp = mfpgen.GetFingerprint(mol)
                fp = np.array(morgan_fp, dtype=np.float32)
            fingerprints.append(fp)
        except Exception as e:
            logger.warning(
                f"Error processing SMILES {smi}: {e}. Returning zero vector."
            )
            fingerprints.append(np.zeros(n_bits, dtype=np.float32))

    return np.array(fingerprints)


def smiles2maccs(smiles: Union[str, List[str]]) -> np.ndarray:
    """
    Convert SMILES string(s) to MACCS keys fingerprint(s).

    Args:
        smiles: SMILES string or list of SMILES strings

    Returns:
        MACCS fingerprint numpy array (shape: B x 167)
        Note: MACCS keys produce 167 bits (index 0 is unused, but included for compatibility)
    """
    if isinstance(smiles, str):
        smiles = [smiles]

    n_bits = 167  # MACCS keys fixed size

    fingerprints = []
    for smi in smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                logger.warning(f"Invalid SMILES string: {smi}. Returning zero vector.")
                fp = np.zeros(n_bits, dtype=np.float32)
            else:
                maccs_fp = MACCSkeys.GenMACCSKeys(mol)
                fp = np.array(maccs_fp, dtype=np.float32)
            fingerprints.append(fp)
        except Exception as e:
            logger.warning(
                f"Error processing SMILES {smi}: {e}. Returning zero vector."
            )
            fingerprints.append(np.zeros(n_bits, dtype=np.float32))

    return np.array(fingerprints)


def smiles2rdkit(
    smiles: Union[str, List[str]], min_path: int = 1, max_path: int = 7, n_bits: int = 2048
) -> np.ndarray:
    """
    Convert SMILES string(s) to RDKit topological fingerprint(s).

    Args:
        smiles: SMILES string or list of SMILES strings
        min_path: Minimum path length (default: 1)
        max_path: Maximum path length (default: 7)
        n_bits: Number of bits for fingerprint (default: 2048)

    Returns:
        RDKit fingerprint numpy array (shape: B x n_bits)
    """
    if isinstance(smiles, str):
        smiles = [smiles]

    rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(
        minPath=min_path, maxPath=max_path, fpSize=n_bits
    )

    fingerprints = []
    for smi in smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                logger.warning(f"Invalid SMILES string: {smi}. Returning zero vector.")
                fp = np.zeros(n_bits, dtype=np.float32)
            else:
                rdkit_fp = rdkit_gen.GetFingerprint(mol)
                fp = np.array(rdkit_fp, dtype=np.float32)
            fingerprints.append(fp)
        except Exception as e:
            logger.warning(
                f"Error processing SMILES {smi}: {e}. Returning zero vector."
            )
            fingerprints.append(np.zeros(n_bits, dtype=np.float32))

    return np.array(fingerprints)


def smiles2fingerprint(
    smiles: Union[str, List[str]],
    fp_type: FingerprintType = "morgan",
    **kwargs
) -> np.ndarray:
    """
    Unified interface for converting SMILES to molecular fingerprints.

    Args:
        smiles: SMILES string or list of SMILES strings
        fp_type: Type of fingerprint ("morgan", "maccs", "rdkit")
        **kwargs: Additional arguments passed to the specific fingerprint function
            - morgan: radius (default=2), n_bits (default=2048)
            - maccs: (no additional args, fixed 167 bits)
            - rdkit: min_path (default=1), max_path (default=7), n_bits (default=2048)

    Returns:
        Fingerprint numpy array (shape depends on fp_type)
    """
    if fp_type == "morgan":
        return smiles2mfp(smiles, **kwargs)
    elif fp_type == "maccs":
        return smiles2maccs(smiles)
    elif fp_type == "rdkit":
        return smiles2rdkit(smiles, **kwargs)
    else:
        raise ValueError(f"Unknown fingerprint type: {fp_type}. Choose from 'morgan', 'maccs', 'rdkit'.")
