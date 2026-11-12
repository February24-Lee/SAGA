"""Centered Kernel Alignment (CKA) loss computation."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .tanimoto import tanimoto_gram_matrix

__all__ = [
    "linear_gram_matrix",
    "center_gram_matrix",
    "compute_cka",
    "MFPGuidedCKALoss",
]


def linear_gram_matrix(features: Tensor) -> Tensor:
    """
    Compute Gram matrix using linear kernel (dot product).

    Args:
        features: Feature tensor of shape (N, D).

    Returns:
        Gram matrix of shape (N, N) where G[i,j] = <x_i, x_j>.
    """
    return torch.mm(features, features.T)


def center_gram_matrix(gram: Tensor, unbiased: bool = False) -> Tensor:
    """
    Center a symmetric Gram matrix (HSIC centering).

    Centering is equivalent to centering the features in the kernel-induced
    feature space before computing the Gram matrix.

    Args:
        gram: Gram matrix of shape (N, N).
        unbiased: Use unbiased HSIC estimator. Note that this may produce
                  negative values.

    Returns:
        Centered Gram matrix of shape (N, N).
    """
    n = gram.size(0)
    gram = gram.clone()

    if unbiased:
        # Unbiased estimator (Song et al., 2012)
        gram.fill_diagonal_(0)
        means = gram.sum(dim=0) / (n - 2)
        means = means - means.sum() / (2 * (n - 1))
        gram = gram - means.unsqueeze(0) - means.unsqueeze(1)
        gram.fill_diagonal_(0)
    else:
        # Biased estimator
        means = gram.mean(dim=0)
        means = means - means.mean() / 2
        gram = gram - means.unsqueeze(0) - means.unsqueeze(1)

    return gram


def compute_cka(
    gram_x: Tensor,
    gram_y: Tensor,
    unbiased: bool = False,
) -> Tensor:
    """
    Compute CKA (Centered Kernel Alignment) between two Gram matrices.

    CKA measures the similarity between two representations by comparing
    their centered kernel matrices:

        CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))

    where HSIC is the Hilbert-Schmidt Independence Criterion.

    Args:
        gram_x: First Gram matrix of shape (N, N).
        gram_y: Second Gram matrix of shape (N, N).
        unbiased: Use unbiased HSIC estimator.

    Returns:
        CKA similarity value in [0, 1] (scalar tensor).
    """
    # Center the Gram matrices
    centered_x = center_gram_matrix(gram_x, unbiased=unbiased)
    centered_y = center_gram_matrix(gram_y, unbiased=unbiased)

    # HSIC(X, Y) ∝ <vec(K_X), vec(K_Y)> = sum(K_X * K_Y)
    # Note: Normalization constants cancel out in CKA
    hsic_xy = torch.sum(centered_x * centered_y)
    hsic_xx = torch.norm(centered_x)
    hsic_yy = torch.norm(centered_y)

    # CKA = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
    cka = hsic_xy / (hsic_xx * hsic_yy + 1e-8)

    return cka


class MFPGuidedCKALoss(nn.Module):
    """
    CKA loss between SMILES features and molecular fingerprints.

    This loss encourages the learned SMILES representations to preserve
    the structural similarity relationships captured by molecular fingerprints.

    Supports multiple fingerprint types:
        - Morgan fingerprints (2048 bits, default)
        - MACCS Keys (167 bits)
        - RDKit topological fingerprints (2048 bits)

    Uses:
        - Linear kernel for SMILES features (continuous embeddings)
        - Tanimoto kernel for fingerprints (binary vectors)

    The loss is computed as: L = 1 - CKA(K_smiles, K_fp)

    Example:
        >>> loss_fn = MFPGuidedCKALoss()
        >>> smiles_feats = torch.randn(32, 768)  # (batch, dim)
        >>> fp = torch.randint(0, 2, (32, 2048)).float()  # (batch, fp_dim)
        >>> loss = loss_fn(smiles_feats, fp)
    """

    def __init__(self, unbiased: bool = True) -> None:
        """
        Args:
            unbiased: Use unbiased CKA estimator. Recommended for small batches.
        """
        super().__init__()
        self.unbiased = unbiased

    def forward(
        self,
        smiles_features: Tensor,
        fingerprints: Tensor,
    ) -> Tensor:
        """
        Compute CKA loss between SMILES features and molecular fingerprints.

        Args:
            smiles_features: SMILES embedding tensor of shape (N, D).
                             Will be L2-normalized internally.
            fingerprints: Fingerprint tensor of shape (N, fp_dim).
                          Supports morgan (2048), maccs (167), or rdkit (2048).
                          Values should be binary {0, 1}.

        Returns:
            CKA loss value: 1 - CKA similarity (scalar in [0, 1]).
        """
        # Normalize SMILES features for stable Gram matrix computation
        smiles_features = torch.nn.functional.normalize(smiles_features, p=2, dim=1)

        # Compute Gram matrices
        gram_smiles = linear_gram_matrix(smiles_features)
        gram_mfp = tanimoto_gram_matrix(fingerprints)

        # Compute CKA similarity
        cka_similarity = compute_cka(
            gram_smiles,
            gram_mfp,
            unbiased=self.unbiased,
        )

        # Loss = 1 - similarity (minimize to maximize alignment)
        return 1.0 - cka_similarity
