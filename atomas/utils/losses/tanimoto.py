"""Tanimoto similarity computation for binary fingerprints."""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["tanimoto_similarity_matrix", "tanimoto_gram_matrix"]


def tanimoto_similarity_matrix(fingerprints: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Compute pairwise Tanimoto similarity matrix for binary fingerprints.

    Tanimoto similarity (Jaccard index for binary vectors):
        T(A, B) = |A ∩ B| / |A ∪ B|
                = |A ∩ B| / (|A| + |B| - |A ∩ B|)

    For binary vectors, intersection is computed via dot product.

    Args:
        fingerprints: Binary fingerprint tensor of shape (N, D).
                      Values should be in {0, 1}.
        eps: Small constant for numerical stability.

    Returns:
        Symmetric similarity matrix of shape (N, N) with values in [0, 1].
        Diagonal elements are 1.0 (self-similarity).

    Example:
        >>> fp = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.float32)
        >>> sim = tanimoto_similarity_matrix(fp)
        >>> sim.shape
        torch.Size([2, 2])
        >>> sim[0, 0]  # Self-similarity
        tensor(1.)
    """
    # |A ∩ B| via dot product: (N, D) @ (D, N) -> (N, N)
    intersection = torch.mm(fingerprints, fingerprints.T)

    # |A| for each sample: (N,)
    bits_per_sample = fingerprints.sum(dim=1)

    # |A| + |B| for all pairs: (N, 1) + (1, N) -> (N, N)
    sum_bits = bits_per_sample.unsqueeze(1) + bits_per_sample.unsqueeze(0)

    # |A ∪ B| = |A| + |B| - |A ∩ B|
    union = sum_bits - intersection

    # Tanimoto = |A ∩ B| / |A ∪ B|
    tanimoto = intersection / (union + eps)

    return tanimoto


def tanimoto_gram_matrix(fingerprints: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Compute Tanimoto Gram matrix for CKA computation.

    This is an alias for tanimoto_similarity_matrix, emphasizing its use
    as a kernel/Gram matrix in CKA computation.

    Args:
        fingerprints: Binary fingerprint tensor of shape (N, D).
        eps: Small constant for numerical stability.

    Returns:
        Gram matrix of shape (N, N) using Tanimoto kernel.
    """
    return tanimoto_similarity_matrix(fingerprints, eps=eps)
