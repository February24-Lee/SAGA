"""Centered Kernel Alignment (CKA) between two Gram matrices.

The SAGA loss is ``1 - CKA(K_embedding, K_fingerprint)``: a cosine Gram matrix over
the learned embeddings against a Tanimoto Gram matrix over the molecular
fingerprints, compared with the unbiased HSIC estimator of Song et al. (2012).
"""

import torch


def gram_tanimoto(x: torch.Tensor) -> torch.Tensor:
    """Compute Tanimoto similarity Gram matrix for binary fingerprints.

    Tanimoto(A, B) = (A · B) / (|A| + |B| - A · B)

    Args:
        x: N x D tensor of binary fingerprints (0/1 values)

    Returns:
        N x N Tanimoto similarity matrix (values in [0, 1])
    """
    # Dot products: N x N
    dot_product = torch.mm(x, x.T)

    # Sum of each fingerprint (for binary = count of 1s)
    sums = torch.sum(x, dim=1)  # N

    # Tanimoto: dot / (sum_a + sum_b - dot)
    denominator = sums.unsqueeze(1) + sums.unsqueeze(0) - dot_product
    tanimoto = dot_product / (denominator + 1e-8)

    return tanimoto


def gram_cosine(x: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity Gram matrix.

    Args:
        x: N x D tensor

    Returns:
        N x N cosine similarity matrix (values in [-1, 1])
    """
    x_norm = x / (torch.norm(x, p=2, dim=1, keepdim=True) + 1e-8)
    return torch.mm(x_norm, x_norm.T)


def center_gram(gram: torch.Tensor, unbiased: bool = False) -> torch.Tensor:
    """Center a symmetric Gram matrix, in place.

    This is equivalent to centering the (possibly infinite-dimensional) features
    induced by the kernel before computing the Gram matrix.

    Args:
      gram: A num_examples x num_examples symmetric matrix.
      unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
        estimate of HSIC. The diagonal is set to zero, so self-similarities do not
        enter the estimate. Note that this estimator may be negative.

    Returns:
      A symmetric matrix with centered columns and rows.
    """
    if unbiased:
        n = gram.shape[0]
        gram.fill_diagonal_(0)
        means = torch.sum(gram, 0, dtype=torch.float64) / (n - 2)
        means -= torch.sum(means) / (2 * (n - 1))
        gram -= means.unsqueeze(1)
        gram -= means.unsqueeze(0)
        gram.fill_diagonal_(0)
    else:
        means = torch.mean(gram, 0, dtype=torch.float64)
        means -= torch.mean(means) / 2
        gram -= means.unsqueeze(1)
        gram -= means.unsqueeze(0)
    return gram


def cka(
    gram_x: torch.Tensor, gram_y: torch.Tensor, debiased: bool = False
) -> torch.Tensor:
    """Compute CKA.

    Args:
      gram_x: A num_examples x num_examples Gram matrix.
      gram_y: A num_examples x num_examples Gram matrix.
      debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
      The value of CKA between X and Y (a scalar).
    """
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    scaled_hsic = torch.sum(gram_x * gram_y)
    normalization_x = torch.norm(gram_x)
    normalization_y = torch.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)


def cka_loss_from_gram(
    gram_x: torch.Tensor, gram_y: torch.Tensor, debiased: bool = False
) -> torch.Tensor:
    """Compute CKA loss between two Gram matrices.

    Args:
        gram_x: A num_examples x num_examples Gram matrix.
        gram_y: A num_examples x num_examples Gram matrix.
        debiased: Whether to use debiased estimator.

    Returns:
        CKA loss value (1 - CKA similarity), a scalar.
    """
    cka_similarity = cka(gram_x, gram_y, debiased=debiased)
    return 1.0 - cka_similarity
