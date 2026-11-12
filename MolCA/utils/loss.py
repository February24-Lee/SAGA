import torch
from torch import Tensor, nn

from .cka import cka_loss_from_gram, gram_cosine, gram_tanimoto


class MiniBatchCKALoss(nn.Module):
    def __init__(self, debiased: bool = True, pool_method: str = "flatten"):
        """
        Args:
            debiased: Whether to use debiased CKA estimator
            pool_method: How to handle graph embeddings ('flatten' or 'mean')
        """
        super().__init__()
        self.debiased = debiased
        self.pool_method = pool_method

    def forward(self, graph_embedding: Tensor, mfp: Tensor) -> Tensor:
        """
        Compute minibatch CKA loss between graph embeddings and molecular fingerprints.

        Args:
            graph_embedding: B x qn x D (query tokens x feature dim)
            mfp: B x M (molecular fingerprint features)

        Returns:
            minibatch cka loss: 1 - cka (float32)
        """
        batch_size = graph_embedding.size(0)

        # Process graph embeddings based on pooling method
        if self.pool_method == "flatten":
            # Flatten: B x qn x D -> B x (qn * D)
            graph_features = graph_embedding.view(batch_size, -1)
        elif self.pool_method == "mean":
            # Mean pooling: B x qn x D -> B x D
            graph_features = torch.mean(graph_embedding, dim=1)
        else:
            raise ValueError(
                f"Unknown pool_method: {self.pool_method}. Use 'flatten' or 'mean'."
            )

        # Ensure both tensors are float
        graph_features = graph_features.float()
        mfp = mfp.float()

        # MFP: Tanimoto similarity Gram matrix (optimal for binary fingerprints)
        gram_mfp = gram_tanimoto(mfp)  # N x N

        # Graph: Cosine similarity Gram matrix (optimal for continuous embeddings)
        gram_graph = gram_cosine(graph_features)  # N x N

        # CKA loss from Gram matrices (unbiased)
        return cka_loss_from_gram(gram_graph, gram_mfp, debiased=self.debiased)
