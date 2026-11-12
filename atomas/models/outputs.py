"""Model output containers."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class AtomasOutput:
    """
    Output container for Atomas model forward pass.

    This dataclass provides a structured way to return multiple loss values
    from the forward pass, avoiding variable-length tuple returns.

    Attributes:
        loss_tsc: Text-SMILES contrastive loss (MoCo-style).
        loss_lm: Language modeling loss (encoder-decoder).
        loss_wti: Word-Token Interaction loss (hierarchical alignment).
        loss_cka: MFP-guided CKA loss. None if include_mfp=False.

    Example:
        >>> output = model(text, smiles, alpha)
        >>> total = output.loss_tsc + output.loss_lm + output.loss_wti
        >>> if output.loss_cka is not None:
        ...     total = total + output.loss_cka
    """

    loss_tsc: Tensor
    loss_lm: Tensor
    loss_wti: Tensor
    loss_cka: Tensor | None = None

    def compute_weighted_loss(
        self,
        tsc_weight: float = 1.0,
        lm_weight: float = 1.0,
        wti_weight: float = 1.0,
        cka_weight: float = 1.0,
    ) -> Tensor:
        """
        Compute weighted sum of all losses.

        Args:
            tsc_weight: Weight for text-SMILES contrastive loss.
            lm_weight: Weight for language modeling loss.
            wti_weight: Weight for word-token interaction loss.
            cka_weight: Weight for CKA loss (ignored if loss_cka is None).

        Returns:
            Weighted total loss tensor.
        """
        total = (
            tsc_weight * self.loss_tsc
            + lm_weight * self.loss_lm
            + wti_weight * self.loss_wti
        )

        if self.loss_cka is not None:
            total = total + cka_weight * self.loss_cka

        return total

    def to_dict(self) -> dict[str, Tensor | None]:
        """Convert to dictionary for logging."""
        return {
            "loss_tsc": self.loss_tsc,
            "loss_lm": self.loss_lm,
            "loss_wti": self.loss_wti,
            "loss_cka": self.loss_cka,
        }
