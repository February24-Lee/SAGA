import logging

import torch
import transformers

logger = logging.getLogger(__name__)


def install_clip_expand_mask_shim():
    try:
        from transformers.models.clip import modeling_clip as clip_mod
    except Exception:
        return

    if hasattr(clip_mod, "_expand_mask"):
        return

    _prep4d = getattr(transfomers := transformers, "_prepare_4d_attention_mask", None)
    if _prep4d is None:
        try:
            from transformers.modeling_attn_mask_utils import (
                _prepare_4d_attention_mask as _prep4d,
            )
        except Exception:
            _prep4d = None

    def _expand_mask(
        attention_mask: torch.Tensor, dtype: torch.dtype, tgt_len: int | None = None
    ) -> torch.Tensor:
        """
        Drop-in for historical CLIP._expand_mask.
        Returns (batch, 1, tgt_len, src_len) with masked positions = large negative.
        """
        if attention_mask is None:
            return None

        if attention_mask.dim() == 4:
            return attention_mask.to(dtype)

        if _prep4d is not None:
            return _prep4d(attention_mask, dtype, tgt_len)

        mask = attention_mask
        if mask.dim() != 2:
            m = mask
            while m.dim() > 2:
                m = m.squeeze(1)
            mask = m
        bsz, src_len = mask.shape
        tgt_len = tgt_len if tgt_len is not None else src_len

        # (B,1,1,S) -> (B,1,T,S)
        expanded = mask[:, None, None, :].to(dtype)
        expanded = (1.0 - expanded) * torch.finfo(dtype).min  # 1->0, 0->-inf_like
        expanded = expanded.expand(bsz, 1, tgt_len, src_len)
        return expanded

    clip_mod._expand_mask = _expand_mask


def transformers_import_monkey_patch():
    """from transformers.modeling_utils import (
        PreTrainedModel, => transformers.PreTrainedModel
        apply_chunking_to_forward, => transformers.apply_chunking_to_forward
        find_pruneable_heads_and_indices, => transformers.pytorch_utils.find_pruneable_heads_and_indices
        prune_linear_layer, => transformers.pytorch_utils.prune_linear_layer
    )"""
    import sys

    import transformers
    import transformers.modeling_utils
    import transformers.pytorch_utils
    from diffusers.models import attention
    from diffusers.models.attention import Attention

    sys.modules["transformers.modeling_utils"] = transformers.modeling_utils

    sys.modules["transformers.modeling_utils.apply_chunking_to_forward"] = (
        transformers.apply_chunking_to_forward
    )
    sys.modules["transformers.modeling_utils.find_pruneable_heads_and_indices"] = (
        transformers.pytorch_utils.find_pruneable_heads_and_indices
    )
    sys.modules["transformers.modeling_utils.prune_linear_layer"] = (
        transformers.pytorch_utils.prune_linear_layer
    )
    sys.modules["diffusers.models.cross_attention"] = attention
    sys.modules["diffusers.models.attention.CrossAttention"] = Attention

    install_clip_expand_mask_shim()
    logger.info("Monkey patch applied successfully!")
    return
