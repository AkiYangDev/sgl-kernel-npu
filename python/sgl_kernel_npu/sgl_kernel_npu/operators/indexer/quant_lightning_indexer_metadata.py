"""Public A5 CANNBot-DSL entry for quant_lightning_indexer_metadata."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl(
    "indexer.quant_lightning_indexer_metadata",
    "auto",
    supported_generations={Generation.ARCH35},
)
@register_impl(
    "indexer.quant_lightning_indexer_metadata",
    "cannbotdsl",
    supported_generations={Generation.ARCH35},
)
def _cannbot_impl(*args, **kwargs):
    from ...implementation.cannbotdsl._runtime import load_operator

    return load_operator("quant_lightning_indexer_metadata")(*args, **kwargs)


@register_operator("indexer.quant_lightning_indexer_metadata")
def quant_lightning_indexer_metadata(
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    *,
    batch_size=None,
    max_seqlen_q=-1,
    max_seqlen_k=-1,
    num_heads_q,
    num_heads_k,
    head_dim,
    topk,
    mask_mode=0,
    cmp_ratio=1,
    layout_q="TND",
    layout_k="TND",
    candidate_topk_blocks=-1,
    candidate_block_size=-1,
    impl="auto",
):
    """Use cannbot-dsl 0.8.0 on A5; auto selects cannbotdsl.

    Preserves the upstream arguments, outputs and in-place writes. See the
    bilingual API documentation for layouts and the CANN 9.2.0 validation scope.
    """
    return get_impl_by_name("indexer.quant_lightning_indexer_metadata", impl)(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        head_dim=head_dim,
        topk=topk,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        layout_q=layout_q,
        layout_k=layout_k,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
    )
