"""Public A5 CANNBot-DSL entry for quant_sparse_lightning_indexer."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl(
    "indexer.quant_sparse_lightning_indexer",
    "auto",
    supported_generations={Generation.ARCH35},
)
@register_impl(
    "indexer.quant_sparse_lightning_indexer",
    "cannbotdsl",
    supported_generations={Generation.ARCH35},
)
def _cannbot_impl(*args, **kwargs):
    from ...implementation.cannbotdsl._runtime import load_operator

    return load_operator("quant_sparse_lightning_indexer")(*args, **kwargs)


@register_operator("indexer.quant_sparse_lightning_indexer")
def quant_sparse_lightning_indexer(
    q,
    k,
    w,
    descale_q,
    candidate_block_indices,
    candidate_block_length,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    output_idx_offset=None,
    metadata=None,
    *,
    descale_k=None,
    topk,
    candidate_block_size,
    quant_mode,
    max_seqlen_q=-1,
    mask_mode=0,
    cmp_ratio=1,
    layout_q="TND",
    layout_k="TND",
    return_value=False,
    impl="auto",
):
    """Use cannbot-dsl 0.8.0 on A5; auto selects cannbotdsl.

    Preserves the upstream arguments, outputs and in-place writes. See the
    bilingual API documentation for layouts and the CANN 9.2.0 validation scope.
    """
    return get_impl_by_name("indexer.quant_sparse_lightning_indexer", impl)(
        q=q,
        k=k,
        w=w,
        descale_q=descale_q,
        candidate_block_indices=candidate_block_indices,
        candidate_block_length=candidate_block_length,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        block_table=block_table,
        output_idx_offset=output_idx_offset,
        metadata=metadata,
        descale_k=descale_k,
        topk=topk,
        candidate_block_size=candidate_block_size,
        quant_mode=quant_mode,
        max_seqlen_q=max_seqlen_q,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        layout_q=layout_q,
        layout_k=layout_k,
        return_value=return_value,
    )
