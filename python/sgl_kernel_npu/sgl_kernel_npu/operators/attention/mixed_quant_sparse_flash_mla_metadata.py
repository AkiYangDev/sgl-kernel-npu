"""Public A5 CANNBot-DSL entry for mixed_quant_sparse_flash_mla_metadata."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl(
    "attention.mixed_quant_sparse_flash_mla_metadata",
    "auto",
    supported_generations={Generation.ARCH35},
)
@register_impl(
    "attention.mixed_quant_sparse_flash_mla_metadata",
    "cannbotdsl",
    supported_generations={Generation.ARCH35},
)
def _cannbot_impl(*args, **kwargs):
    from ...implementation.cannbotdsl._runtime import load_operator

    return load_operator("mixed_quant_sparse_flash_mla_metadata")(*args, **kwargs)


@register_operator("attention.mixed_quant_sparse_flash_mla_metadata")
def mixed_quant_sparse_flash_mla_metadata(
    ori_topk_length,
    cmp_topk_length,
    *,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_ori_kv=None,
    max_seqlen_cmp_kv=None,
    num_heads_q,
    num_heads_kv,
    head_dim,
    quant_mode,
    layout_q="TND",
    layout_kv="PA_BBND",
    has_ori_kv=True,
    has_cmp_kv=True,
    impl="auto",
):
    """Use cannbot-dsl 0.8.0 on A5; auto selects cannbotdsl.

    Preserves the upstream arguments, outputs and in-place writes. See the
    bilingual API documentation for layouts and the CANN 9.2.0 validation scope.
    """
    return get_impl_by_name("attention.mixed_quant_sparse_flash_mla_metadata", impl)(
        ori_topk_length=ori_topk_length,
        cmp_topk_length=cmp_topk_length,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_ori_kv=max_seqlen_ori_kv,
        max_seqlen_cmp_kv=max_seqlen_cmp_kv,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        quant_mode=quant_mode,
        layout_q=layout_q,
        layout_kv=layout_kv,
        has_ori_kv=has_ori_kv,
        has_cmp_kv=has_cmp_kv,
    )
