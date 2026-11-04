"""Public A5 CANNBot-DSL entry for indexer_prologue_k."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl(
    "indexer.indexer_prologue_k", "auto", supported_generations={Generation.ARCH35}
)
@register_impl(
    "indexer.indexer_prologue_k",
    "cannbotdsl",
    supported_generations={Generation.ARCH35},
)
def _cannbot_impl(*args, **kwargs):
    from ...implementation.cannbotdsl._runtime import load_operator

    return load_operator("indexer_prologue_k")(*args, **kwargs)


@register_operator("indexer.indexer_prologue_k")
def indexer_prologue_k(
    latent,
    wk,
    norm_weight,
    rope_sin,
    rope_cos,
    k_cache,
    k_scale_cache=None,
    *,
    cache_index,
    storage_mode,
    norm_eps,
    combined_block_size=-1,
    impl="auto",
):
    """Use cannbot-dsl 0.8.0 on A5; auto selects cannbotdsl.

    Preserves the upstream arguments, outputs and in-place writes. See the
    bilingual API documentation for layouts and the CANN 9.2.0 validation scope.
    """
    return get_impl_by_name("indexer.indexer_prologue_k", impl)(
        latent=latent,
        wk=wk,
        norm_weight=norm_weight,
        rope_sin=rope_sin,
        rope_cos=rope_cos,
        k_cache=k_cache,
        k_scale_cache=k_scale_cache,
        cache_index=cache_index,
        storage_mode=storage_mode,
        norm_eps=norm_eps,
        combined_block_size=combined_block_size,
    )
