# quant_lightning_indexer_metadata

Registers the A5 / arch35 cannbotdsl backend; auto selects the same implementation. Target: DSL 0.8.0 + CANN 9.2.0. Device compilation, accuracy and graph replay await A5 validation.

```python
from sgl_kernel_npu.operators.indexer import quant_lightning_indexer_metadata

def quant_lightning_indexer_metadata(cu_seqlens_q=None, cu_seqlens_k=None, seqused_q=None, seqused_k=None, cmp_residual_k=None, *, batch_size=None, max_seqlen_q=-1, max_seqlen_k=-1, num_heads_q, num_heads_k, head_dim, topk, mask_mode=0, cmp_ratio=1, layout_q='TND', layout_k='TND', candidate_topk_blocks=-1, candidate_block_size=-1, impl='auto'):
    ...
```

Generates a device int32 scheduling tensor on AICPU for [quant_lightning_indexer](quant_lightning_indexer.md). Pass the result directly to the computation. Match heads, dimensions, TopK, mask, compression ratio, layouts and sequence boundaries. Regenerate after length changes, or capture metadata computation and update input lengths in place. TND K infers batch from cumulative boundaries and requires batch_size=None.

Argument defaults and returned objects preserve upstream behavior. The added keyword-only impl accepts "auto" or "cannbotdsl". See the pinned [implementation](https://gitcode.com/cann/cannbot-dsl/blob/52d0d8bdd852112999f8fb5c873095830302fdd7/samples/attention/quant_lightning_indexer_metadata/quant_lightning_indexer_metadata.py).

[Build and validation / 构建与验收](../../../../developer_guide/cannbotdsl.md)
