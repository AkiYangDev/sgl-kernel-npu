# quant_sparse_lightning_indexer

Registers the A5 / arch35 cannbotdsl backend; auto selects the same implementation. Target: DSL 0.8.0 + CANN 9.2.0. Device compilation, accuracy and graph replay await A5 validation.

```python
from sgl_kernel_npu.operators.indexer import quant_sparse_lightning_indexer

def quant_sparse_lightning_indexer(q, k, w, descale_q, candidate_block_indices, candidate_block_length, cu_seqlens_q=None, cu_seqlens_k=None, seqused_q=None, seqused_k=None, cmp_residual_k=None, block_table=None, output_idx_offset=None, metadata=None, *, descale_k=None, topk, candidate_block_size, quant_mode, max_seqlen_q=-1, mask_mode=0, cmp_ratio=1, layout_q='TND', layout_k='TND', return_value=False, impl='auto'):
    ...
```

QSLI consumes candidate blocks from QLI. candidate_block_length[t,0] specifies the valid prefix, from 0 to 2048. Each prefix entry is a valid logical eight-token block in the row’s batch; the suffix is ignored. The caller guarantees these value constraints without a host copy. Returns (indices, values); values is empty when return_value=False. Metadata must match candidate lengths and computation settings.

Argument defaults and returned objects preserve upstream behavior. The added keyword-only impl accepts "auto" or "cannbotdsl". See the pinned [implementation](https://gitcode.com/cann/cannbot-dsl/blob/52d0d8bdd852112999f8fb5c873095830302fdd7/samples/attention/quant_sparse_lightning_indexer/quant_sparse_lightning_indexer.py).

[Build and validation / 构建与验收](../../../../developer_guide/cannbotdsl.md)
