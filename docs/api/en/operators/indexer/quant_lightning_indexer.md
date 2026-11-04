# quant_lightning_indexer

Registers the A5 / arch35 cannbotdsl backend; auto selects the same implementation. Target: DSL 0.8.0 + CANN 9.2.0. Device compilation, accuracy and graph replay await A5 validation.

```python
from sgl_kernel_npu.operators.indexer import quant_lightning_indexer

def quant_lightning_indexer(q, k, w, descale_q, descale_k, cu_seqlens_q=None, cu_seqlens_k=None, seqused_q=None, seqused_k=None, cmp_residual_k=None, block_table=None, output_idx_offset=None, metadata=None, *, topk, quant_mode, max_seqlen_q=-1, mask_mode=0, cmp_ratio=1, layout_q='TND', layout_k='TND', return_value=False, candidate_topk_blocks=-1, candidate_block_size=-1, impl='auto'):
    ...
```

MXFP4 QLI uses 32 Q heads, one K head and logical dimension 128. cu_seqlens_q supplies TND Q boundaries; block_table maps paged K. Returns a four-tuple of TopK indices, values, candidate block indices and candidate lengths. Disabled value/candidate outputs are empty tensors. Metadata must match lengths, mask, compression ratio and candidate configuration.

Argument defaults and returned objects preserve upstream behavior. The added keyword-only impl accepts "auto" or "cannbotdsl". See the pinned [implementation](https://gitcode.com/cann/cannbot-dsl/blob/52d0d8bdd852112999f8fb5c873095830302fdd7/samples/attention/quant_lightning_indexer/quant_lightning_indexer.py).

[Build and validation / 构建与验收](../../../../developer_guide/cannbotdsl.md)
