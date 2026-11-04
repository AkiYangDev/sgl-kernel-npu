# mixed_quant_sparse_flash_mla

Registers the A5 / arch35 cannbotdsl backend; auto selects the same implementation. Target: DSL 0.8.0 + CANN 9.2.0. Device compilation, accuracy and graph replay await A5 validation.

```python
from sgl_kernel_npu.operators.attention import mixed_quant_sparse_flash_mla

def mixed_quant_sparse_flash_mla(q, *, ori_kv=None, cmp_kv=None, ori_sparse_indices=None, cmp_sparse_indices=None, ori_block_table=None, cmp_block_table=None, cu_seqlens_q=None, seqused_q=None, seqused_ori_kv=None, seqused_cmp_kv=None, ori_topk_length=None, cmp_topk_length=None, sinks=None, metadata=None, quant_mode, softmax_scale=None, layout_q='TND', layout_kv='PA_BBND', return_softmax_lse=False, out, lse, impl='auto'):
    ...
```

Sparse MLA over original and compressed KV, with the pinned implementation using 64 Q heads, one KV head and head_dim=512. The caller allocates out and lse; the function writes outputs in place and returns None. return_softmax_lse controls use of lse. Metadata must match both topk lengths, layouts and current Q boundaries.

Argument defaults and returned objects preserve upstream behavior. The added keyword-only impl accepts "auto" or "cannbotdsl". See the pinned [implementation](https://gitcode.com/cann/cannbot-dsl/blob/52d0d8bdd852112999f8fb5c873095830302fdd7/samples/attention/mixed_quant_sparse_flash_mla/mixed_quant_sparse_flash_mla.py).

[Build and validation / 构建与验收](../../../../developer_guide/cannbotdsl.md)
