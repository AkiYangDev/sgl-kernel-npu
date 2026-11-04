# mixed_quant_sparse_flash_mla_metadata

Registers the A5 / arch35 cannbotdsl backend; auto selects the same implementation. Target: DSL 0.8.0 + CANN 9.2.0. Device compilation, accuracy and graph replay await A5 validation.

```python
from sgl_kernel_npu.operators.attention import mixed_quant_sparse_flash_mla_metadata

def mixed_quant_sparse_flash_mla_metadata(ori_topk_length, cmp_topk_length, *, cu_seqlens_q=None, seqused_q=None, seqused_ori_kv=None, seqused_cmp_kv=None, batch_size=None, max_seqlen_q=None, max_seqlen_ori_kv=None, max_seqlen_cmp_kv=None, num_heads_q, num_heads_kv, head_dim, quant_mode, layout_q='TND', layout_kv='PA_BBND', has_ori_kv=True, has_cmp_kv=True, impl='auto'):
    ...
```

Generates an int32 AICPU scheduling tensor for [mixed_quant_sparse_flash_mla](mixed_quant_sparse_flash_mla.md). ori_topk_length and cmp_topk_length are contiguous [T,1] int32 tensors on the same device, both required even without compressed KV. The fixed configuration is 64 Q heads, one KV head, head_dim=512, quant_mode=1, TND / PA_BBND and has_ori_kv=True. cu_seqlens_q contains contiguous int32 [B+1] cumulative boundaries. Refresh metadata when lengths change.

Argument defaults and returned objects preserve upstream behavior. The added keyword-only impl accepts "auto" or "cannbotdsl". See the pinned [implementation](https://gitcode.com/cann/cannbot-dsl/blob/52d0d8bdd852112999f8fb5c873095830302fdd7/samples/attention/mixed_quant_sparse_flash_mla/mixed_quant_sparse_flash_mla_metadata.py).

[Build and validation / 构建与验收](../../../../developer_guide/cannbotdsl.md)
