# quant_sparse_lightning_indexer

仅注册 A5 / arch35 后端 `cannbotdsl`，`auto` 选择同一实现。目标为 DSL 0.8.0 + CANN 9.2.0；设备编译、精度与图重放尚待 A5 验证。

```python
from sgl_kernel_npu.operators.indexer import quant_sparse_lightning_indexer

def quant_sparse_lightning_indexer(q, k, w, descale_q, candidate_block_indices, candidate_block_length, cu_seqlens_q=None, cu_seqlens_k=None, seqused_q=None, seqused_k=None, cmp_residual_k=None, block_table=None, output_idx_offset=None, metadata=None, *, descale_k=None, topk, candidate_block_size, quant_mode, max_seqlen_q=-1, mask_mode=0, cmp_ratio=1, layout_q='TND', layout_k='TND', return_value=False, impl='auto'):
    ...
```

QSLI 消费 QLI 产生的候选块。candidate_block_length[t,0] 是有效前缀长度，取值为 0 到 2048；前缀中的每个索引是当前 batch 内有效的逻辑八 token 块号，后缀忽略。调用方保证值域，不将设备张量复制到 CPU 检查。返回 (indices, values)；return_value=False 时 values 为空 Tensor。metadata 与候选长度及计算配置必须一致。

所有参数默认值和返回对象保持上游行为，新增关键字参数 `impl="auto"`，也可显式选 `"cannbotdsl"`。完整参数校验与支持范围见固定版本的[实现源码](https://gitcode.com/cann/cannbot-dsl/blob/52d0d8bdd852112999f8fb5c873095830302fdd7/samples/attention/quant_sparse_lightning_indexer/quant_sparse_lightning_indexer.py).

[Build and validation / 构建与验收](../../../../developer_guide/cannbotdsl.zh.md)
