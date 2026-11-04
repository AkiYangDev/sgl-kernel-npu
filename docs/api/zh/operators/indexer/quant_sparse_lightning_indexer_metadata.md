# quant_sparse_lightning_indexer_metadata

仅注册 A5 / arch35 后端 `cannbotdsl`，`auto` 选择同一实现。目标为 DSL 0.8.0 + CANN 9.2.0；设备编译、精度与图重放尚待 A5 验证。

```python
from sgl_kernel_npu.operators.indexer import quant_sparse_lightning_indexer_metadata

def quant_sparse_lightning_indexer_metadata(candidate_block_length, cu_seqlens_q=None, cu_seqlens_k=None, seqused_q=None, seqused_k=None, cmp_residual_k=None, *, batch_size=None, max_seqlen_q=-1, max_seqlen_k=-1, num_heads_q, num_heads_k, head_dim, topk, quant_mode, candidate_block_size, mask_mode=0, cmp_ratio=1, layout_q='TND', layout_k='TND', impl='auto'):
    ...
```

在 AICPU 上生成供 [quant_sparse_lightning_indexer](quant_sparse_lightning_indexer.md) 使用的设备端 int32 调度 Tensor，返回值直接传给计算接口。调用方必须对齐 head 数、维度、TopK、mask、压缩比例、布局及序列边界。长度改变后应重新生成，或在图内执行 metadata 并原地更新输入长度。TND K 的 batch 从累积边界推导，batch_size 必须为 None。

所有参数默认值和返回对象保持上游行为，新增关键字参数 `impl="auto"`，也可显式选 `"cannbotdsl"`。完整参数校验与支持范围见固定版本的[实现源码](https://gitcode.com/cann/cannbot-dsl/blob/52d0d8bdd852112999f8fb5c873095830302fdd7/samples/attention/quant_sparse_lightning_indexer_metadata/quant_sparse_lightning_indexer_metadata.py).

[Build and validation / 构建与验收](../../../../developer_guide/cannbotdsl.zh.md)
