# quant_lightning_indexer

仅注册 A5 / arch35 后端 `cannbotdsl`，`auto` 选择同一实现。目标为 DSL 0.8.0 + CANN 9.2.0；设备编译、精度与图重放尚待 A5 验证。

```python
from sgl_kernel_npu.operators.indexer import quant_lightning_indexer

def quant_lightning_indexer(q, k, w, descale_q, descale_k, cu_seqlens_q=None, cu_seqlens_k=None, seqused_q=None, seqused_k=None, cmp_residual_k=None, block_table=None, output_idx_offset=None, metadata=None, *, topk, quant_mode, max_seqlen_q=-1, mask_mode=0, cmp_ratio=1, layout_q='TND', layout_k='TND', return_value=False, candidate_topk_blocks=-1, candidate_block_size=-1, impl='auto'):
    ...
```

MXFP4 QLI 使用 32 个 Q heads、1 个 K head、逻辑维度 128。TND 的 Q 边界由 cu_seqlens_q 给出；分页 K 的逻辑页由 block_table 映射。返回四元组：TopK indices、values、candidate block indices、candidate lengths。关闭 return_value 或候选输出时，相应结果为空 Tensor。metadata 必须与当前长度、mask、压缩比例和候选配置一致。

所有参数默认值和返回对象保持上游行为，新增关键字参数 `impl="auto"`，也可显式选 `"cannbotdsl"`。完整参数校验与支持范围见固定版本的[实现源码](https://gitcode.com/cann/cannbot-dsl/blob/52d0d8bdd852112999f8fb5c873095830302fdd7/samples/attention/quant_lightning_indexer/quant_lightning_indexer.py).

[Build and validation / 构建与验收](../../../../developer_guide/cannbotdsl.zh.md)
