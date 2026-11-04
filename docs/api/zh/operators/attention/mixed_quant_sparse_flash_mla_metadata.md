# mixed_quant_sparse_flash_mla_metadata

仅注册 A5 / arch35 后端 `cannbotdsl`，`auto` 选择同一实现。目标为 DSL 0.8.0 + CANN 9.2.0；设备编译、精度与图重放尚待 A5 验证。

```python
from sgl_kernel_npu.operators.attention import mixed_quant_sparse_flash_mla_metadata

def mixed_quant_sparse_flash_mla_metadata(ori_topk_length, cmp_topk_length, *, cu_seqlens_q=None, seqused_q=None, seqused_ori_kv=None, seqused_cmp_kv=None, batch_size=None, max_seqlen_q=None, max_seqlen_ori_kv=None, max_seqlen_cmp_kv=None, num_heads_q, num_heads_kv, head_dim, quant_mode, layout_q='TND', layout_kv='PA_BBND', has_ori_kv=True, has_cmp_kv=True, impl='auto'):
    ...
```

在 AICPU 上生成供 [mixed_quant_sparse_flash_mla](mixed_quant_sparse_flash_mla.md) 使用的 int32 调度 Tensor。ori_topk_length 与 cmp_topk_length 为同设备、连续的 [T,1] int32 Tensor；即使没有压缩 KV，两者仍须传入。固定组合为 num_heads_q=64、num_heads_kv=1、head_dim=512、quant_mode=1，布局 TND / PA_BBND，has_ori_kv=True。cu_seqlens_q 是 [B+1] 连续 int32 累积边界。长度变化后更新 metadata，再交给计算接口。

所有参数默认值和返回对象保持上游行为，新增关键字参数 `impl="auto"`，也可显式选 `"cannbotdsl"`。完整参数校验与支持范围见固定版本的[实现源码](https://gitcode.com/cann/cannbot-dsl/blob/52d0d8bdd852112999f8fb5c873095830302fdd7/samples/attention/mixed_quant_sparse_flash_mla/mixed_quant_sparse_flash_mla_metadata.py).

[Build and validation / 构建与验收](../../../../developer_guide/cannbotdsl.zh.md)
