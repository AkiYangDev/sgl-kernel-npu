# mixed_quant_sparse_flash_mla

仅注册 A5 / arch35 后端 `cannbotdsl`，`auto` 选择同一实现。目标为 DSL 0.8.0 + CANN 9.2.0；设备编译、精度与图重放尚待 A5 验证。

```python
from sgl_kernel_npu.operators.attention import mixed_quant_sparse_flash_mla

def mixed_quant_sparse_flash_mla(q, *, ori_kv=None, cmp_kv=None, ori_sparse_indices=None, cmp_sparse_indices=None, ori_block_table=None, cmp_block_table=None, cu_seqlens_q=None, seqused_q=None, seqused_ori_kv=None, seqused_cmp_kv=None, ori_topk_length=None, cmp_topk_length=None, sinks=None, metadata=None, quant_mode, softmax_scale=None, layout_q='TND', layout_kv='PA_BBND', return_softmax_lse=False, out, lse, impl='auto'):
    ...
```

混合原始与压缩 KV 的稀疏 MLA，当前固定实现使用 64 个 Q heads、1 个 KV head、head_dim=512。调用方分配并传入 out 与 lse；函数原地写入输出，返回 None。return_softmax_lse 控制 lse 是否参与计算。metadata 与两类 topk length、布局和当前 Q 边界必须一致。

所有参数默认值和返回对象保持上游行为，新增关键字参数 `impl="auto"`，也可显式选 `"cannbotdsl"`。完整参数校验与支持范围见固定版本的[实现源码](https://gitcode.com/cann/cannbot-dsl/blob/52d0d8bdd852112999f8fb5c873095830302fdd7/samples/attention/mixed_quant_sparse_flash_mla/mixed_quant_sparse_flash_mla.py).

[Build and validation / 构建与验收](../../../../developer_guide/cannbotdsl.zh.md)
