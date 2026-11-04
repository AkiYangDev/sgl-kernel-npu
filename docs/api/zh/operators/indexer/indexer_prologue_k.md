# indexer_prologue_k

仅注册 A5 / arch35 后端 `cannbotdsl`，`auto` 选择同一实现。目标为 DSL 0.8.0 + CANN 9.2.0；设备编译、精度与图重放尚待 A5 验证。

```python
from sgl_kernel_npu.operators.indexer import indexer_prologue_k

def indexer_prologue_k(latent, wk, norm_weight, rope_sin, rope_cos, k_cache, k_scale_cache=None, *, cache_index, storage_mode, norm_eps, combined_block_size=-1, impl='auto'):
    ...
```

将 latent 投影、归一化、旋转及量化后写入 K cache。返回同一个 k_cache Tensor；cache_index 为 -1 的 token 跳过。storage_mode=0 将 MXFP4 数据与 scale 分开保存，storage_mode=1 按 combined_block_size 合并记录。缓存容量、stride 与 slot 映射由调用方提供。

所有参数默认值和返回对象保持上游行为，新增关键字参数 `impl="auto"`，也可显式选 `"cannbotdsl"`。完整参数校验与支持范围见固定版本的[实现源码](https://gitcode.com/cann/cannbot-dsl/blob/52d0d8bdd852112999f8fb5c873095830302fdd7/samples/attention/indexer_prologue_k/indexer_prologue_k.py).

[Build and validation / 构建与验收](../../../../developer_guide/cannbotdsl.zh.md)
