# indexer_prologue_k

Registers the A5 / arch35 cannbotdsl backend; auto selects the same implementation. Target: DSL 0.8.0 + CANN 9.2.0. Device compilation, accuracy and graph replay await A5 validation.

```python
from sgl_kernel_npu.operators.indexer import indexer_prologue_k

def indexer_prologue_k(latent, wk, norm_weight, rope_sin, rope_cos, k_cache, k_scale_cache=None, *, cache_index, storage_mode, norm_eps, combined_block_size=-1, impl='auto'):
    ...
```

Projects, normalizes, rotates and quantizes latent rows into the K cache. Returns the same k_cache tensor; cache_index=-1 skips a token. storage_mode=0 separates MXFP4 data and scales; storage_mode=1 combines records using combined_block_size. The caller supplies cache capacity, strides and slot mapping.

Argument defaults and returned objects preserve upstream behavior. The added keyword-only impl accepts "auto" or "cannbotdsl". See the pinned [implementation](https://gitcode.com/cann/cannbot-dsl/blob/52d0d8bdd852112999f8fb5c873095830302fdd7/samples/attention/indexer_prologue_k/indexer_prologue_k.py).

[Build and validation / 构建与验收](../../../../developer_guide/cannbotdsl.md)
