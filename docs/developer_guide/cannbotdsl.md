# DeepSeek-V4.1 CANNBot-DSL operators

The seven attention/indexer entries ship as repository-owned Python source,
like the Triton implementations. `implementation/cannbot-dsl/UPSTREAM.json`
records the source commit, original paths and hashes; the CANN license is
packaged beside it. `auto` and `cannbot_dsl` select the same implementation.

The selected validation target is **A5 / ARCH 3510, CANN 9.2.0 and
cannbot-dsl 0.7.1**. Device numerical and graph-replay validation is still
required before using this alternative in a deployment.

This independent variant pins the seven samples at upstream commit
`0cca3f5313ff7c09e3bcd1453e82388b2f612872`, before their transition to the
0.8 on-chip buffer API. It uses `Channel`, `GmIn` and `GmOut` directly and
does not emulate `RingBuffer` with a global compatibility shim. The runtime
and optional wheel extra select `cannbot-dsl>=0.7.1,<0.8`; this interval is
the branch's compatibility policy, not an upstream promise about all future
0.7 releases. The validated compiler version must be recorded separately.

The seven public signatures are unchanged, so the existing SGLang resolver
can be reused. The earlier sample snapshot also predates later algorithm and
performance changes; numerical and performance equivalence with the 0.8
variant is not established.

The K prologue includes upstream fix `31bbd706a4e6a72b278da4b70718e60286ccb9e7`:
round its projection allocation up to 16 rows and pass only the logical T-row
view to the pipeline. A fixpipe tail tile can otherwise write beyond a
`[T,D]` allocation, including the common decode case T=1. This is a concrete
memory-safety fix, not proof that it resolves the model's observed NaNs.

## Build and use

Build the ordinary kernel wheel using the repository-root build command:

```bash
python3 setup.py --arch-version 3510 --without-third-party -j8
python3 -m pip install --force-reinstall --no-deps output/sgl_kernel_npu-*.whl
python3 -m pip install 'cannbot-dsl==0.7.1'
```

`--without-third-party` isolates the migrated sources. Select the required
third-party profiles when composing a complete deployment wheel. This command
does not imply that all operators needed by SGLang are included.

DSL functions are imported and JIT-compiled on use. Warm the actual shapes
before graph capture. No precompiled DSL bundle is produced by this build.
The optional `cannbot_dsl` wheel extra installs the DSL dependency; unrelated
operators do not require it. Use a matching Linux/Python and torch/torch_npu
stack (the A5 validation environment uses Python 3.12 and torch/torch_npu 2.13.0).

Import the public entries from `sgl_kernel_npu.operators.indexer` (K prologue,
QLI/QSLI and metadata) and `sgl_kernel_npu.operators.attention` (MLA and metadata).
Their signatures match the pinned sources with an extra `impl="auto"` keyword.
MLA writes caller-owned `out`/`lse`; K prologue updates its cache arguments.
Inputs must share an A5 device. The separate AscendC compressor still requires
`cann_ops_transformer` in SGLang.

## Validate on A5

```bash
python3 -m pytest tests/python/sgl_kernel_npu/dsv41_cannbot_dsl --npu-device 0 -q
```

The suites retain independent numerical references. `test_metadata_graph.py`
checks all three metadata operators after lengths change between replays.
The K projection-tail guard covers T=1 and T=17 and checks that the full
fixpipe write stays inside the actual allocation, not just that cache values
match a reference.
Record the installed versions, package hashes and executed test results;
skipped cases do not establish device correctness.
Full compute graph capture, model/PD integration and performance require
separate A5 validation. Offline and fixture-wheel tests do not establish
hardware compatibility or speedup.

Use `--npu-device` to select a visible card; CPU references remain on CPU.
Source provenance and API forwarding can be checked without an NPU:

```bash
python3 -m pytest tests/python/sgl_kernel_npu/test_dsv41_cannbot_sources.py tests/python/sgl_kernel_npu/test_dsv41_cannbot_dsl_api.py -q
```
