# DSv4.1 DSL correctness tests

The four operator suites and MLA paramset originate from
cann/cannbot-dsl commit `52d0d8bdd852112999f8fb5c873095830302fdd7`,
under `test/attention/<operator>/`. The accompanying CANN license applies.
Imports are adapted to the public sgl-kernel-npu API, the K signature check
includes `impl`, and unused vendor-internal imports are removed. Public API
contract tests also require NPU registration. The local fixture keeps golden
allocation on CPU and honors the selected NPU; MLA no longer forces device 0.
Independent CPU references are preserved. `test_metadata_graph.py` is an
integration test.

See the [validation guide](../../../../docs/developer_guide/cannbotdsl.md).
Device cases have not been run on A5 / CANN 9.2.0.
