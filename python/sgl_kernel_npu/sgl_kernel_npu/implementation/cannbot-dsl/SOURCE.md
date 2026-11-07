# DeepSeek V4.1 CANNBot-DSL 0.7.1 source provenance

This branch carries the seven upstream sample implementations at
`0cca3f5313ff7c09e3bcd1453e82388b2f612872`, before their migration to the 0.8 buffer API.
It is an alternative source snapshot, not a compatibility shim around the
newer 0.8 implementations. All seven public argument signatures are unchanged.
The two source revisions are not claimed to be numerically or performance equivalent.

[UPSTREAM.json](UPSTREAM.json) records original paths, original SHA-256 values
(`upstream_sha256`), packaged SHA-256 values (`sha256`), and local backports.
Six files are byte-identical to this upstream tree. `indexer_prologue_k.py`
backports the projection-tail allocation fix from upstream commit
`31bbd706a4e6a72b278da4b70718e60286ccb9e7`: the allocation is rounded to 16 rows and only the
logical T-row view is passed to the pipeline. No kernel math is changed by
that backport.

Target validation environment: CANN 9.2.0, Ascend 950 (3510), DSL 0.7.1.
Device precision, graph replay and model validation remain required.
These files use the CANN Open Software License Agreement Version 2.0; see
[LICENSE.cannbot-dsl](LICENSE.cannbot-dsl).
