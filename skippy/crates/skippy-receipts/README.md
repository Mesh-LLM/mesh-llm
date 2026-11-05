# skippy-receipts

Verifiable inference receipts for Skippy, as designed in
[RECEIPTS.md](../../../mesh/docs/design/RECEIPTS.md).

The crate owns:

- the per-layer tensors a stage reports, keyed by layer, kind and index,
  with no model-specific names (`layer_tensors`);
- the wire shapes of a receipt, a challenge, and an opening (`wire`);
- leaf hashing and the BLAKE3 Merkle tree over a stage's turn (`commitment`);
- the node-side recorder that hashes each ubatch's leaves, keeps a replay log
  per sequence, commits a turn, opens challenged leaves by replay, and keeps
  the openings it made for repeated challenges (`recorder`);
- the client side: drawing a challenge of several layers (`challenge`),
  binding a receipt to the claimed model's layers and an opening to the
  receipt: Merkle paths, the input of layer 0 with its token, and stage
  boundaries (`binding`), reading rows of the claimed GGUF (`weights`), from a sparse copy of a
  remote one that fetches by range (`sparse_copy`), and the bytes the glue
  check needs first (`claimed_reads`),
  checking matmul rows and the per-role fail rule (`audit`), and the glue
  check (`glue`). The glue check computes each challenged layer on the
  opened values through a `GlueEvaluator`, and the input of layer 0 from the
  tokens, and compares the results with the opening
  ([RECEIPTS_GLUE.md](../../../mesh/docs/design/RECEIPTS_GLUE.md)),
  and the output head check (`head`): the sampled tokens against the
  logits the final stage opened, which the client's secret head key
  checks (`head::key`), or against every logit of the output matrix.

It does not own the native recorder, replay or glue evaluation, serving
routes, or the CLI. Those live in `skippy-runtime`, `skippy-serving`,
`skippy-commands` and `skippy-cli`. The crate has no native dependency: the
`GlueEvaluator` over libllama is in `skippy-cli`.

Its consumers are `skippy-serving`, which records and commits on each stage,
and `skippy-commands`, which runs `skippy audit-receipt`. To serve and audit
receipts, see [skippy/docs/RECEIPTS.md](../../docs/RECEIPTS.md).
