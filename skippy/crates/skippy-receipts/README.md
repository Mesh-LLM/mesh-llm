# skippy-receipts

Verifiable inference receipts for Skippy, as designed in
[RECEIPTS.md](../../../mesh/docs/design/RECEIPTS.md).

The crate owns:

- the wire shapes of a receipt, a challenge, and an opening (`wire`);
- leaf hashing and the BLAKE3 Merkle tree over a stage's turn (`commitment`);
- the node-side recorder that hashes each ubatch's leaves, keeps a replay log
  per sequence, commits a turn, and opens challenged leaves by replay
  (`recorder`);
- the client side: drawing a challenge (`challenge`), reading rows and norms of
  the claimed GGUF (`weights`), checking an opening (`audit`), and checking the
  operations that join its matmuls to the residual stream (`glue`).

It does not own the native recorder and replay, serving routes, or the CLI.
Those live in `skippy-runtime`, `skippy-serving`, and `skippy-commands`.

Its consumers are `skippy-serving`, which records and commits on each stage,
and `skippy-commands`, which runs `skippy audit-receipt`. To serve and audit
receipts, see [skippy/docs/RECEIPTS.md](../../docs/RECEIPTS.md).
