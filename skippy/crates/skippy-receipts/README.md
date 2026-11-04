# skippy-receipts

Verifiable inference receipts for Skippy. A node that serves a turn signs a
commitment to it. A client then audits which model ran. The design is in
[RECEIPTS.md](../../../mesh/docs/design/RECEIPTS.md) and
[RECEIPTS_GLUE.md](../../../mesh/docs/design/RECEIPTS_GLUE.md). The feature
is experimental. Its open limitations are in
[Open limitations](../../../mesh/docs/design/RECEIPTS.md#open-limitations).

## Layers

The crate has three layers. Every top-level module is in one of them.

| Layer | What it holds | What it may use |
|---|---|---|
| Shared | The format of a receipt, and the arithmetic that a node and a client must do the same way | Shared modules |
| Node | What a serving stage runs: record, commit, open by replay | Shared modules |
| Client | What an auditor runs: the checks of an opening, and the reader of the claimed GGUF | Shared and client modules |

A client module never uses `recorder`, and `recorder` never uses a client
module. Test code may use all three layers.

[`src/lib.rs`](src/lib.rs) lists the modules of each layer under a
`// Layer:` comment. [`tests/layers.rs`](tests/layers.rs) reads those
comments and every file under `src/`. It fails when a module has a
`crate::` path into a layer that it may not use. To add a module, put its
`mod` line under the comment of its layer.

### Shared

| Module | Responsibility |
|---|---|
| `wire` | The JSON shapes of a receipt, a challenge, an opening and a "cannot open" answer. The receipt version |
| `layer_tensors` | The tensors that a stage reports per layer, keyed by layer, kind and index, with no model-specific names |
| `commitment` | Leaf hashes, the digest of the leaves of a position, the leaf of a prefix cell, the digest of the prefix tokens, and the BLAKE3 Merkle tree over the turn of a stage |
| `turn` | The signed turn root, the signatures of an opening and of attention rows, and the VRF inputs |
| `vrf` | ECVRF-EDWARDS25519-SHA512-TAI (RFC 9381) |
| `draw` | Deterministic draws from a VRF output |
| `challenge` | The draw of a challenge, and the check that a challenge is one that a draw makes |
| `attention` | The layout of a layer's attention, the positions and blocks that a query attends to, and the partial of a block. `attention::opening` is what the two sides send for the attention check |
| `ggml_types` | The ids, names and block sizes of the ggml tensor types |
| `dequant` | Reference dequantization of ggml rows in f64 |
| `cache_types` | The types that a stored KV row can be decoded in, the types that an audit covers (f32 and f16, and no quantized cache), and the decoder of such a row |
| `activation_rounding` | How each kernel rounds f32 values before it multiplies them |
| `node_fault` | The mark on an error that a defective value of the node causes. The shared decoders set it. Only a client reads it |

The node computes attention partials from its own cache rows, and the
client computes them again from the rows that the node opens. Both must
decode and round the values the same way. That is why `dequant`,
`cache_types` and `activation_rounding` are shared.

### Node

| Module | Responsibility |
|---|---|
| `recorder` | Hashes the leaves of each ubatch, keeps a replay log per sequence, commits a turn, opens challenged leaves by replay, pins cells, and keeps the openings that it made. For every KV cell it keeps the leaves of the position and the ubatch that wrote it, so it can open a position of a cached prefix |

### Client

| Module | Responsibility |
|---|---|
| `coverage` | Decides from a receipt alone whether this verifier can audit it |
| `binding` | Checks a receipt against the claimed model and the request, and an opening against the receipt: its form, its Merkle paths, the stage boundaries, and the leaves and tokens of a cached prefix |
| `opened_values` | Checks every opened value against the shapes of the claimed model before a check reads it |
| `error_bound` | Compares an error with a tolerance so that a value that is not a number fails |
| `weights` | Reads the header and the rows of the claimed GGUF, with limits on every count and size |
| `shards` | Reads a split GGUF as one file |
| `sparse_copy` | A sparse copy of a remote GGUF that fetches by range |
| `claimed_reads` | The bytes of the claimed GGUF that the glue check reads first |
| `audit` | The row checks, the rounding profiles that each backend may use per weight type, the near-tie repair and the fail rule per weight role |
| `glue` | The glue check. It computes each challenged layer on the opened values through a `GlueEvaluator`, and the input of layer 0 from the tokens |
| `attention_check` | The client side of the attention check: the combination of the partials, the blocks that the client picks, the proofs of the opened rows, and the tolerances |
| `head` | The output head check: the opened logits against the claimed output matrix, with the secret head key (`head::key`) or on every row (`head::every_row`), and the sampled tokens on those logits |
| `head::draw` | The replay of the draw of a sampled turn: the uniform of a seed and a position, and the interval of uniforms that pick the sampled token |
| `head::reasoning_budget` | The replay of the reasoning-budget sampler of a turn, from the tags its statement names: which tokens the sampler forced. A port of `common/reasoning-budget.cpp` of the pinned llama.cpp |
| `sampler_request` | Compares the sampler statement of a receipt with the request of the client |
| `recurrent` | The client side of the recurrent check |
| `evidence` | The evidence of an audit, and the check that a file is evidence |

### Test support

`toy_turn` is a toy model with one honest turn. It is for the tests of this
crate and of the crates that audit receipts, behind the `test-support`
feature. It uses all three layers.

## What the crate does not own

The crate has no native dependency.

- The native recorder, the replay and the glue evaluation are in the
  llama.cpp patch queue and in `skippy-runtime`.
- The `GlueEvaluator` over libllama, the claimed-file cache, the head key
  file and the audit of one turn are in `skippy-receipt-audit`.
- The `audit-receipt` and `recheck-receipt` commands are in
  `skippy-commands`.
- The serving routes, the receipt retention and the stage connection are
  in `skippy-serving`.
- The CLI flags are in `skippy-cli`.

## Consumers

| Crate | Layers | What it does |
|---|---|---|
| `skippy-serving` | Shared and node | Records and commits on each stage, and opens challenges. Its tests also use `binding` to check what a stage opened |
| `skippy-receipt-audit` | Shared and client | Audits one received turn against a claimed GGUF, and rechecks evidence |
| `skippy-commands` | Shared and client | Runs `skippy audit-receipt` and `skippy recheck-receipt` through `skippy-receipt-audit` |
| `mesh-llm-host-runtime` | Shared and client | Audits the turns that the peers of a Mesh node serve, through `skippy-receipt-audit` |

To serve and audit receipts, see
[skippy/docs/RECEIPTS.md](../../docs/RECEIPTS.md).
