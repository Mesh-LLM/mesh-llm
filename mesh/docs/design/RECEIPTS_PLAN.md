# Receipts implementation plan

This plan turns [RECEIPTS.md](RECEIPTS.md) into milestones. Each milestone
leaves `skippy serve` working with receipts off, which stays the default.

## Milestone 0: end-to-end demo (built, then replaced by milestone 1)

Milestone 0 stored every matmul column. Milestone 1 replaced its native tap
and leaf layout; serving, the HTTP API and the auditor carried over.

What it built:

- **Native tap** (ABI 0.1.75). `skippy_model_set_receipt_tap` reported every
  `blk.<layer>.*.weight` matmul of the stage context with its f32 input and
  output columns and each column's sequence and position.
- **`skippy-receipts` crate.** Wire shapes, BLAKE3 leaves and Merkle paths,
  the node-side recorder, challenge drawing, GGUF row reads, and the client
  row checks with the CUDA and CPU Q8 rounding profiles.
- **Serving.** `skippy serve --stage-transport binary --receipts-listen ADDR
  [--receipts-downstream URL] [--receipts-claimed-model NAME]`. At the end of a
  turn, stage 0 commits its leaves, asks the next stage to commit the same
  session and positions, and keeps the receipt under the request's
  `x-request-id`. Each stage serves `GET /receipts/{id}`,
  `POST /receipts/commit` and `POST /receipts/open`. An open for a layer of a
  later stage is forwarded down the chain. The API binds only a loopback
  address unless the operator passes `--receipts-allow-remote`.
- **Auditor.** `skippy audit-receipt --claimed-model GGUF` sends a chat
  request, fetches the receipt, draws a challenge (one layer weighted by
  bytes, 16 positions), and checks Merkle paths and 16 random rows per matrix.
  It exits non-zero on failure.

The operator guide is [skippy/RECEIPTS.md](../../../skippy/docs/RECEIPTS.md).

Verified on a two-stage Qwen3-4B split on one RTX 5060 Ti:

| Case | Result |
|---|---|
| Serve Q8_0, claim Q8_0 | PASS on stage 0 and stage 1 layers, max error 1e-4 to 4e-4 |
| Serve Q6_K on both stages, claim Q8_0 | FAIL on every matrix, error 7e-3 to 5e-2 |

How the demo differs from the design:

| Design | Demo |
|---|---|
| Leaf = residual and mixer outputs; node replays the turn to open | Leaf = every weight matmul's input and output column; the node stores them in host memory: the live turn in full, committed turns up to 8 GiB per stage |
| Signed, chained turn roots | Unsigned per-turn roots, no chaining |
| Receipt at the end of the response stream | Receipt fetched by `x-request-id` after the response |
| Output head, attention, DeltaNet and MoE checks | Dense weight matmuls only. The output head and glue ops are not checked |
| Verifier in the host runtime, rows from Hugging Face | CLI verifier, rows from a local GGUF |
| All quant types | F32, F16, BF16 and Q8_0 weights only |

Known limits of the demo:

- Receipts made decode 1.8x slower on an RTX 5060 Ti and 2.3x slower on an
  RTX 5090, and prefill 9x to 10x slower, in a release build. The scheduler
  stopped after every matmul and copied its columns to the host.
- Only binary split serving commits receipts. Single-stage `skippy serve` and
  Mesh-launched stages do not.
- On the last layer of the final stage, the attention output and FFN matmuls
  run only for output rows, so their leaves cover fewer positions. The
  verifier does not yet check that every expected matrix is present.

## Milestone 1: replay instead of storage (in progress)

Built:

- **Native recorder and replay**
  (`model_support/0011-skippy-add-receipt-recorder-and-replay.patch`,
  ABI 0.1.75). `skippy_model_set_receipt_recorder` reports each computed
  ubatch: its geometry (tokens, sequences, positions, output rows, KV cells,
  n_kv), the stage input rows, the residual stream entering every layer and the
  stage output, and each layer's attn_q, attn_k and attn_v outputs and
  attention output. It also reports KV-cache operations.
  `skippy_model_replay_receipt` replays one sequence in a scratch context. It
  forces the recorded KV cells and n_kv and puts dummy columns where other
  sequences were. It also reports the matmul inputs and outputs of the
  challenged layer.
- **Leaves.** Per position, a stage commits `2 · layers + 1` leaves: the
  residual entering each layer, an attention leaf per layer, and the stage
  output. Receipt wire version 2.
- **Recorder.** Keeps leaf hashes and a replay log per native sequence, not
  activations. It refuses a sequence whose cells it cannot reproduce: a copied
  prefix, a position shift, a state load, or compaction.
- **Glue checks** in the auditor bind the replayed matmuls to the committed
  residual stream: both RMS norms, SwiGLU and the residual add. Supported for
  the `llama`, `qwen2` and `qwen3` architectures.
- **Throughput.** Committed tensors are copied into pinned host memory and
  hashed on up to 8 threads, off the decoding thread, while the next ubatch
  computes. Two-stage Qwen3-4B Q8_0 on one GPU, release build:

  | GPU | Decode | Prefill |
  |---|---|---|
  | RTX 5090 | 3% slower | 1.11x slower |
  | RTX 5060 Ti | 3% slower | 1.28x slower |

Verified on a two-stage Qwen3-4B split on one RTX 5060 Ti and on one RTX 5090:

| Case | Result |
|---|---|
| Serve Q8_0, claim Q8_0 | PASS on stage 0 and stage 1 layers, including the last layer. Row errors 1e-4 to 1e-3, glue errors about 1e-7 |
| Serve Q6_K on both stages, claim Q8_0 | FAIL on every matrix, error 4e-3 to 4e-2. Merkle paths, replay and glue pass |

Left to do:

- Replay on Metal, Vulkan (forced f32 accumulation), ROCm and ARM CPU. Only
  CUDA on Blackwell has been tested.
- Prefix cache, native MTP and multi-stream KV caches. They are refused now.
- Overlap the device-to-host copy with compute, or hash on the device, to
  remove the remaining per-ubatch copy cost.
- Glue checks for other layer shapes, expressed from the graph or model
  metadata instead of an architecture list.
- Done when: replay reproduces committed leaves bit for bit on CUDA, Metal,
  Vulkan and ARM CPU through real `skippy serve`, with prefix cache and native
  MTP on.

## Milestone 2: protocol on the response

- Put the receipt in the final response body and the last stream chunk.
- Sign and chain turn roots with the node key. Bind sampler, seed, llama.cpp
  pin, backend id and ubatch boundaries.
- Carry commit and open over the stage control plane instead of a side HTTP
  port, and enable receipts for Mesh-launched stages.

## Milestone 3: host runtime verifier

- Move the verifier into the host runtime. Read the GGUF header and rows from
  Hugging Face with range requests, pinned by revision and LFS hash.
- Add the output head unit, the loose attention check, DeltaNet and Mamba
  checks, and MoE (token, expert) pairs.
- Add rounding profiles for K-quants and the other common types, and the
  per-backend tables.
- Audit at least 10% of turns with a 10-second challenge window.

## Milestone 4: payments and trust tiers

- Add the `audit_pending` final state and pay audited finals after the verdict.
- Block a failing peer locally and keep the failure as provable evidence.
- Let a request policy require `receipts` ([TRUST_TIERS.md](TRUST_TIERS.md)).

## Later

- DSA (DeepSeek-V4, GLM-5.x) openings and the indexer glue check.
- ROCm profiles and replay.
- The phone verifier.
