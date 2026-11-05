# Receipts implementation plan

This plan turns [RECEIPTS.md](RECEIPTS.md) into milestones. Each milestone
leaves `skippy serve` working with receipts off, which stays the default.

## Milestone 0: end-to-end demo (built, then replaced by milestone 1)

Milestone 0 stored every matmul column. Milestone 1 replaced its native tap
and leaf layout; serving, the HTTP API and the auditor carried over.

What it built:

- **Native tap** (ABI 0.1.79). `skippy_model_set_receipt_tap` reported every
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
  ABI 0.1.79). `skippy_model_set_receipt_recorder` reports each computed
  ubatch: its geometry (tokens, sequences, positions, output rows, KV cells,
  n_kv), the stage input rows, the residual stream entering every layer and the
  stage output, and each layer's token-mixing inputs, stored KV rows (in the
  cache's type) and token-mixing outputs. It also reports KV-cache operations.
  `skippy_model_replay_receipt` replays one sequence in a scratch context. It
  forces the recorded KV cells and n_kv and puts dummy columns where other
  sequences were. It reports, for each challenged layer of the stage, the
  committed tensors, every weight matmul's input and output (with the
  experts of a `mul_mat_id`), the selections, and stops each step after the
  last of those layers.
  `skippy_model_release_receipt_scratch` frees the scratch context.
- **Receipt tensors from graph structure**
  ([RECEIPTS_GLUE.md](RECEIPTS_GLUE.md)). A classifier with a per-ggml-op
  table finds each layer's weight matmuls, token-mixing regions, stored rows
  and routers between Skippy's block boundaries. No code names an
  architecture, a weight or a node. A graph it cannot classify gets no
  receipt, with a reason.
- **Leaves.** Per position, a stage commits `2 · layers + 1` leaves: the
  residual entering each layer, a token-mixing leaf per layer, and the stage
  output. Each value is hashed with its ggml type. On the stage that holds
  layer 0, the residual leaf of layer 0 also binds the position's token.
  Receipt wire version 0.
- **Recorder.** Keeps leaf hashes and a replay log per native sequence, not
  activations. It refuses a sequence whose cells it cannot reproduce: a copied
  prefix, a position shift, a state load, or compaction.
- **Opening cost.** A replay stops after the challenged layer, which cuts an
  opening to between 1/3 and 2/3 of its earlier time on the two-stage split.
  A repeated challenge is answered from kept openings (128 MiB per stage). A
  receipt is replayed for at most 4 distinct challenges. Serving frees the
  replay context after 60 seconds idle.
- **Challenge and binding.** A challenge picks four layers,
  weighted by weight bytes, with four positions in each, half from the
  generated positions. Every stage opens its part in one replay: its picked
  layers, the input of layer 0 at every picked position (with the token), and
  the residual leaves at its boundaries. The client checks that the stages
  cover every layer of the claimed GGUF (its layer count comes from the GGUF
  metadata), that the residual one stage sent is the one the next stage
  received, that the input of layer 0 is what layer 0 of the claimed GGUF
  computes from the receipt's token (`skippy_model_eval_receipt_entry`), and
  that the response is the text of the sampled tokens. The row fail rule
  judges each weight role across the opened layers.
- **Glue checks.** The auditor loads the challenged layer of the claimed GGUF
  on the CPU and computes it on the opened values with libllama
  (`skippy_model_eval_receipt_glue`). It compares the matmul inputs, expert
  ids, token-mixing inputs, stored KV rows, selections and the residual out
  with the opening. Routers are row-checked on every row.
- **Throughput.** Committed tensors are copied into pinned host memory and
  hashed on up to 8 threads, off the decoding thread, while the next ubatch
  computes. Two-stage Qwen3-4B Q8_0 on one RTX 5060 Ti, release build, one
  layer per challenge: decode 3% slower, prefill 1.38x slower.

Verified with the bindings across layers, the two-stage Qwen3-4B split on
the CPU:

| Case | Result |
|---|---|
| Serve Q8_0, claim Q8_0 | PASS: four layers on both stages, input of layer 0, stage boundaries, sampled tokens against the response |
| Serve Q6_K, claim Q8_0 | FAIL: every weight role, and the input of layer 0 (error 1.9e-2). Merkle paths, boundaries, replay and glue pass |

Verified before the bindings across layers, with one layer of 16 positions
per challenge, two-stage splits on one RTX 5060 Ti and on the CPU:

| Case | Result |
|---|---|
| Qwen3-4B, serve Q8_0, claim Q8_0 | PASS on layers 0, 5, 17, 18, 30 and 35. Row errors 1e-4 to 3e-4; glue errors at most 3e-6 (GPU against the CPU) |
| Qwen3-4B, serve Q6_K, claim Q8_0 | FAIL on every matrix, error 1e-2 to 1e-1. Merkle paths, replay and glue pass |
| granite-3.1-3b-a800m (MoE), serve Q8_0 | PASS on layers 0, 5, 20 and 31. Expert ids and selections match the CPU exactly |
| granite-3.1-3b-a800m, serve Q6_K, claim Q8_0 | FAIL on every quantized matrix; the f32 router passes |
| Qwen3-4B with a q8_0 KV cache | No receipt: the key rotation llama.cpp applies before a quantized cache is not classified yet |
| Qwen3.5-9B (hybrid) | No receipt: the recorder needs a plain KV cache |

Left to do. Serving cost and agentic turns come first; backend and
architecture coverage moved to milestone 5. A stage that receipts do not
support serves normally without them, so the deferred work costs coverage,
not correctness.

- Overlap the device-to-host copy with compute, or hash on the device, so a
  ubatch copies leaf hashes instead of the committed tensors. Budget: decode
  at most 3% and prefill at most 10% slower than with receipts off. The
  copy now overlaps compute, and on CUDA the GPU hashes the values and
  copies only digests: decode is within noise, and prefill is 1.06x slower
  on both PCIe 5.0 x16 and PCIe 4.0 x4. Done for CUDA; other backends copy
  the values (milestone 5).
- Replay only the turn. Copy the sequence's KV prefix from the live cache
  into the scratch context and pin it for the challenge window, instead of
  replaying from the first position, which grows with the conversation. Size
  the scratch context to the turn, not the whole context.
  An opening now copies the prefix cells the turn read from the live cache,
  after checking they still hold what the turn read, and replays only the
  turn; the scratch context holds only the cells the turn uses. Pinning is
  done: a freed cell keeps its rows until a ubatch writes it, and each
  stage pins the cells of a bound turn for the 10-second challenge window
  and 2 seconds more, and of an opened one until the client has its
  rows, so a later request writes other cells while the cache has room.
  A full cache still writes pinned cells, and the turn is then
  unopenable. Pinned cells widen later requests' attention window:
  back-to-back bound Qwen3-4B requests on an RTX 5090 decoded 1.7%
  slower than with receipts off, and unbound ones and continued
  conversations ran at the same speed.
- Prefix cache and native MTP, which are on by default. They are refused now,
  so most agentic turns get no receipt. A copied prefix was written by
  another request, so the root binds a hash of the prefix cells the turn
  reads, and the loose attention check (milestone 3) samples them.
  The prefix cache is done: each stage commits a prefix root, and a turn
  from a cached prefix opens and passes on CUDA. Exact replay of cached
  generated tokens is off with receipts. Native MTP stages now issue
  receipts too. The replay context keeps every row through the last layer,
  as the stage does when an MTP head reads them. A rejected draft removes
  the tail of its verify window, and an opening still replays only the
  windows it opens, from copies of the turn's cells. The receipt ends at
  the last token the turn fed, not at drafts the last window rejected. On
  a two-stage CPU split of Qwen3-30B-A3B with an MTP head, turns with
  accepted drafts and turns that rejected every draft open and pass.
- Done when: replay reproduces committed leaves bit for bit on CUDA and CPU
  (x86 and ARM) through real `skippy serve`, with prefix cache and native MTP
  on; serving stays within the budget above; and an opening of a turn with a
  32k-token context finishes within the 10-second challenge window.
  An opening now replays only the ubatches that hold a challenged
  position, from copies of the turn's own cells. On a two-stage Qwen3-4B
  split, a fresh 15,000-token turn opens in 1.7 s on an RTX 5060 Ti
  (5.5 to 8 s before) and a fresh 31,000-token turn in 1.9 s on an RTX
  5090; turns from a cached prefix open in about 1 s. Both pass.

## Milestone 2: protocol on the response

- Check the output head and the sampler first: replay to the end of the last
  stage, and the client computes the head and the sampled token from the
  final residual. Until this is done, a node can run the claimed layers and
  still change the response.
- Compare the sampled tokens with the response exactly, after the node's
  output parser, instead of the containment check. An empty response must not
  pass.
- Put the receipt in the final response body and the last stream chunk.
- Sign and chain turn roots with the node key. Bind the client's key,
  sampler, seed, llama.cpp pin, backend id, rounding profile and ubatch
  boundaries.
  The root now binds the sampler's parameters and seed, each stage's
  ubatch boundaries and prefix root, and a client checks the parent chain.
  The runtime names the Skippy ABI and the llama.cpp commit and patch
  queue the native library was built from.
- Derive the challenge with a VRF (RFC 9381, ECVRF-EDWARDS25519-SHA512). The
  client holds a receipts VRF key that its node key signs, so it is not the
  transport key. The VRF of the root picks whether to audit, the layers and
  the positions; the client sends that proof with the challenge, and the node
  checks it. The VRF of the opening's hash picks the rows; that proof is
  shown only in a dispute. The client cannot draw rows again until an honest
  node's rare outliers fail it, and the node cannot predict the rows.
- Check rows with the rounding profile of the backend the receipt names, not
  the best of all profiles.
- Count the sampled tokens that fall in the tie band of the sampler check,
  and flag a turn with many more than chance. The rate needs calibration.
- Carry commit and open over the stage control plane instead of a side HTTP
  port, and enable receipts for Mesh-launched stages.
  Each stage now commits and opens the next one over a receipts connection
  on the binary stage protocol, and a Mesh split records receipts on every
  stage when the coordinator has `[receipts]` on: it asks for them in the
  stage load (`issue_receipts`). Verified on a two-node Qwen3-4B split on
  one RTX 5060 Ti, with challenges of the layer on the other peer. A peer
  that predates the change records nothing; the turn then gets no receipt
  and keeps the reason. No subprotocol feature lets the coordinator see
  that before a turn.

## Milestone 3: host runtime verifier and payments

- Move the verifier into the host runtime. Read the GGUF header, the rows,
  and the small tensors the glue check uses from Hugging Face with range
  requests, pinned by revision and LFS hash. The client keeps no copy of the
  model.
  `skippy audit-receipt` and `recheck-receipt` now take
  `--claimed-model repo@revision/file`. They read the bytes the checks use
  by range into a sparse copy keyed by the resolved commit. A Mesh split
  names a Hugging Face GGUF that way in its receipts. A split GGUF is
  named by its first shard, and the verifier reads every shard. Verified
  with Qwen3-4B Q8_0 in three shards on a two-stage split (CPU): the
  drawn layers in the first two shards and the output head in the third
  pass, and so does the recheck.
  The verifier is in the host runtime: with `[receipts] audit = true` a
  Mesh node names a fresh receipt key on the chat completions its clients
  send to peers, reads the receipt from the relayed response, and audits
  the turns the VRF draws on one worker thread. Verified on two nodes on
  one machine with Qwen3-4B: honest turns passed in about 8 s each (debug
  build); a peer that served Q6_K while its receipts named the Q8_0 file
  failed every weight role, the input of layer 0 and the output head, and
  was blocked before the next request.
- Add the output head unit, the loose attention check, and MoE (token,
  expert) pairs. The attention check matters most: cutting the context is
  the cheat that saves the most compute on long agentic turns.
  The head unit is done: the final stage opens its replay's logits
  (`SKIPPY_RECEIPT_TENSOR_LOGITS`, ABI 0.1.84), and the client checks them
  with a secret four-vector Freivalds key, 16 rows and the sampled row.
  The head check went from about 6 s and 405 MiB of cached matrix to
  about 10 ms and a 107 KiB key on Qwen3-4B. The honest key error was at
  most 1.5e-7 of the logits' norm on CUDA.
  MoE pairs are done: each layer checks 16 (position, expert) pairs drawn
  from the routed ones, and the router whole. A model that scales its
  logits (Granite) is checked with the key at that scale. A sampled
  turn's token must be one its stated top-k, top-p, min-p and temperature
  filters keep; penalties and grammars leave the token unjudged.
  The attention check is done for softmax attention (causal, sliding and
  chunked windows, soft caps and sinks): per picked layer, two positions
  with a drawn head each open per-64-position flash partials from the
  live cache (ABI 0.1.87). The client checks that they combine to the
  committed output, then draws two blocks per pick from the opening's VRF
  output and asks for their K and V rows in a second request, which it
  checks against the committed stored rows and recomputes. On a
  two-stage Qwen3-4B Q8_0 split on CUDA, the partials added about 3 KB to
  a 9.7 MB opening and the rows were about 1 MB in 0.01 s; honest combine
  errors were at most 4e-3 (tolerance 2e-2) and block errors at most
  5e-8 (tolerance 1e-6).
- Add rounding profiles for K-quants and the other common types, and the
  per-backend tables. Most served models are K-quants.
- Audit at least 10% of turns with a 10-second challenge window.
  A Mesh node audits 10% of receipted turns, drawn by the VRF, and drops
  a turn that would wait past 10 seconds as no audit.
- Add the `audit_pending` final state and pay audited finals after the
  verdict. Without it, a buyer that runs fast agent turns reaches
  `STRIKE_LIMIT` as soon as audits start.
  Built in the node first, then removed when payments moved out of the node
  into the `pay-lightning` plugin (`Mesh-LLM/ln-payments-plugin`). A failed
  audit still blocks the peer, but no longer withholds its final invoice.
  Restoring this needs a plugin seam that carries the audit verdict to the
  payment plugin, and the pending state in that plugin.
- An audit that cannot run counts as no audit: a llama.cpp pin the verifier
  cannot match, a backend or weight type without a profile, or Hugging Face
  out of reach. Only a check that ran and failed withholds an invoice.

## Milestone 4: consequences and trust tiers

- Block a failing peer locally and keep the failure as provable evidence:
  the signed receipt, the challenge, the opening and both VRF proofs.
  Done in the host runtime: a failed audit blocks the peer for seven
  days with `receipts` as the requester, and keeps the evidence, which
  `skippy recheck-receipt` fails again from public data.
- Add a command that rechecks the evidence from public data alone.
  `skippy recheck-receipt --evidence FILE` does this when the receipt names
  a Hugging Face file. Verified on a two-node Mesh split with an empty
  cache, in 12.7 s.
- Let a request policy require `receipts` ([TRUST_TIERS.md](TRUST_TIERS.md)).
  Done: a served-model descriptor gossips whether its runtime issues
  receipts. `x-mesh-trust: receipts`, or `[receipts] require = true` on
  the node, routes a chat completion only to receipted runtimes, audits
  its turns, and fails closed with `trust_policy_unsatisfied`.

## Milestone 5: coverage

- Replay through real `skippy serve` on Metal and Vulkan. Vulkan needs forced
  f32 accumulation, which costs 27% of prefill throughput, so the operator
  turns receipts on for it.
  Done for Vulkan: a recorder on a Vulkan stage forces f32 accumulation,
  and the auditor checks Vulkan rows with an `f16_weights` profile and
  `cpu_q8_0`. Two-stage Qwen3-4B audits pass on an Intel iGPU (Q8_0) and
  an RTX 5060 Ti (Q8_0 and Q4_K_M), and a Q4_K_M node that claims Q5_K_M
  fails. Prefill with receipts is 1.66 times slower there.
  Metal: replay works once values are read into pageable memory with
  blocking copies, and decode rows match exactly. Prefill rows are 1e-4
  to 8e-4 from the exact product and fit no rounding profile, so Metal
  stages stay unaudited until the `mul_mm` rounding is modeled.
- Multi-stream KV caches.
  Done: a stage whose cache keeps a stream per sequence issues receipts.
  The mirror, replay placement, seeds, row reads and pins follow each
  sequence's stream. Four concurrent audits passed on a 4-lane Qwen3-4B
  CPU split with stage 1 so configured. Prefix reuse on such a stage
  aborts in serving, with receipts off too (a sequence id past the
  cache's), so the cross-stream copy is covered by unit tests only.
- Recurrent and hybrid memory (Mamba, the gated delta net), so their state
  can be replayed, with the DeltaNet and Mamba checks, and the key rotation
  of quantized KV caches.
  Done for replay: a hybrid stage reads the recurrent state each turn
  starts from, commits its digest (receipts v4) and keeps it while the
  turn can be challenged, and a replay starts from it. Two-stage
  Qwen3.5-9B Q4_K_M audits pass on the CPU, with the projections, FFN
  and attention layers checked. Left: the DeltaNet and Mamba checks,
  which need the state committed at chunk boundaries during serving,
  since a node that sees the challenge could solve for a starting state.
  Done since: the attention check follows gated outputs (Qwen3.5 layer
  3: combine error 6e-4) and the rotation of a quantized KV cache.
  Qwen3-4B with a q8_0 cache issues receipts, and audits of layers 0
  and 20 pass on the CPU with combine errors under 2e-7 once the check
  rounds q to q8_0 as the kernel does.
  Done since: the recurrent check (receipts v5). Each ubatch commits the
  state it read in the mixing leaf of its first position, so the state
  is fixed before any challenge. For the first attention pick of a stage
  on a recurrent layer, the stage opens the mixing leaves of the pick's
  ubatch and of the position after it. The auditor runs the layer's own
  token-mixing ops on the CPU (`skippy_model_eval_receipt_region`), which
  covers the delta net, Mamba and the short convolution alike, and
  compares the outputs, the state left and a zero first state. On the
  two-stage Qwen3.5-9B CPU split, regions of 1 to 16 positions on layers
  0 to 2 match exactly in about 0.1 s. With the node on CUDA (RTX 5060
  Ti), 64-position regions are within 2.6e-5 of the CPU auditor, and
  decode slows by about 4%.
- Open lanes that shared ubatches in one replay. A lane's bits do not depend
  on the other lanes' values, so the columns that hold dummy tokens now can
  hold other challenged lanes.
  Deferred: a shared replay saves work only when two lanes that shared
  ubatches are both challenged while one replay can serve them. At a 10%
  audit rate that is rare. Each client challenges at the end of its own
  turn, so one opening would have to wait for the other lane's turn to
  end, and that wait adds latency for the client. It also needs native
  replay of several seeded sequences and merged step plans. Revisit if
  openings become a measurable share of node time.

## Later

- DSA (DeepSeek-V4, GLM-5.x) openings and the indexer glue check.
- ROCm profiles and replay.
- The phone verifier.
