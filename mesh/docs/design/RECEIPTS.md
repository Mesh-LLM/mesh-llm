# Inference Receipts

Status: proposal. The numbers below come from lab measurements. A first,
simpler version runs in Skippy split serving: see
[RECEIPTS_PLAN.md](RECEIPTS_PLAN.md) for how it differs from this design and
[skippy/RECEIPTS.md](../../../skippy/docs/RECEIPTS.md) for how to use it.

Related:

- [TRUST_TIERS.md](TRUST_TIERS.md): the `receipts` tier
- [../specs/lightning-payments.md](../specs/lightning-payments.md)
- [../NODE_REP.md](../NODE_REP.md)

## Summary

A client that pays for inference can check on its own CPU that the serving node
ran the model it claims. "The model" means the bytes of a GGUF file published on
Hugging Face. The client never downloads that file.

The node commits to its computation while it serves each turn. Right after the
turn, the client sometimes picks one layer and a few tokens at random and asks
the node to open them. The client checks the opened values against random weight
rows. It fetches those rows from Hugging Face with range requests only after the
node is bound by its commitment.

If the check fails, the client does not pay the final invoice and blocks the
node locally. Everything in the proof is either signed by the node or public, so
anyone can recheck it.

What a passing audit shows, for the tokens and layer opened:

- The node used the claimed weights, including the claimed quantization, for
  every matrix in the layer.
- The token-local operations between the matrices (norms, RoPE, activations,
  residual adds, MoE routing and combine) match the reference.
- Attention used the committed context (a loose check).

Because the node cannot know in advance which layer and tokens will be opened, a
node that cheats on every token fails almost every audit.

## Goals

- The paying client verifies. No trusted third party, stake, or mesh-wide
  reputation.
- Work for every model llama.cpp can serve, from small dense models to 400B+
  MoE, without hand-written per-model parameters.
- Work for agentic traffic: long contexts, many short turns, and little added
  latency.
- Keep the client's download and storage small enough for a phone later.
- Make failures provable.

## Non-goals

- Privacy. Receipts say nothing about who can read a prompt. See
  [TRUST_TIERS.md](TRUST_TIERS.md).
- Exact attention. Checking that the node's attention matches bit for bit needs
  an emulation for each GPU, kernel, and llama.cpp pin. The lab showed this is
  possible, but it is too costly to maintain. Attention gets a loose check.
- Catching a node that cheats rarely. The chance of catching it per turn is the
  audit rate times the share of turns it cheats on.
- Mesh-wide punishment. Consequences stay local to the client.

## Decisions

These choices bound the design:

| Topic | Decision |
|---|---|
| Who verifies | The paying client, on its own CPU |
| Model sizes | All, including 235B+ MoE |
| Devices | Desktop first. Phone integration deferred, but the parameters must fit a phone |
| Attention | Loose check only |
| Consequences | Withhold the final invoice. Block the peer locally. No gossip, no stake |
| Node identities | Free. A blocked node can rejoin under a new key |
| Failures | Must be provable |
| Audit rate | At least 10% of turns |
| Audit shape | One layer, chosen by the client after the node commits |
| Agentic | Commit and challenge only each turn's new tokens. Challenge at once, with a short window |
| Parameters | Derived from the GGUF header and per-backend tables, not per model |

## Protocol

### Model identity

A receipt names the model as `repo@revision/file` (all parts of a split GGUF)
together with the LFS SHA-256 of each part. This is the weights digest from
[TRUST_TIERS.md](TRUST_TIERS.md#model-identity-as-a-weight-hash). Pinning the
revision pins the bytes, so later range requests read the same file.

The client reads the GGUF header once per model with range requests. The header
holds the tokenizer, so this is 8 to 16 MiB, and it takes 2 to 7 seconds. The
client caches it. Everything else the client needs comes from the header:
architecture, layer count, tensor shapes, quantization types, byte offsets, and
expert counts.

### Commitment

For every new token `t` of a turn and every layer `l`, the node forms a leaf:

```text
leaf(t, l) = H( x_l(t) || mix_l(t) )
```

- `x_l(t)` is the residual stream entering layer `l`, as the backend holds it.
- `mix_l(t)` holds the outputs of the layer's token-mixing operations for
  token `t`: the attention output before the output projection, the K and V
  rows written to the cache, and for recurrent mixers (DeltaNet, Mamba) the
  mixer output.
- A final leaf binds the last hidden state, which feeds the output head.

`H` is BLAKE3. Leaves for each ubatch form a subtree, and the subtrees form the
turn tree. Turns are chained:

```text
root_n = H( root_{n-1} || H(new tokens) || tree_n || sampler and seed
            || llama.cpp pin || backend id || ubatch boundaries )
```

The node signs `root_n` with its node key and sends the receipt at the end of
the response stream, before the client decides whether to audit. The client
supplies the sampling seed, so sampling is checkable too.

Chaining means a later turn never re-commits earlier tokens. K and V from
earlier turns are bound by the earlier roots that the client already holds.

### Challenge

When the receipt arrives, the client decides with fresh randomness whether to
audit this turn. The rate is at least 10%. If it audits, it picks:

- **Unit:** one transformer block, or the output head. The chance of each unit
  is proportional to its matrix bytes, read from the header. This spreads the
  checks over the weights a cheater would want to shrink.
- **Tokens:** 16 of the turn's new tokens, uniformly, with at least 4 decode
  tokens when the turn generated any.
- **MoE pairs:** for MoE units, 16 (token, expert) pairs drawn from the experts
  the opened tokens were routed to.

The node must answer within the challenge window: 10 seconds by default. No
answer counts as a failure.

### Opening

For each chosen token, the node returns:

- `x_l(t)`, `mix_l(t)`, and `x_{l+1}(t)`, with Merkle paths to the turn root.
- The full output column of every matrix in the unit. For MoE units this
  includes the router logits and the outputs of the chosen experts.
- For attention: K and V for a sample of context blocks, and per-block
  flash-attention partials (running max, sum, and weighted value) for sampled
  heads. Recurrent mixers return their committed state at the chunk boundary
  before the token.

The node does not store residuals. It keeps a replay log for the window and
recomputes the opening in a scratch context, which reproduces every committed
bit (see [Node: commitment and replay](#node-commitment-and-replay)).

### Verification

The client:

1. Checks the Merkle paths against the signed root.
2. Picks 16 random rows of every opened matrix (and of each chosen expert),
   using fresh randomness drawn after the opening arrives. It fetches only
   those rows' bytes from Hugging Face with range requests.
3. Computes those rows of each matrix output on CPU: reference dequantization,
   the activation rounding that the node's backend uses, and f64 accumulation.
   It compares the result with the same rows of the opened column.
4. Rebuilds the token-local glue with llama.cpp's graph builder for the
   architecture on ggml CPU: norms, RoPE, activations, residual adds, and the
   MoE top-k and combine. It checks that the opened values chain from `x_l` to
   the committed `x_{l+1}` within tolerance.
5. Loose attention check: checks the sampled K and V against the K and V
   projections, recomputes the sampled blocks' partials, and checks that the
   partials combine to the opened attention output.
6. For the output head: checks the logits rows, runs the full-vocab Freivalds
   check with its stored head key, and checks the sampled token against the
   sampler, its parameters, the client's seed and the draw index.

The node cannot adapt to the rows, because it learns them only after it has
opened full columns. Rows are public weight bytes. The only secret the client
keeps is the output-head key (a few hundred KiB per model). A proven failure on
the head reveals that key, and the client builds a new one.

**Fail rule.** For each opened matrix and token, the client computes the
relative error over the checked rows:

```text
e = || y[R] - y_ref[R] || / ( || y || * sqrt(|R| / M) )
```

Here `y` is the opened column, `R` is the set of checked rows, and `M` is the
row count. A matrix fails when more than half of its opened tokens have
`e > 1e-3`. The audit fails when any matrix fails. The rule is per matrix, so
substituting a single tensor type still fails. Taking the majority over tokens
absorbs rare honest outliers.

### Consequences

- **Pass:** the client pays the final invoice.
- **Fail, or no answer in the window:** the client withholds the final invoice
  and blocks the node locally. The verifier is built into the host runtime, so
  it adds a host-side block next to the plugin peer block (`allow_peer_blocks`,
  published on `routing.choice.v1`). Blocks stay local and last the standard
  7 days.
- **Proof:** the signed receipt, the opening, the challenge, and the row
  indices. The rows are public, so anyone can rerun the check. When rows are
  fetched on demand there is nothing secret to reveal or rotate.

**Payments change.** The seller refuses a buyer that has `STRIKE_LIMIT` (3)
unpaid final invoices (`mesh-llm-payments/src/ledger/metered.rs`). An audited
final is paid 1 to 3 seconds late, so a buyer running fast agent turns could
trip that limit. The fix: the seller records each challenge it receives and
treats that request's final as `audit_pending`, not unpaid, until the window and
a grace period end. If the buyer never pays after a failed audit, the strike
then counts on the seller's side. That is harmless, because the buyer has
already blocked that seller.

## Parameters

| Parameter | Default | Source |
|---|---|---|
| Audit rate | 10% minimum, client policy | Decision |
| Units | Each block, plus the output head | GGUF tensor names `blk.N.*`, `output.weight` (or tied `token_embd.weight`) |
| Unit weight | Proportional to matrix bytes | GGUF shapes and types |
| Tokens per audit | 16, at least 4 decode | Calibration below |
| Rows per matrix | 16 | Calibration below |
| MoE pairs per audit | 16 | Fetch time below |
| Expert router | Fetched whole | A 2D tensor whose row count equals `expert_count` |
| Tolerance | `e > 1e-3` on more than half the tokens of a matrix | Calibration below |
| Activation rounding | Per backend and weight type | Backend table |
| Challenge window | 10 s | Agentic latency |
| Commitment hash | BLAKE3 | Throughput below |

Only two tables are maintained by hand: activation rounding profiles per
backend and weight type, and the tolerance. Neither is per model. A missing
table entry shows up as a systematic failure for that backend and type. It does
not cause a silent pass.

## Measurements

The lab harness hooks llama.cpp and compares backend outputs with a CPU
reference. It runs a 3,072-token prompt plus 16 decode tokens.

### Detection

CUDA on an RTX 5090. Real kernels, with no tie repair and no committed
activation blocks. Each "column check" is one matrix and one token, checked on
16 random rows. Eight simulated row sets per tensor.

| Run | Column checks | Median `e` | Share over 1e-3 | P(audit fails) |
|---|---|---|---|---|
| Honest, 6 quants of Qwen3-4B | 28k to 29k each | 1e-7 | 0 to 0.03% | 0 |
| Honest, Qwen3-8B Q4_K_M and Q8_0 | 29k each | 1e-7 | 0.003 to 0.05% | 0 |
| Honest, Qwen3.5-9B (hybrid) Q4_K_M and Q8_0 | 24k each | 1e-7 | 0 | 0 |
| Honest, Gemma 4 E2B Q4_K_M and Q8_0 | 33k each | 1e-7 | 0.04 to 0.22% | 0 |
| Honest, Qwen3-30B-A3B MoE Q4_K_M | 59k | 1e-7 | 0.01% | 0 |
| Honest, Qwen3-4B Q8_0, 8 batched lanes | 172k | 1e-7 | 0.02% | 0 |
| Honest, Qwen3-4B Q4_K_M, 16 batched lanes (prefill and mixed batches only) | 355k | 1e-7 | 0.22% | 0 |
| Q6_K served as Q8_0 (4B, 8B) | 29k each | 1.2e-2 to 1.3e-2 | 100% | 1 |
| Q6_K served as Q8_0 (Qwen3.5-9B, Gemma 4 E2B) | 24k, 34k | 1.1e-2 to 1.2e-2 | 76 to 89% (f32 and bf16 tensors, unquantized in both files, pass) | 1 |
| IQ4_XS served as Q4_K_M (4B) | 29k | 5.8e-2 | 99.5% | 1 |
| Q4_K_M as Q5_K_M, Q5_K_M as Q6_K (4B) | 29k each | 2e-2 to 4e-2 | 71% (tensors with the same type in both files pass) | 1 |
| Q3_K_M as Q4_K_M, Q4_K_M as Q5_K_M (Qwen3-30B-A3B MoE) | 59k each | 4.3e-2 to 7.2e-2 | 78 to 80% | 1 |

P(audit fails) is simulated: one layer, 16 tokens, the fail rule above,
bootstrapped from the runs, 20,000 trials. "0" means no failure in 20,000
trials, and "1" means every trial failed.

Two honest outlier groups were found, and both were gaps in the rounding
tables, not noise. Both profiles are now found:

- Gemma's `per_layer_model_proj` (bf16, cuBLAS): cuBLAS rounds the activations
  to bf16 in prefill. With that profile, `e` drops from 1.4e-3 to 1.7e-6.
- q4_1 `ffn_down` decode: the CUDA MMVQ kernel multiplies the block scales in
  half precision (`__hmul2`), so `d_w·d_x` and `m_w·s_x` are each rounded to
  f16. With that profile, `e` drops from 1.3e-2 to 3e-8. CPU q4_1 (x86 and
  ARM) has its own quirk: the q8_1 block sum is `f16(d·Σq)` with the unrounded
  `d`. With that profile, `e` drops from 9.7e-4 to 1e-6.

Before the fix, these sat near or above 1e-3, and a strict rule would have
failed honest nodes. This is the failure mode the tables must guard against.

Below 1e-3, attention projections in prefill reach 1e-4 on up to 15% of
columns. These come from activation-quantization near-ties, which is why the
threshold is 1e-3 and not 1e-4.

Apple M6 (Metal tensor API), same method: seven honest runs (Qwen3-4B in four
quants, Qwen3-8B, Qwen3.5-9B, Gemma 4 E2B) have a median `e` of 4e-7 and a
maximum of 5.5e-6, with no column check over 1e-3. Q6_K served as Q8_0 fails
every simulated audit on all three models tested.

**DGX Spark (GB10, CUDA sm_121).** Same method and rounding profiles as the
RTX 5090. Seven honest runs (Qwen3-4B in Q4_K_M, Q4_0, Q4_1 and Q8_0, Q4_K_M
with 8 batched lanes, Gemma 4 E2B and Qwen3.5-9B) have a median `e` of 1.3e-7
to 1.8e-7, and none fails a simulated audit. The share of column checks over
1e-3 is 0 to 0.34%, higher than on the RTX 5090. Most of them are in prefill
`attn_output`: one token in one layer, on every row set. The majority rule
absorbs them. The rarer decode outliers (up to 1e-4) come from CUDA op fusion
at activation-rounding near-ties: `GGML_CUDA_DISABLE_FUSION=1` removes most of
them. The q8 block scale is `amax · (1/127)` rounded, which differs from
`amax / 127` in 4.4% of blocks. That does not explain the outliers.

**Vulkan (RTX 5090, NV_coopmat2).** Decode matches the f32 profile (`e` ≤
3e-7). For quantized weights, prefill accumulates in f16 by default, which
puts honest `e` at 1e-3 to 4e-3, above the threshold. Every honest node would
fail. With f32 accumulation, prefill matches the f16-weight profile (`e` ≤
1.1e-5). For q6_K, the profile also rounds `d·scale` to f16 before the weight
(two roundings). Turning off coopmat or f16 alone does not fix it.

f32 accumulation costs 27% of prefill throughput on Qwen3-8B Q4_K_M (8,150 to
5,870 tokens/s); decode is unchanged. A Vulkan node that serves receipts must
force f32 accumulation for matmuls, and the receipt must record that it did.
Other Vulkan devices pick other paths (KHR coopmat, the scalar path, or integer
dot products), and each needs its own profile entry.

**ARM CPU (DGX Spark, Neon with i8mm and SVE).** The CPU profile matches as on
x86: `e` ≤ 4e-7 for q4_K, q4_0 (repacked) and q8_0, and ≤ 9e-7 for q4_1 with
the quirk above.

### Row fetch

These timings are from ben-ai-server to Hugging Face's CDN. Range requests only
(the CDN rejects multi-range requests). Each audit uses 16 rows per matrix and
fresh rows, so the client keeps no stored key.

| Model | Size | Header | One audit |
|---|---|---|---|
| Qwen3.5-27B Q4_K_M | 15.6 GiB | 16 MiB, 2.2 s | 144 ranges, 0.6 MiB, 0.7 to 1.1 s |
| Qwen3-235B-A22B Q4_K_M (3 parts) | 132 GiB | 8.5 MiB, 3.3 s | 800 ranges, 3.5 MiB, 1.2 to 2.5 s |
| Qwen3.5-397B-A17B Q4_K_M (6 parts) | 227 GiB | 11.7 MiB, 4.5 s | 900 ranges, 10 to 14 MiB, 1.4 to 2.0 s |
| DeepSeek-V3.2 Q4_K_M (9 parts) | 378 GiB | 10 MiB, 6.4 s | 1,300 ranges, 10 MiB, 1.3 to 2.3 s |

For MoE audits, most of the bytes are the whole expert router. Checking every
routed (token, expert) pair instead of 16 pairs took 21 to 26 seconds, which is
why pairs are sampled.

Fetched bytes matched the local copy of `unsloth/Qwen3-4B-GGUF` exactly
(4,048 ranges).

A client that prefers a stored key (offline audits, or less traffic to Hugging
Face) can fetch one ahead of time. A key holds 16 rows per matrix and 8 experts
per layer:

| Model | Key size |
|---|---|
| Qwen3.5-27B | 36 MiB |
| GLM-4.7-Flash | 37 MiB |
| Qwen3.5-397B-A17B | 86 MiB |
| Qwen3-235B-A22B | 106 MiB |
| DeepSeek-V3.2 | 129 MiB |

A stored key must stay secret. A proven failure reveals the rows that were
used, and the client then fetches new ones.

**At scale.** Limits seen from one IP:

- The resolver (`huggingface.co/.../resolve/...`) allows 3,000 requests per
  300 s (`ratelimit-policy` header). It redirects to a signed CDN URL that
  expires after about 60 minutes. The client resolves each file once and
  reuses the URL, so an audit costs one resolver call per file at most.
- The CDN (`cdn.hf.co`) set no limit over 24,000 range requests at about 800
  per second.
- HTTP/2 with 64 requests in flight on one connection: 423 requests per
  second, median 131 ms. A pool of 64 HTTP/1.1 connections: 342 per second.
  The verifier should use HTTP/2.

**Layer packages.** A `meshllm/*-layers` package stores each layer as its own
GGUF (`layers/layer-NNNNN.gguf`) plus `shared/*.gguf`, and its manifest names
the source repo, revision and file hashes. The tensor bytes equal the source
GGUF's: every tensor of Llama-3.2-1B, and layers 40 and 93 of
Qwen3-235B-A22B (24 tensors). So a receipt can name the source GGUF, and the
client can fetch rows from either copy. Fetching from a package costs one
header per audited layer file instead of one per source shard.

### Node: commitment and replay

Replay was tested on CUDA (RTX 5060 Ti). The probe hashed every layer's residual
output for every token and compared the hashes across recomputations:

| Recomputation | Qwen3-8B | Qwen3.5-9B (hybrid) | Gemma 4 E2B (sliding window) |
|---|---|---|---|
| Full rerun, same ubatches | identical | identical | identical |
| Drop KV from a ubatch boundary, replay to the end | identical | not supported (recurrent) | identical |
| Restore a sequence-state snapshot, replay | identical | identical (98 MiB snapshot) | differs |
| Replay decode tokens from the cache | identical | n/a | identical |
| Different ubatch size | differs | differs | differs |
| One token replayed alone | differs | n/a | differs |

A lane's bits depend on its own tokens and the geometry of each ubatch it
was in: the ubatch size, its column offsets, its KV cells, `n_kv`, and the
graph flags. They do not depend on what the other lanes hold. A lab probe on
Skippy's llama tree with 4 lanes and a unified KV cache showed this for dense,
MoE, sliding-window and hybrid models: replaying one lane with dummy tokens in
the other lanes' columns, at the recorded cells, gave identical residuals.

The same probes ran on the other backends, with the lab tree built for each
(Qwen3-4B, Gemma 4 E2B and Qwen3.5-9B on every backend, plus Qwen3-30B-A3B on
Vulkan):

| Backend | Rerun, prefix replay, snapshot restore, decode replay | 4-lane replay with dummy tokens | Different ubatch size |
|---|---|---|---|
| Vulkan (RTX 5090) | identical | identical (all 4 models) | differs |
| CUDA (DGX Spark GB10) | identical, except Gemma's snapshot restore (as on the RTX 5060 Ti) | identical | differs |
| ARM CPU (DGX Spark) | identical | identical | identical |
| Metal (Apple M6) | identical | identical | identical |

The CPU and Metal kernels do not depend on the batch size, but the replay
keeps the recorded geometry anyway. ROCm is not yet tested. These probes use
the lab tree. Skippy's serving stack is measured below, on CUDA and ARM CPU.

Replay was then tested on Skippy's real serving stack (RTX 5090, CUDA graphs,
the real scheduler, 4 concurrent requests). A scratch build of Skippy wrote a
replay log. An offline tool replayed each lane in a scratch context and
compared every logits row with the served one:

| Serving setup | Model | Logits rows identical |
|---|---|---|
| 4 lanes, staggered prefill and batched decode | Qwen3-8B | 181 of 181 |
| 4 lanes, prefix cache on, multi-turn reuse across lanes | Qwen3-8B | 141 of 141 |
| Same | Qwen3-30B-A3B (MoE) | 141 of 141 |
| Same | Gemma 4 E2B (sliding window) | 141 of 141 |
| Native MTP, 1 lane (26 verify windows, 5 rejected) | Qwen3.5-9B (hybrid) | 55 of 55 |
| Native MTP, 4 lanes | Qwen3.5-9B (hybrid) | 197 of 197 |
| 2-stage split, 4 lanes: stage 1 logits | Qwen3-8B | 180 of 180 |
| 2-stage split: stage 0 exports vs. stage 1 inputs | Qwen3-8B | 3,511 of 3,511 activation rows |

The same tools on a DGX Spark (GB10, CUDA), 4 lanes, prefix cache on, two
waves of requests that share a prefix:

| Model | Logits rows identical |
|---|---|
| Qwen3-4B Q4_K_M | 241 of 241 |
| Gemma 4 E2B (sliding window) | 241 of 241 |
| Qwen3.5-9B with native MTP (hybrid) | 267 of 267 |

| Qwen3-4B Q4_K_M on the Spark's ARM CPU (GPU hidden) | 241 of 241 |

The Qwen3.5 replay must use the server's recurrent-state slot count
(`n_rs_seq`). With a different count, most rows differ, so the log records it.

On CPU, two more node facts decide the bits. The thread count matters: CPU
decode attention splits the KV range per thread, and a replay with 4 threads
instead of the server's 20 differs on every decode row. The CPU backend's build
variant also matters: the release runtime builds ggml-cpu for generic ARM
(`GGML_NATIVE=OFF`), and a `-march=native` build of the same tree differs. The
node replays with its own runtime, so this only affects a lab tool. A receipt
must record the thread count and the runtime build.

**Two-node split with sparse attention.** DeepSeek-V4-Flash (43 layers,
145 GiB, DSA with a lightning indexer, hyper-connections) from its `meshllm`
layer package, split across two DGX Sparks over their direct link: layers 0 to
20 on one, 21 to 42 on the other. 4 lanes, prefix cache on:

| Run | Stage 1 logits rows | Stage 0 exports vs. stage 1 inputs | Checkpoints |
|---|---|---|---|
| 6 requests, prompts of 274 to 1,532 tokens | 240 of 240 | 6,386 of 6,386 | 48 of 48 |
| 5 requests sharing a prefix, prompts of 4.6k to 7k tokens (top-k selective) | 100 of 100 | 13,810 of 13,810 | 20 of 20, with 3 cross-lane restores |

Three things differ from the other models:

- The DSV4 cache keeps one stream per sequence and splits every ubatch by
  sequence, so lanes never share a ubatch. The replay lets the cache place
  cells itself. Forcing the recorded cells is not needed.
- With long prompts, Skippy moves lanes through state checkpoints: it saves
  one sequence and restores it into another. The replay follows each restore
  back to the sequence that saved it.
- Stages number lanes independently. Stage 0's sequence 2 was stage 1's
  sequence 3. Exports are matched to imports by position and row hash. The
  boundary activation is the hyper-connection state: 4 × 4,096 values per
  token.

The replay log, per stage, holds:

- Per ubatch: tokens, sequence ids, positions, output flags, the KV cell
  indices and `n_kv` of each cache, and the graph flags (`embeddings`,
  `embeddings_nextn` and its mask, causal attention). MTP turns on
  `embeddings_nextn` for every prompt row. That changes the last layer's
  shapes and its rounding, so a replay without the flag fails on the prefill.
- Memory operations at the leaf caches (KV cache, recurrent state): trim, copy,
  keep, shift, clear. Skippy trims hybrid memory and rolls back speculative
  windows through internal calls, so logging at the public API is not enough.
- State checkpoint saves and restores, matched by hash.
- For a stage after the first: the imported activation rows. Raw f32 on the
  wire is lossless, so these are the previous stage's committed outputs.

To replay a lane, the node starts from an empty scratch context and runs the
log in order. It mirrors operations on the lane and on every sequence whose
cells were copied into it (prefix-cache entries and the lanes that wrote
them), and puts dummy tokens in all other columns. A real implementation can
copy the lane's KV prefix from the live cache instead of replaying it, and
replay only the turn.

Stages batch differently. Stage 0 grouped decode tokens of up to 4 lanes in
one ubatch; stage 1 ran each lane's token in its own ubatch. Each stage keeps
its own log and replays its own layers.

What the node must keep for the window:

- The replay log for the turn, and the KV cells the turn reads (pinned, or
  rebuilt by replaying the lane from its first token).
- For recurrent models, the state at the start of the turn.
- A scratch context with room for the replayed lane's KV.

Without replay, the node stores the residual stream for every layer and token:
576 KiB per token for Qwen3-8B in f32. That is 1.7 GiB for a 3,072-token turn.

Commitment cost: BLAKE3 hashes 1.8 GB in 0.35 s on one core and 0.09 s on
8 cores. A 3,072-token turn of Qwen3-8B produces 1.7 GiB of f32 residuals, so
hashing is small next to the prefill (1.4 s on the 5060 Ti). It can overlap the
next ubatch. Copying the residuals to the host took 0.27 s when synchronized per
tensor. A real implementation would copy them asynchronously, or hash on the
GPU.

### Client checks

CUDA on the RTX 5090, prototyped in the lab harness. "Rel" is the relative L2
error per token. "Max" is the largest single element, relative to the
column's RMS.

**Glue.** The client rebuilds each matrix input from committed values in f64:
the previous layer's output, matrix outputs, attention outputs and weights.
That covers norms, SwiGLU, residual adds, RoPE, output-row selection, and the
MoE softmax, expert weights, normalization and weighted sum.

| Rebuilt value | Qwen3-8B | Qwen3-30B-A3B |
|---|---|---|
| Matrix inputs (norm, SwiGLU, attention output) | rel ≤ 1.5e-7, max ≤ 1.3e-5 | rel ≤ 1.3e-7 |
| Layer output `x_{l+1}` (incl. MoE combine) | rel ≤ 7e-8 | rel ≤ 1.1e-7 |
| V rows in the cache | bit-identical | bit-identical |
| K rows in the cache (norm, RoPE, f16) | 2 to 10% of values off by 1 f16 ulp, rel ≤ 1.3e-4 | rel ≤ 1.2e-4 |
| Q | rel ≤ 1.4e-4 | rel ≤ 1.8e-4 |

- Row checks on rebuilt inputs match row checks on the GPU's own inputs. For
  Qwen3-8B prefill, p99 `e` is 2.4e-7 for both, and the maximum is 7e-4 for
  both, from activation-rounding ties.
- RoPE error grows with position because the angle is computed in f32. At
  16k positions K reaches rel 4e-4. Exact RoPE is closer to the GPU than an
  emulation of the GPU's float formula, which uses fast-math sine. The K and Q
  tolerance should scale with position. A K or Q change within 1e-3 cannot
  change the attention output in a useful way.
- Q is committed with the attention output. The client checks the committed Q
  against its rebuild, then runs the attention check on the committed Q.
  Otherwise the Q error (1e-4) would hide the tight decode attention check.
- Wrapped graph inputs, such as the embedding the scheduler copies to the GPU,
  must be read while their consumers run. In the replay node this is the
  replay context's own buffer.

**Attention.** The honest error is GPU flash attention against exact f64
attention on the committed Q, K and V. The cheat error is a cheated output
against honest exact attention:

| Model, context | Honest prefill (max) | Honest decode (max) | Window 1024 (min) | KV q8_0 (min) |
|---|---|---|---|---|
| Qwen3-8B, 3k | 2.4e-3 | 7.6e-7 | — | — |
| Qwen3-8B, 16k | 5.7e-3 | 7.2e-7 | 1.4e-2 | 5.1e-3 |
| Qwen3-8B, q8_0 KV cache | 4.4e-3 (16k) | 3.2e-4 at 4k with Q as q8_1 (p99 6e-7) | 1.4e-2 | — |
| Gemma 4 E2B, 16k (sliding window, heads 256/512) | 9.0e-3 | 6.5e-4 | 6.3e-2 | 1.3e-3 |
| Qwen3.5-9B, 16k (hybrid) | 6.8e-3 | 3.6e-7 | 1.5e-2 | 1.8e-3 |

- Decode is tight on every model. With a q8_0 KV cache the CUDA kernel rounds Q
  to q8_1, so that is the profile for that pair.
- Prefill error grows about as the square root of the context. The combine
  check alone cannot separate mild cheats (q8_0 KV) from honest prefill.
  Recomputing sampled blocks does the detecting, and truncation fails both.

**Recurrent mixer (Qwen3.5-9B Gated DeltaNet).** The client recomputes the
state update and outputs in f64 from the committed q, k, v, gate, beta and
incoming state. Honest output error is at most 2e-5 in prefill and 1.6e-7 in
decode, and the state error is at most 6.2e-6. A state kept in f16 shows
about 1e-4 (median), bf16 about 1e-3, and a zeroed state 0.25 or more. A
tolerance near 5e-5 separates them. The short conv1d state must be committed
with the recurrent state.

**MoE routing.** On Qwen3-30B-A3B the top-k of the client's router logits
matched the experts used for every checked token, prefill and decode. The
smallest top-k gap seen was 6.7e-4, against router logit errors near 1e-7.
Expert ids are committed. The client accepts them when they are the top-k
within a small band.

**Sparse attention (DSA).** DeepSeek-V4-Flash scores the compressed KV blocks
with a lightning indexer (64 heads × 128, ReLU, per-head weights), and
attention reads the top 512. Top-k selects only past 2,048 tokens, because
the blocks are compressed 4×. On the long-prompt split run above, the client
recomputed the indexer scores in f64 from the GPU's q, k and weights, and
compared the GPU's selection with the exact top 512, across all 21 indexer
layers:

| | Rows (token × layer) | Result |
|---|---|---|
| Honest score error, relative to the row's score spread | 290k | p99 ≤ 9.6e-5, max 2.3e-3 |
| Honest selections that differ from the exact top 512 | 204k (104M selections) | 7,818, all within 1e-3 of the spread from the 512th score (a band that holds 0.7% of positions) |
| Cheat: indexer with half its heads | 107k | 99.8% of rows have a selection outside the 1e-3 band (9% of selections); 97% at 1e-2 |
| q and k rounded to bf16 | 107k | 3 rows outside 1e-3, none at 1e-2: within tolerance |

The client accepts a selection when it has exactly `top_k` entries (from the
GGUF), every selected position scores at least `τ - band`, and every
unselected position at most `τ + band`. Here `τ` is the exact 512th score, and
the band is 1e-2 of the row's score spread, above the honest maximum. A full
check needs all of the token's indexer keys in that layer: 1,750 × 128 values
at 7k tokens. A cheaper opening gives the selected indices and `τ`. The client
then checks 16 selected and 16 unselected positions, which catches the
half-heads cheat on most rows. Each sampled key must also chain to the
committed residuals through the indexer's compressor. The eval callback
that reads the indexer splits the graph and changes rounding, so these
numbers come from a separate pass, not the bit-exact replay.

**Output head and sampling.**

- Logits rows pass the row check like any other matrix (`e` ≤ 1.1e-7).
- 16 random rows of 151k cannot catch steering: lowering the honest top token
  or raising one target token changes one or two rows. Steering does not save
  compute, but it changes the output. A full-vocab Freivalds check closes this.
  The client keeps a secret ±1 vector `r` over the vocabulary and stores
  `r^T W` (K values). It compares `r · y` with `(r^T W) · x`. On Qwen3-8B
  (q6_K head, CUDA) the honest difference is at most 3.7e-4 logits, so any
  single-logit change over about 1e-3 shows up. Four vectors make a 128 KiB key,
  built in one pass over `output.weight` (480 MiB). The top-two logit gap was
  as small as 7.4e-4, so near-ties may flip within tolerance.
- Skippy samples on CPU with llama.cpp's sampler chain. The `dist` sampler
  draws once per output token, also in speculative verification (sample and
  match, stopping at the first mismatch). The draw index is therefore the
  output-token index. `std::uniform_real_distribution<double>` over `mt19937`
  gave identical draws under GCC 15 libstdc++ and Apple clang 21 libc++. The
  Rust verifier should implement the same two-draw formula rather than rely on
  a C++ library.
- Requirements for a checkable sample: a nonzero seed (Skippy maps seed 0 to a
  random seed), no Mirostat (its state depends on every earlier logit vector),
  and CPU sampling (in-graph backend sampling does the softmax on the GPU).
  The client accepts a token when the draw falls inside its interval, or
  within 1e-6 of an edge, to absorb `expf` differences between libraries.
- Greedy and MTP greedy decoding: the token must be the argmax, with the same
  tie band.

### Client CPU

The row checks are 16 dot products per matrix per token: under a millisecond.
The glue recomputation for one layer and 16 tokens is of the same order.
Network time for the row fetch dominates an audit.

## Agentic Workloads

- Each turn commits and opens only its new tokens. The chained roots bind
  earlier turns, so long contexts do not make audits larger. The exception is
  the sampled attention blocks.
- The client challenges as soon as the receipt arrives. The node pins the
  turn's KV (and snapshots) for the window, then releases them. Prefix caching
  across turns is unaffected.
- Turns that are not audited pay at once. Audited turns pay after the verdict,
  1 to 3 seconds later, overlapping the next turn.
- Split serving: each stage host commits its own layers and its imported
  activation rows, and the challenge goes to the stage that owns the chosen
  layer. A stage's input rows must equal the previous stage's committed
  outputs; the client checks this at the boundary.
- Prefix caching: a lane can attend to cells that another request computed. The
  node replays those cells from their writer's log, and it can open them like
  the turn's own positions.

## Phones (deferred)

The embedded `mesh-client` has no peer blocks and no payments yet. The
parameters already fit a phone:

- No storage is needed when rows are fetched on demand.
- A dense audit downloads about 0.6 MiB.
- MoE audits download 3.5 to 14 MiB, mostly routers. A phone can check router
  rows instead of the whole router. This weakens the routing check.
- A stored key is under 130 MiB even for the largest model tested. Fewer rows
  or experts bring it lower.

## Accepted Risks

- **Low audit rate and free identities.** At 10%, a node cheating on every turn
  expects to serve about 10 turns to a client before that client catches it.
  It can then rejoin under a new key. Cheating can still pay. This was
  accepted.
- **Selective cheating.** A node can cheat only toward clients that never
  audit, or only on some turns. Each client's protection is its own.
- **Loose attention.** KV cache quantization and attention approximations
  within tolerance pass.
- **Partial-row cheats.** A node that degrades a fraction `q` of a matrix's rows
  is caught by that matrix's check with probability about `1 - (1 - q)^16`.
- **Hugging Face availability.** If the CDN cannot be reached, the client uses
  a stored key or skips the audit.

## Open Items

Not yet measured or built (see [RECEIPTS_PLAN.md](RECEIPTS_PLAN.md) for the build plan):

- ROCm is deferred: no rounding profile or replay test yet, so ROCm nodes
  advertise no receipts until one is measured.
- Vulkan on AMD and Intel devices. Vulkan is measured on NVIDIA (coopmat2)
  only, and needs forced f32 accumulation, which costs 27% of prefill
  throughput.
- Replay through real `skippy serve` is measured on CUDA (RTX 5090, DGX Spark
  GB10) and ARM CPU, not yet on Vulkan or Metal. Lab replay is identical on
  CUDA, Vulkan, Metal and ARM CPU.
- Draft-model speculation: `--draft-model-path` loaded the draft but never
  proposed in standalone `skippy serve` (`draft_n` stayed 0), so only native
  MTP rollback was measured.
- The opening format for attention partials. The f64 attention and DeltaNet
  recomputations are measured (see [Client checks](#client-checks)); the
  partial-block encoding and its size are not.
- The DSA opening format (sampled positions, and how many) and the indexer
  compressor glue check. Selection and replay are measured on
  DeepSeek-V4-Flash. GLM-5.3-Flash does not split on the current Skippy pin:
  its `meshllm` package names the old architecture (`glm5next`), and after a
  rename the native planner rejects the graph ("unregistered state write
  before a registered write").

## Rollout

1. **Node:** in Skippy, behind a feature flag. Per-layer residual hashing, turn
   trees, chained signed receipts, KV pinning, and replay. Expose the receipt
   at the end of the response stream.
2. **Client:** a verifier built into the host runtime (not a plugin). It reads headers, makes
   challenges, fetches rows, runs the CPU checks, and requests peer blocks.
3. **Payments:** the `audit_pending` final state, and paying audited finals
   after the verdict.
4. **Trust tiers:** let a request policy require `receipts`.
5. **Phones:** port the verifier to the embedded client once it has payments.
