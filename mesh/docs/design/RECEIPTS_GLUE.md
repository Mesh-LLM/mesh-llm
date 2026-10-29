# Receipts without model-specific code

Status: built (receipt wire version 3). The native classifier is in
`src/skippy/receipts_layers.cpp` of patch 0011, the auditor's checks are in
`skippy-receipts` (`glue.rs`, `audit.rs`), and the glue evaluator over
libllama is in `skippy-cli`. Measured costs are at the end.

Receipts ([RECEIPTS.md](RECEIPTS.md)) had model-specific code in two places
before this design (wire version 2):

1. **Leaf discovery** (native recorder). It finds the receipt tensors by GGUF
   weight names (`blk.N.attn_q.weight`, `ffn_gate`, ...), takes the residual
   from the `rms_norm` in front of `attn_q`, and finds the stage output by the
   node name `l_out-N`.
2. **Glue checks** (auditor). `glue.rs` re-implements the operations between
   the matmuls of one layer (two RMS norms, SwiGLU, residual adds) for the
   `llama`, `qwen2` and `qwen3` architectures only.

This design removes both. No code names an architecture, a tensor role or a
node name.

## Requirements

The decisions below are measured against these requirements, agreed for
receipts so far:

- **No model-specific code.** Nothing may list architectures, tensor names or
  node names.
- **Security.** The audit must catch a node that serves a cheaper model than
  it claims (a lower quantization, a smaller model, skipped work). What happens
  inside attention is not checked today, and that stays accepted.
- **Honest nodes never fail an audit.** When a node cannot support a check, it
  refuses to issue a receipt (fail closed, as it does today for a shared prefix
  cache). It never issues a receipt that an honest node would then fail.
- **Cost.** Receipts stay at about 5% decode overhead or less. Openings stay
  bounded by replay. KV-cache snapshots were rejected as too expensive.

## Idea

Split every layer's computation for one position into three parts:

- **Weight matmuls.** Any `MUL_MAT` or `MUL_MAT_ID` whose weight operand is a
  model tensor of the layer, whatever its name. The row checks verify these.
- **Token-mixing operations.** Operations where a token's output depends on
  other tokens' values. Attention, `ssm_conv`, `ssm_scan`, the gated delta net
  and RWKV's WKV are examples. Their inputs and outputs are committed. What
  happens inside them is not checked, as with attention today.
- **Glue.** Everything else. Given the residual entering the layer, the
  matmul outputs and the token-mixing outputs, glue is a function of one
  position: norms, RoPE, biases, activations, gating, MoE routing, residual
  adds.

The client checks the glue by **evaluating it itself**. It builds the layer's
graph with its own pinned llama.cpp and Skippy graph builder from the claimed
GGUF. It runs the graph on the CPU, one position at a time, with the opened
matmul outputs and token-mixing outputs substituted for those nodes. Then it
compares what it computes with the opened matmul inputs and the opened
residual leaving the layer. Any architecture that llama.cpp can build is
covered, with ggml's own op implementations.

## Decision: which operations mix tokens

An operation mixes tokens when its output column for one token depends on any
other token's column. This is decided from what each ggml op does along the
token axis, by a dependency pass over the built graph:

- Every graph input that holds one column per token (the block entry, token
  ids, positions) starts as per-token.
- A pass in graph order marks each node per-token when its op keeps the token
  columns separate and all its inputs are per-token or weights. Element-wise
  ops, norms over the embedding axis, RoPE, `get_rows` on weights, and
  `mul_mat` against a weight keep columns separate.
- A node that combines columns of different tokens is token-mixing:
  `mul_mat` between two activations along the token axis, `soft_max` over the
  key axis, `flash_attn_ext`, `ssm_conv`, `ssm_scan`, the delta-net and WKV
  ops, and views or concatenations that shift along the token axis. Any read
  of a KV or recurrent-state buffer also counts as token-mixing, because that
  buffer holds other tokens.
- The table that says how each ggml op treats the token axis covers ggml ops,
  not models. A node whose op is not in the table makes the graph
  unreceiptable until the op is added.

I chose this over the rule proposed first ("an op that reads a memory tensor
mixes tokens"):

- **Honest nodes.** The memory rule files non-causal attention over the ubatch
  (encoder and embedding models, which have no KV cache) as glue. The client
  cannot evaluate that one position at a time, so honest nodes would fail.
  The structural rule files it as token-mixing.
- **Security.** Both rules need the guards below. The structural rule does
  not trust anything the memory rule checks.
- **No model code.** The op table is per ggml op, about a hundred entries,
  and every model reuses it. An unknown op stops receipts instead of guessing,
  which keeps both of the above.

## Guards

The split alone leaves four holes, found while reviewing the memory rule.
Each guard below closes one, using only graph structure.

### 1. Every glue input is committed or computed by the client

A layer's glue may read only:

- the block entry tensor (committed residual),
- outputs of the layer's weight matmuls (opened and row-checked),
- inputs and outputs of its token-mixing ops (committed),
- weights from the claimed GGUF,
- graph inputs the client derives itself: positions, token ids, and constants
  from the GGUF header.

Anything else makes the graph unreceiptable. Without this rule, a node could
open a value nobody committed and choose it so the glue checks out. Examples
in llama.cpp today:

- RWKV7's `v_first`: layer 0's value vector is passed to every later layer
  (`models/rwkv7.cpp`).
- Gemma 3n's per-layer inputs: computed once before layer 0 from the token
  embedding and read by every layer (`models/gemma3n.cpp`).

*Requirements:* security (closes the hole) and honest nodes (the node
refuses up front instead of failing later). A later step can commit such
cross-layer values as extra leaves, so these models become receiptable.

### 2. Only small weights may hide inside token-mixing ops

A token-mixing op can take weights directly, and those weights are never
row-checked. Examples: Mamba-style `ssm_conv1d` kernels into `ssm_conv`
(`models/qwen35.cpp`), and gpt-oss's `attn_sinks` into flash attention
(`models/openai-moe.cpp`).

Rule: weights read by a layer's token-mixing ops may total at most 1% of the
layer's weight bytes. Above that, the graph is unreceiptable. The audit
output lists such weights as unchecked.

*Requirements:* security against a cheaper model means catching substitutions
that save memory or compute. Changing weights that are under 1% of a layer
saves under 1%, so it is not worth doing. This keeps hybrid and
attention-sink models receiptable (honest nodes, cost), and needs no list of
which weights are small (no model code).

### 3. Commit the KV cache rows as stored

In wire version 2 the attention leaf committed k and v before they were
written to the cache. Attention reads the stored rows. A node could store them
at lower precision (`-ctk q4_0`), or store different values, and nothing would
catch it.

Rule: when a token-mixing op reads a KV buffer, its committed inputs are the
rows as stored, in the cache's type, and the receipt names the cache type.
The client recomputes k and v through the glue in f32. It measures how far
its own values move when they are rounded into the named type, and requires
the opened stored row to be no further from its values than 1.5 times that
distance, plus 1e-4 of the row's norm. A client decides which cache types it
accepts.

*Requirements:* security (closes a cheap, real substitution). Cost: the node
hashes the stored rows instead of the f32 values, which is the same number of
bytes or fewer, plus a gather of the ubatch's cells. Honest nodes: the
bound is the client's own rounding error, so it absorbs a GPU and a CPU
rounding the same value to neighbouring steps.

### 4. Recompute matmuls that feed a discrete choice

Opened matmul outputs pass the row check within a tolerance. When glue turns
one into a discrete choice (`argsort` or top-k for MoE routing), a node could
move near-tied values inside that tolerance and pick a different expert. The
client's glue evaluation would agree.

Rule: when a weight matmul's output reaches an `argsort` or top-k op before
any other weight matmul, the matmul is a router. The client recomputes every
row of it from the GGUF, not a random sample, and every opened column must
pass, not a majority. The glue then computes the scores from the opened
router output and checks the opened choice against them: a top-k must hold
the highest scores, and an argsort must be in order. A pick that loses to an
unpicked score by more than 1e-4 of the largest score fails.

A node can still move a router output within the row tolerance (1e-3), so it
can swap experts whose scores are within that distance of each other. Those
are numeric ties: an honest GPU and CPU can disagree about them too.

*Requirements:* security (routing cannot be steered beyond real numeric
ties) and honest nodes (genuine ties are accepted). Cost: router matrices are
small (number of experts × embedding size), so a full recomputation per
opened position is cheap. It is found by graph structure, so no model code.

## Leaves

| Leaf | Found by | Wire version 2 |
|---|---|---|
| Residual entering layer N | Skippy's block boundary records (entry tensor of layer N) | `rms_norm` input in front of `attn_q` |
| Stage output | Exit tensor of the last stage layer | node named `l_out-N` |
| Token-mixing leaf of layer N | Inputs (KV rows as stored) and outputs of the layer's token-mixing ops | attn_q, attn_k, attn_v outputs and attention output, by weight name |
| Opened matmuls | Every weight matmul of layer N | seven matmuls by weight name |

The block boundaries already exist. Model graph builders register a canonical
entry and exit tensor per layer, validated structurally
(`src/skippy/block_boundaries.h`), and stage splitting relies on them.

For attention, the token-mixing leaf holds q after RoPE and the stored k and
v rows, instead of the `attn_q`, `attn_k` and `attn_v` outputs. The
commitment stays about the same size. Those matmul outputs are still opened by
replay and row-checked, and the glue evaluation links them to the committed
values. This changes the attention leaf, so the receipt wire version goes to
3.

## Auditor

The auditor depends on libllama with the CPU backend. Ben accepted this
dependency. The native entry point is `skippy_model_eval_receipt_glue`:

- `skippy audit-receipt` plans and loads layer N of the claimed GGUF alone,
  as a one-layer stage on the CPU (`n_gpu_layers 0`, no op or KV offload).
- It builds the layer graph with the normal model builder, for all opened
  positions in one ubatch. Each weight-matmul output, token-mixing output and
  selection becomes an input that holds the opened value, and the residual
  entering the layer is the opened one. The large weights are never read.
- It computes in f32 and returns every weight-matmul input, the experts of
  each `mul_mat_id`, every token-mixing input, every computed stored row,
  every selection's scores, and the exit tensor.
- The Rust auditor (`glue.rs`) compares them with the opening, with a
  relative-error tolerance of 1e-4 (expert ids must match exactly), and
  checks stored rows and selections by the rules of guards 3 and 4.
  `skippy_receipt_stored_row_error` gives the rounding distance in the cache
  type. The old per-architecture glue code is gone.
- Before comparing, it checks that the opening has the plan of its own
  graph: the same weights, by index and name, the same routers, and the same
  number of token-mixing values and selections.

The node never supplies the glue program. A node that could send its own
program could make any values check out. The client's graph comes from its own
llama.cpp build and the GGUF it already holds for the row checks.

The node runs the same classification when it builds a graph, so it knows
before serving whether it can issue receipts, and refuses them with a reason
if it cannot.

## What this covers and does not

- Covered: any decoder layer llama.cpp builds that passes the guards,
  including MoE (routing is glue, recomputed by guard 4, and the selected
  experts' matmuls are opened and row-checked), biases, gated attention
  outputs, attention sinks, and hybrid or recurrent layers (their state ops
  are token-mixing). The KV-cache storage type is now checked.
- Refused for now: graphs with uncommitted cross-layer inputs (RWKV7, Gemma
  3n) and graphs with large weights inside token-mixing ops. These are refused,
  not failed. Two more limits are outside the classifier: quantized KV caches,
  whose keys llama.cpp rotates with a Hadamard matmul that merges heads and
  tokens into one axis, and recurrent or hybrid memory, which replay cannot
  restore yet. Both are refused with a reason.
- Still unchecked: the arithmetic inside token-mixing ops (attention scores,
  SSM scans), as today. This includes the attention mask: a node could attend
  only to recent tokens and the audit would not notice. The output head and
  the input embedding are not part of any layer.
- Fused kernels on the node do not matter. The client computes unfused on the
  CPU and compares with a tolerance.
- Version skew: the client's llama.cpp must build the architecture in the same
  way as the node's. A model whose graph changes between llama.cpp versions
  can fail glue checks until both sides use the same pin. Receipts should
  carry the llama.cpp pin (planned for milestone 2), so the client can tell
  skew from cheating.

## Costs

Measured with a release build on two-stage splits. Each stage ran on one
RTX 5060 Ti, and the auditor ran on the CPU of the same machine.

- Node: Qwen3-4B Q8_0 decodes 3% slower with receipts on (71.7 → 69.5
  tok/s, median of five runs), as with wire version 2. Prefill is 1.38x
  slower (2292 → 1661 tok/s; 1.28x with version 2): each ubatch now also
  reads back the stored KV rows.
- Auditor, per opening of 16 positions:

  | Model | Layer load | Glue, first evaluation | Glue, per position after | Row checks |
  |---|---|---|---|---|
  | Qwen3-4B Q8_0 | 0.5 to 1.3 s | 76 to 117 ms | 0.04 to 0.27 ms | 21 to 461 ms |
  | granite-3.1-3b-a800m Q8_0 (MoE) | 0.9 s | 99 to 161 ms | 0.03 to 0.27 ms | 0.6 to 1.0 s |

  The first evaluation creates the glue context and builds the graph. Later
  evaluations of the same layer reuse them, and the per-position figure is
  the fastest of five. The row checks include a full recomputation of
  granite's router (40 rows) and reading rows from the GGUF, so they vary
  with the page cache. Fixed costs dominate: one audit costs about 1 to 2
  seconds of CPU time, and the glue evaluation is under a tenth of it.
- A client that cannot load native code (a browser, a small SDK) would need
  another path.
