# Receipts without model-specific code

Status: built. This document describes receipt wire version 7 and Skippy
ABI 0.1.82. The native code is in
`skippy/llama_cpp/patches/model_support/0017-skippy-add-receipt-recorder-and-replay.patch`.
The classifier is `src/skippy/receipts_layers.cpp` of that patch. The
checks of the auditor are in `skippy-receipts` (`glue.rs`, `audit.rs`). The
glue evaluator over libllama is in `skippy-receipt-audit`
(`native_model.rs`). The measured costs are in
[RECEIPTS_MEASUREMENTS.md](RECEIPTS_MEASUREMENTS.md).

Receipts ([RECEIPTS.md](RECEIPTS.md)) had model-specific code in two places
before this design:

1. **Leaf discovery** in the native recorder. It found the receipt tensors
   by GGUF weight names (`blk.N.attn_q.weight`, `ffn_gate`, and others). It
   took the residual from the `rms_norm` before `attn_q`, and found the
   stage output by the node name `l_out-N`.
2. **Glue checks** in the auditor. `glue.rs` implemented the operations
   between the matmuls of one layer again (two RMS norms, SwiGLU, residual
   adds) for the `llama`, `qwen2` and `qwen3` architectures only.

This design removed both. No code names an architecture, a tensor role or
a node name.

## Requirements

- **No model-specific code.** Nothing lists architectures, tensor names or
  node names.
- **Security.** The audit must catch a node that serves a cheaper model
  than it claims: a lower quantization, a smaller model, or skipped work.
  The arithmetic inside attention gets a loose check only.
- **Honest nodes do not fail an audit.** When a stage cannot support a
  check, it issues no receipt and gives a reason. It does not issue a
  receipt that an honest node then fails.
- **Cost.** Receipts stay inside the serving budget of
  [RECEIPTS.md](RECEIPTS.md#parameters). Replay bounds the cost of an
  opening. Snapshots of the KV cache were rejected as too expensive.

## Idea

The computation of each layer for one position has three parts:

- **Weight matmuls.** Each `MUL_MAT` or `MUL_MAT_ID` whose weight operand
  is a model tensor of the layer, with any name. The row checks cover
  these.
- **Token-mixing operations.** Operations where the output of a token
  depends on the values of other tokens. Attention, `ssm_conv`, `ssm_scan`,
  the gated delta net and RWKV's WKV are examples. Their inputs and outputs
  are committed.
- **Glue.** All other operations. With the residual that enters the layer,
  the matmul outputs and the token-mixing outputs, glue is a function of
  one position: norms, RoPE, biases, activations, gating, MoE routing and
  residual adds.

The client checks the glue because it computes the glue itself. It builds
the graph of the layer with its own llama.cpp and Skippy graph builder from
the claimed GGUF. It runs the graph on the CPU, with the opened matmul
outputs and token-mixing outputs in place of those nodes. Then it compares
its results with the opened matmul inputs and the opened residual that
leaves the layer. This covers each architecture that llama.cpp can build,
with ggml's own implementation of each op.

## Decision: which operations mix tokens

An operation mixes tokens when its output column for one token depends on
the column of another token. A dependency pass over the built graph
decides this from what each ggml op does along the token axis:

- Each graph input that holds one column per token (the block entry, token
  ids, positions) starts as per-token.
- A pass in graph order marks each node per-token when its op keeps the
  token columns apart and all its inputs are per-token or weights.
  Element-wise ops, norms over the embedding axis, RoPE, `get_rows` on
  weights, and `mul_mat` against a weight keep the columns apart.
- A node that combines columns of different tokens is token-mixing:
  `mul_mat` between two activations along the token axis, `soft_max` over
  the key axis, `flash_attn_ext`, `ssm_conv`, `ssm_scan`, the delta-net and
  WKV ops, and views or concatenations that shift along the token axis. A
  read of a KV buffer or a recurrent-state buffer is also token-mixing,
  because that buffer holds other tokens.
- The table that says how each ggml op treats the token axis covers ggml
  ops, not models. A node whose op is not in the table makes the graph
  unreceiptable until someone adds the op.

This rule replaced the first proposal, "an op that reads a memory tensor
mixes tokens", for these reasons:

- **Honest nodes.** The memory rule files non-causal attention over the
  ubatch (encoder and embedding models, which have no KV cache) as glue.
  The client cannot compute that one position at a time, so honest nodes
  fail. The structural rule files it as token-mixing.
- **Security.** Both rules need the guards below. The structural rule
  trusts nothing that the memory rule checks.
- **No model code.** The op table is per ggml op, with about a hundred
  entries, and every model uses it. An unknown op stops receipts and does
  not guess.

## Guards

The split alone leaves four holes. Each guard closes one, with graph
structure only.

### 1. Every glue input is committed or computed by the client

The glue of a layer can read only:

- the block entry tensor (the committed residual).
- outputs of the weight matmuls of the layer (opened and row-checked).
- inputs and outputs of its token-mixing ops (committed).
- weights from the claimed GGUF.
- graph inputs that the client derives itself: positions, token ids, and
  constants from the GGUF header.

All other inputs make the graph unreceiptable. Without this rule, a node
can open a value that nobody committed and choose it so that the glue
passes. Two examples in llama.cpp:

- RWKV7's `v_first`: the value vector of layer 0 goes to every later layer
  (`models/rwkv7.cpp`).
- Gemma 3n's per-layer inputs: computed one time before layer 0 from the
  token embedding and read by every layer (`models/gemma3n.cpp`).

A later step can commit such cross-layer values as more leaves, so that
these models become receiptable. That step is not built.

### 2. Only small weights can be inside token-mixing ops

A token-mixing op can take weights directly, and those weights get no row
check. Examples are the `ssm_conv1d` kernels into `ssm_conv`
(`models/qwen35.cpp`), and the `attn_sinks` of gpt-oss into flash attention
(`models/openai-moe.cpp`).

Rule: the weights that the token-mixing ops of a layer read can be at most
1% of the weight bytes of the layer. With more, the graph is
unreceiptable. The audit output lists such weights as unchecked.

Weights that the ops reach through values derived from state count toward
the 1% too.

Security against a cheaper model means detection of substitutions that
save memory or compute. A change of weights that are less than 1% of a
layer saves less than 1%. The rule keeps hybrid models and models with
attention sinks receiptable, and it needs no list of small weights.

### 3. Commit the KV cache rows as stored

Before this design, the attention leaf committed k and v before the cache
write. Attention reads the stored rows. A node was thus free to store them
at lower precision (`-ctk q4_0`), or to store different values. No check
saw it.

Rule: when a token-mixing op reads a KV buffer, its committed inputs are
the rows as stored, in the type of the cache. The receipt names the cache
type. The client computes k and v through the glue in f32. It measures how
far its own values move when it rounds them into the named type. The
opened stored row can be no further from its values than 1.5 times that
distance, plus 1e-4 of the norm of the row. The auditor accepts the cache
types f32, f16, q4_0 and q8_0.

The node hashes the stored rows and not the f32 values. That is the same
number of bytes or fewer, plus a gather of the cells of the ubatch. The
bound is the client's own rounding error, so it absorbs a GPU and a CPU
that round one value to adjacent steps.

### 4. Compute again the matmuls that feed a discrete choice

Opened matmul outputs pass the row check within a tolerance. When glue
turns one into a discrete choice (`argsort` or top-k for MoE routing), a
node can move near-tied values inside that tolerance and pick a different
expert. The glue evaluation of the client then agrees.

Rule: when the output of a weight matmul reaches an `argsort` or top-k op
before another weight matmul, the matmul is a router. The client computes
every row of it from the GGUF, not a random sample, and every opened
column must pass. The glue then computes the scores from the opened router
output and checks the opened choice against them. A top-k must hold the
highest scores, and an argsort must be in order. A pick that loses to an
unpicked score by more than 1e-4 of the largest score fails.

The router flag comes from the plan of the auditor's own graph. It does
not come from the node.

A node can still move a router output within the row tolerance (1e-3), so
it can swap experts whose scores are within that distance. Those are
numeric ties. An honest GPU and CPU can disagree about them too.

## Refusals of the classifier

The classifier fails closed. When it cannot place a node of the graph, the
stage issues no receipt for the graph and keeps the first reason. Every
rule below reads graph structure only.

| Refusal | Reason text contains |
|---|---|
| An op that is not in the op table | `is not in the receipt op table` |
| A weight matmul that reads a value that nothing commits | `weight matmul ... reads ..., which nothing commits` |
| A weight matmul with a batched weight, or one that contracts over tokens | `multiplies by a batched weight`, `contracts over tokens` |
| The experts of a `MUL_MAT_ID` come from a value that nothing commits | `the experts of ... come from a value nothing commits` |
| A KV cache write whose source is not a clean per-token value: a value from inside a token-mixing region, from state, or from an uncommitted value | `cache write ... stores ...` |
| A V cache that is stored transposed (flash attention is off) | `the V cache is stored transposed; receipts need flash attention` |
| A cache write whose row indices are not the cells of the tokens | `uses row indices that are not the tokens' cells` |
| A recurrent state write that orders its rows by an uncommitted value, or that stores one | `state write ...` |
| A per-token value that is copied into persistent state | `a per-token value is copied into persistent state` |
| A token-mixing node that reads a value that is neither persistent state nor committed by the layer | `token-mixing node ... reads ...` |
| A glue node that reads a value that nothing commits and the client cannot compute | `glue node ... reads ...` |
| A token-mixing output that is a view, or that the pass cannot follow | `token-mixing output ...` |
| Token mixing that reaches the layer output without a weight matmul | `token mixing reaches the layer output without a weight matmul` |
| More than 1% of the weight bytes of the layer inside token-mixing ops | `more than receipts leave unchecked` |
| A block exit that is not a per-token value | `the block exit is not a per-token value the client computes` |
| No block boundary, or layers that are not contiguous | `no block boundary`, `the graph's layers are not contiguous` |

A token-mixing node, a weight matmul, a cache write or a state write can
read only three types of value: persistent state, a state selector, or a
value that the layer commits. A state selector is a request input whose
name starts with `attention.` or `recurrent.`. Those inputs hold the masks
and cell indices of attention and the state indices of recurrent ops.

The recorder also refuses a ubatch that it cannot record:

- a token that more than one sequence shares.
- a token whose other position sections are not those of a text token,
  which is the case for image tokens.
- a ubatch of embeddings on a stage that imports no activations.

## Leaves

| Leaf | Found by | Before this design |
|---|---|---|
| Residual that enters layer N | Skippy's block boundary records (the entry tensor of layer N) | The `rms_norm` input before `attn_q` |
| Stage output | The exit tensor of the last stage layer | The node with the name `l_out-N` |
| Token-mixing leaf of layer N | Inputs (KV rows as stored) and outputs of the token-mixing ops of the layer | The `attn_q`, `attn_k`, `attn_v` outputs and the attention output, by weight name |
| Opened matmuls | Every weight matmul of layer N | Seven matmuls by weight name |

The block boundaries existed before receipts. Model graph builders
register one entry tensor and one exit tensor per layer, and
`src/skippy/block_boundaries.h` checks them by structure. Stage splits use
them too.

For attention, the token-mixing leaf holds q after RoPE and the stored k
and v rows. It does not hold the `attn_q`, `attn_k` and `attn_v` outputs.
A replay still opens those matmul outputs, the row checks cover them, and
the glue evaluation links them to the committed values.

## Bindings across layers

The first build of this design checked each challenged layer alone.
Nothing tied the layers to each other across stages, to the tokens, or to
the layer count of the claimed model. These bindings close that:

- **Tokens.** On the stage that holds layer 0, the residual leaf of layer
  0 hashes the token of the position with the residual. The receipt lists
  the token of every position, the first generated position, and the last
  sampled token, which no ubatch computes.
- **Input of layer 0.** Every opening shows the residual that enters layer
  0 at all picked positions, with its token and path. The auditor computes
  it from the token with layer 0 of the claimed GGUF
  (`skippy_model_eval_receipt_entry`, which stops after the embedding).
- **Stage boundaries.** At each picked position, each stage opens the
  residual leaf where it receives its input and the leaf where it hands
  over. The leaf that one stage sent must be the leaf that the next stage
  received, and both must be in their trees.
- **Layer coverage.** The stages must cover layers 0 to the executable
  layer count of the claimed GGUF, in order. The auditor reads that count
  from the GGUF metadata, not from the receipt.
- **More layers per challenge.** A challenge picks four layers by weight
  bytes, with four positions in each, half from the generated positions.
  Each stage replays one time for all of its picked layers. The row fail
  rule judges each weight role across the opened layers, because four
  columns per matrix are too few for the `2 + n/32` allowance.
- **Response.** The response must be the text of the sampled tokens. The
  output head check shows that the head gives those tokens.

## Attention with a recorder

A context with a receipt capture is a stage context with a recorder, a
replay scratch context, or an evaluation context of the auditor. Two rules
apply to it.

**Dense-mask attention.** Upstream llama.cpp can give the attention op a
list of rows per sequence for a unified KV cache
(`ggml_flash_attn_ext_rows`). A context with a receipt capture turns that
off. It uses the dense-mask attention, where the mask hides the cells of
other sequences. The classifier, the replay and the attention opening do
not read the row lists. Both methods attend to the same cells. A stage
that serves more than one sequence from a unified cache loses the speed
of the row lists. A part of the decode cost of receipts comes from this.
Native support for the row lists is not built.

**f32 accumulation.** A recorder asks every attention op to accumulate in
f32, except on Vulkan. On the CPU backend, the kernel for a ubatch of 2 to
63 tokens summed f16 V rows in f16. Over a long cached context, its output
moved far enough from the exact value to fail the attention check of an
honest node. On Vulkan, the recorder does not ask for f32 attention,
because that switches the attention pipeline, and no receipt was measured
with it. A recorder on Vulkan asks every weight matmul to accumulate in
f32. Since 2026-11-10 no Vulkan stage starts a recorder, because Vulkan
has no rounding profiles, so this native code does not run. The numbers
are in
[RECEIPTS_MEASUREMENTS.md](RECEIPTS_MEASUREMENTS.md#attention-accumulation-on-the-cpu).

## Auditor

The auditor uses libllama with the CPU backend. This dependency is an
accepted cost of the design. The native entry points are
`skippy_model_eval_receipt_entry`, `skippy_model_eval_receipt_glue` and
`skippy_model_eval_receipt_region`:

- `skippy audit-receipt` plans and loads layer N of the claimed GGUF
  alone, as a one-layer stage on the CPU (`n_gpu_layers 0`, no op or KV
  offload). It builds the layer with the cache types of the opened rows.
- It builds the layer graph with the normal model builder, for all opened
  positions in one ubatch. Each weight-matmul output, token-mixing output
  and selection becomes an input that holds the opened value. The residual
  that enters the layer is the opened one. The large weights are not
  read.
- It computes in f32. It returns every weight-matmul input, the experts of
  each `mul_mat_id`, and every token-mixing input. It also returns every
  computed stored row, the scores of every selection, and the exit tensor.
- The Rust auditor (`glue.rs`) compares them with the opening, with a
  relative tolerance of 1e-4. Expert ids must be equal. Stored rows and
  selections follow the rules of guards 3 and 4.
  `skippy_receipt_stored_row_error` gives the rounding distance in the
  cache type.
- Before it compares, it makes sure that the opening has the plan of its
  own graph: the same weights by index and name, the same routers, and the
  same number of token-mixing values and selections. Each matmul index is
  there exactly one time per leaf. Every position that keeps its row opens
  every weight matmul. The evaluator computes an unopened matmul as zeros.
  Without this rule, a node can skip a matmul (a full FFN, for example)
  and commit the residual that the skip gives.
- Merkle paths must have the shape of the tree that the receipt names. The
  `leaf_count` of the stage must be its positions times its slots, and each
  path has a sibling exactly where that tree has one. Without this rule, a
  node can build a tree in which one index opens two different leaves.
- Only the final layer of the model can open a position without a residual
  out, for a row that it did not output. All other layers must open it.

The evaluators and `skippy_receipt_stored_row_error` check every argument:
ids, lengths, positions, tokens and the length of a stored row. A
malformed opening returns an error and cannot abort the process of the
auditor. Some errors show only when the graph runs, for example a mixing
value of the wrong length. The auditor reports those as "could not run
here", not as a failure of the node.

The node does not supply the glue program. A node that sends its own
program can make any values pass. The graph of the client comes from its
own llama.cpp build and from the GGUF that it holds for the row checks.

The node runs the same classification when it builds a graph. So it knows
before it serves whether it can issue receipts, and it gives a reason when
it cannot.

## What this covers and does not cover

**Covered.**

- Each decoder layer that llama.cpp builds and that passes the guards.
  This includes MoE: routing is glue, guard 4 computes it again, and the
  matmuls of the selected experts are opened and row-checked.
- Biases, gated attention outputs and attention sinks.
- The storage type of the KV cache.
- The input embedding, through the input of layer 0.
- The output head, by a check of its own (see
  [RECEIPTS.md](RECEIPTS.md#verification), check 8).

**Hybrid models.** A hybrid model has attention layers and recurrent
layers, for example Qwen3.5. Its stages issue receipts.

- The state ops of a recurrent layer are token-mixing. The classifier
  follows the state write that a recurrent op makes with `SET_ROWS`.
- The stage commits the digest of the recurrent state at the start of each
  turn, and each ubatch commits the state that it read.
- For the first attention pick of a stage on a recurrent layer, the auditor
  runs the token-mixing ops of the layer over the opened region
  (`skippy_model_eval_receipt_region`). The tolerance is 1e-3.
- Gap: a turn that continues a cached prefix gets no receipt on a hybrid
  stage. The stage restores the prefix with a KV page import and a state
  read, and the recorder cannot replay those.
- A model with recurrent memory and no attention gets no receipts.

**Quantized KV cache.** A stage with a q8_0 or q4_0 KV cache issues no
receipts.

- An honest stage fails the loose attention check. On CUDA with a q8_0
  cache, the partials of a prompt position combined up to 0.11 from the
  output, 5.5 times the limit, and 65 of 290 honest turns had a failed
  pick. See [RECEIPTS_MEASUREMENTS.md](RECEIPTS_MEASUREMENTS.md).
- The recorder of such a stage does not start, so the stage serves
  without receipts. A turn that stored quantized rows is refused again
  when it commits. A client reports no audit for a receipt that names a
  quantized cache type.
- The replay and the attention check still follow what llama.cpp does for
  a quantized cache. It rotates q, k and v before the cache stores them,
  and rotates the output back. With a quantized K cache, the kernel rounds
  q to q8_0 blocks of 32 before its dot products with K. What is missing
  is a model of the error of the stored rows at prompt positions.

**Native MTP.** A stage with a native MTP head issues receipts. The trunk
checks every draft, so the receipt covers the trunk and not the head.

**Attention.** The arithmetic inside attention gets a loose check: the
partials of one head at two positions per picked layer, and two drawn
blocks. The check follows causal, sliding-window and chunked attention,
logit soft-capping, attention sinks, the rotation of a quantized cache and
a gated output. It does not follow these cases:

- attention with positional biases (ALiBi)
- a mask or a shape that it does not know
- an output that passes through other ops before it leaves the
  token-mixing region
- K or V rows that the layer does not store

Such a layer is reported as not checked, and the audit then ends with no
audit.

**Not covered.**

- Graphs with uncommitted cross-layer inputs (RWKV7, Gemma 3n), and graphs
  with large weights inside token-mixing ops. The stage issues no receipt
  for them.
- The arithmetic inside recurrent ops, except for the one opened region
  per stage.
- The cached prefix of a turn, except at the two positions that a
  challenge picks. The client checks those like positions of the turn.
- Image tokens, DSA models, and draft-model speculation.

**Fused kernels.** Fused kernels on the node do not matter for the glue
check. The client computes without fusion on the CPU and compares with a
tolerance.

**Version skew.** The llama.cpp of the client must build the architecture
as the node's does. A receipt names its runtime: the Skippy ABI, the
llama.cpp commit and a hash of the patch queue. When the runtime of the
auditor differs, the result is no audit, not a failure.

**Rounding profiles.** The row checks use rounding profiles that model the
kernels of one llama.cpp pin. A new pin can change a kernel. The
maintenance rule is in [RECEIPTS.md](RECEIPTS.md#parameters).

## Costs

The costs of the node and of the auditor are in
[RECEIPTS_MEASUREMENTS.md](RECEIPTS_MEASUREMENTS.md): the serving cost of
the current build, and the history of the auditor cost. A client that
cannot load native code, for example a browser or a small SDK, needs
another method. None is built.
