# Inference receipts

An inference receipt lets a client check that a split-serving node ran the
model it claims to run. This feature is experimental and off by default.

With receipts on, each stage commits to the residual stream entering every
layer it runs, and to each layer's token-mixing values: the inputs of its
attention (or other token-mixing op), the KV rows as stored in the cache, and
its outputs. The stage that holds layer 0 also binds each position's token
into the leaf of the residual entering layer 0. At the end of a turn, each
stage commits to these values with a BLAKE3 Merkle root, and the receipt
lists the turn's tokens and the position where generation started.

The client names a VRF key in its request, and stage 0 binds that key into
a turn root over the whole receipt and signs the root with its node key.
The client's challenge comes from the VRF of that root under its key, so
neither side chooses it: four layers, weighted by their weight bytes, and
four token positions in each, half of them generated positions. To open them,
each stage replays the turn once in a scratch context and reports every
weight matmul of its picked layers. The client checks that the replay
matches the committed values, and recomputes random rows of each matmul from
the claimed GGUF, drawn from the VRF of the opening as it arrived. Then it
computes each picked layer itself, on the CPU with
libllama, from the opened matmul and token-mixing outputs, and compares the
result with the opening. Every opening also shows the input of layer 0 at
the picked positions, which the client recomputes from the tokens, and the
residual where each stage hands over to the next, which must be the same on
both sides. The client checks that the response is exactly the text of the
tokens the receipt says were sampled. Last, it checks attention loosely: at
two picked positions per layer, the node opens one head's attention as
partial sums over blocks of the context, which must combine to the committed
attention output, and then the K and V rows of blocks the client draws, from
which it recomputes those blocks. A node that serves a different
quantization, or a different model, fails these checks.

The receipt tensors are found from the structure of the graph llama.cpp
builds, so no code lists architectures or tensor names. The design is in
[RECEIPTS.md](../../mesh/docs/design/RECEIPTS.md) and
[RECEIPTS_GLUE.md](../../mesh/docs/design/RECEIPTS_GLUE.md), and the
remaining milestones are in
[RECEIPTS_PLAN.md](../../mesh/docs/design/RECEIPTS_PLAN.md).

Do not confuse inference receipts with the generation receipt in
`skippy-serving`. A generation receipt is a digest of the generated token ids
for local generation hooks. It does not prove which model computed them.

## Requirements

- Binary split serving (`skippy serve --stage-transport binary`), or a single
  stage, which commits only turns whose request names a client key. Mesh
  records receipts on the models it serves alone and on its splits when
  `[receipts] enabled` is on.
- Weight matmuls, dense or MoE, with F32, F16, BF16, Q8_0, Q4_0, Q5_0, IQ4_NL,
  IQ4_XS or K-quant (Q2_K to Q6_K) weights. A receipt with other weight types
  gets no audit, because no rounding profile exists for them yet.
- A graph the receipt classifier accepts. Each op must be in its per-ggml-op
  table, a layer may read nothing from another layer except the residual
  stream, and weights that token-mixing ops read directly (such as attention
  sinks) may be at most 1% of the layer's weights. Those are listed as
  unchecked in the audit. A stage whose graph is refused serves normally, and
  its turns get no receipt; `GET /receipts/{id}` returns the reason.
- A plain KV cache with one stream, flash attention (a transposed V cache is
  refused). A stage with a native MTP head issues receipts: the trunk
  verifies every draft, so the receipt covers the trunk and not the head.
  A hybrid model (attention layers with Mamba or gated delta net layers,
  such as Qwen3.5) issues receipts; see [Hybrid memory](#hybrid-memory). A
  model with recurrent memory alone is refused when the stage starts. A
  quantized KV cache (tested with q8_0) issues receipts. llama.cpp rotates
  q, K and V before the cache stores them and rotates the output back; the
  rotation is a graph input, so the replay and the attention check use the
  stage's own values. The auditor builds each layer with the cache types
  the opened rows were stored in.
- The auditor needs a local copy of the GGUF that the node claims, and loads
  each challenged layer of it, and layer 0, on the CPU.

## Serve with receipts

Start stage 0 with `--receipts-listen`, which serves the receipts API that
clients and auditors use. Start every later stage with `--receipts`. Each
stage reaches the receipts of the stage after it over the binary stage
connection: a dedicated connection to the next stage's binary address that
carries only receipts messages, and holds no lane.

```bash
skippy serve --stage-transport binary --worker-only --config plan/stage-1.json \
  --receipts --receipts-claimed-model Qwen3-4B-Q8_0

skippy serve --stage-transport binary --config plan/stage-0.json \
  --bind-addr 127.0.0.1:9337 \
  --receipts-listen 127.0.0.1:9501 \
  --receipts-claimed-model Qwen3-4B-Q8_0
```

A later stage can also serve its own receipts API with `--receipts-listen`.
The stage before it then uses that API instead of the stage connection when
it is started with `--receipts-downstream URL`.

| Flag | Purpose |
|---|---|
| `--receipts` | Turn receipts on without a receipts API: a later stage |
| `--receipts-listen ADDR` | Turn receipts on and serve the receipts API on `ADDR` |
| `--receipts-downstream URL` | Receipts API of the next stage. Commits and openings for its layers go there. Without it, they go over the stage connection |
| `--receipts-claimed-model NAME` | Model named in each receipt. The default is the stage model id |
| `--receipts-allow-remote` | Permit a `--receipts-listen` address that is not loopback |
| `--receipts-signing-key FILE` | File holding the hex Ed25519 seed that signs turn roots. Without it, the stage signs with a key it makes at startup |

A request gets a bound receipt when it names the client's VRF key in the
`x-receipt-client-key` header (64 hex digits). It can also name the turn
root of the receipt it continues in `x-receipt-parent`, which the receipt
then binds, so a conversation's receipts form a chain. Stage 0 binds the
client key, the parent, the runtime (the `skippy-abi` version, and the llama.cpp commit and patch
queue the native library was built from), the sampler,
its node key and every field of the receipt into the turn root, and signs
it. The sampler is `greedy` when every token was the best logit (sampling
off, or temperature 0 without penalties, logit biases, a grammar or a
reasoning budget). Otherwise it is `sampled` followed by a JSON object of
the sampler's parameters and seed, with a grammar named by its BLAKE3 hash
and the reasoning budget's tokens. Each stage
commitment also names the first position of every ubatch of the turn, and,
for a turn that started from cached KV cells, its prefix root. A client
that keeps a conversation's receipts checks that each names the previous
turn root as its parent, for the same client key and node
(`turn::verify_chain`). A request without a client key gets an unbound receipt,
which the auditor rejects.

A bound receipt also rides in the response, so a client that cannot reach the
receipts API still gets it: in the `receipt` field of a chat or text completion
body, and of the last chunk of a stream, the one that carries `finish_reason`.
An unbound receipt is only served by the API.

The receipts API refuses a non-loopback address unless you also pass
`--receipts-allow-remote`. An opening contains activations that come from the
prompt, so anyone who can reach the API can learn about the prompts it served.
Expose it only to auditors that you trust with the prompts.

Each stage keeps leaf hashes and a replay log, not activations. Per position,
that is `2 · layers + 1` hashes (about 1.2 KB for 18 layers) plus the Merkle
tree above them. Per ubatch, the log holds the tokens, positions, output rows,
KV cells and n_kv, and, on a stage after the first, the stage input rows. The
stage input row is the largest part: one hidden-state row per token. Committed
turns are kept up to 1 GiB per stage. The oldest are removed first, and a
removed turn can no longer be opened.

A replay runs only the turn, on top of the KV cells the turn read below its
first position: the sequence's earlier turns, or a prompt prefix that the
prefix cache shared with it. The receipt binds those cells by a prefix root,
and an opening copies them from the live cache, so it fails once another
request has reused one of them. A turn gets no receipt when its positions
are shifted, when cells are loaded from saved state, or when the cache is
compacted. With receipts on, a repeated greedy request does not reuse the
tokens an earlier request generated: the turn computes them again.

A Mesh node serves receipts when its config turns them on:

```toml
[receipts]
enabled = true
```

The node then commits bound turns of the models it serves, alone or as the
coordinator of a split, and signs their turn roots with its node key, the Ed25519 key
of its mesh identity. It serves no receipts address: a client takes the
receipt from the response and opens it on the OpenAI address, below. On a
split, the coordinator asks every later stage to record its part, and
commits and opens it over the stage connection, across peers. A
request without a client key takes no receipt work.

A model this node serves alone, or a split, names its model as
`repo@commit/file` when the GGUF is a Hugging Face file, so a client can
audit the turn without the model. A split GGUF
(`name-00001-of-00003.gguf` and its siblings) is named by its first shard.

A Mesh node audits the turns its peers serve its clients when its config
turns that on:

```toml
[receipts]
audit = true
```

Every chat completion a local client sends through the node to a peer then
names a fresh receipt key, unless the client named one itself. A peer that
issues receipts binds the turn to that key and returns the receipt with the
response, which the node reads as it relays it, streamed or not. The VRF of
the turn root picks about one turn in ten, which one worker thread audits
off the request path, as `audit-receipt` does: it opens the challenge on the
peer over the mesh, reads the claimed Hugging Face GGUF by range, and checks
the opening on the CPU. The response check runs only for a request without
stop strings, tools, a response format or several choices, and an answer
without tool calls, because those change the text.

A failed audit blocks the peer for seven days (`requested_by: receipts` in
`peer_blocks.json`), with the failures and the path of the evidence as the
block's reason. The evidence is kept under `receipt-evidence/` in the node's
state directory, for `skippy recheck-receipt`. A turn the node cannot audit
is no audit, logged with its reason: no receipt, a model that is not one
Hugging Face file, a check that could not run here, a worker still busy
with an earlier audit, or a peer that did not answer the challenge.

A model a Mesh node serves with receipts is gossiped with a `receipts`
flag, so a client can require them. A chat completion sent with
`x-mesh-trust: receipts`, or every chat completion when the node sets
`[receipts] require = true`, goes only to this node's own receipted runtime
or a peer that advertises receipts for the model, and its turns are audited
even when `audit` is off. A request that no such runtime serves fails with
`trust_policy_unsatisfied`, as does any other inference endpoint, because
receipts cover chat completions only. See
[Choose a specific inference provider](../../mesh/docs/PROVIDER_SELECTION.md).

### Hybrid memory

A recurrent layer (Mamba, the gated delta net) keeps one state per
sequence, and each ubatch writes over it. After a turn, the stage no
longer holds the state the turn started from, so a replay cannot rebuild
it from the KV cache as it rebuilds attention. Before the first ubatch of
a turn, the stage reads that state from its recurrent memory. The stage
commitment names the state's BLAKE3 digest in `recurrent_state`, and the
stage keeps the state while a client can challenge the turn. An opening's
replay starts from that state and runs every ubatch of the turn. A turn
that no client can challenge drops its state when it is committed.

A stage that issues receipts does not restore cached prefixes, so each
turn starts from an empty state, which is 300 bytes. A turn that started
from a full state would hold about 25 MB per stage for Qwen3.5-9B. The
client checks the projections, the FFN and the attention layers of a
hybrid model. It does not check inside the recurrent mixers yet: the
opened mixer output is used as committed.

## Audit a response

`skippy audit-receipt` makes a fresh VRF key, sends one chat request that
names it, gets the receipt, draws the challenge from the VRF of its turn
root, and checks the opening:

```bash
skippy audit-receipt \
  --api http://127.0.0.1:9337 \
  --claimed-model /models/Qwen3-4B-Q8_0.gguf \
  --prompt "Name three rivers in Europe."
```

Without `--receipts`, the command takes the receipt from the response and
opens it on the OpenAI API (`POST /v1/receipts/open`), which a Mesh node
serves. The challenge names the model and the receipt's node key in
`x-mesh-target`, so the mesh sends it to the peer that signed the receipt.
Pass `--receipts URL` to open on a stage's receipts API instead, or to audit
a node whose responses do not carry the receipt.

`--claimed-model` takes a local GGUF or a Hugging Face file as
`repo@revision/file`, for example
`unsloth/Qwen3-4B-GGUF@main/Qwen3-4B-Q8_0.gguf`. A split GGUF is named by
its first shard; the command reads every shard. The command never downloads a
Hugging Face file. It reads only the bytes the checks use, with range
requests, into a sparse copy under `~/.cache/mesh-llm/receipts/claimed/`.
These bytes are the header, the small tensors of the challenged layers, the
checked rows, the embedding rows of the challenged tokens, and the output
matrix. The copy is keyed by the commit the revision resolves to. Later audits
fetch only the bytes the copy does not hold yet. `HF_ENDPOINT` and `HF_TOKEN`
apply, and the token goes only to the Hugging Face host, not to the CDN. The
header is read before the request, so it does not delay the challenge.

The output matrix is read once, to build the head key (check 5), and the copy
does not keep it. Before the head key, the copy held 405 MiB of the 4.0 GiB
Qwen3-4B Q8_0 file, almost all of it the output matrix. The first audit took
19 s from ben-ai-server, and later audits about 9 s.

Use `--layer N` to challenge one specific layer instead of the drawn ones (the
report then says the challenge was picked by hand and is not evidence), and
`--max-tokens N` to change the response length (default 32). The command exits
with a non-zero status when the audit fails. Use `--output json` for a
machine-readable report.

Some receipts this machine cannot audit: the node ran another `skippy-abi`
version than this verifier, whose glue evaluation would then not reproduce the
node's graph; a stage ran on a backend without rounding profiles (anything but
CPU and CUDA); a stage multiplies a weight type without a profile (see check
3); or the output matrix has a type the verifier cannot decode. The command
checks this once the receipt is bound and covers the model, before it
challenges, and reports `NO AUDIT` with the reason instead of a verdict
(`"outcome": "no_audit"` in JSON), with a non-zero exit status. No audit is
not a failure: it says nothing about the node, so it must not withhold
payment or block the node. A node whose receipts are never auditable gets no
passes either.

Use `--timing-runs N` to repeat each layer's glue evaluation N times and
report the fastest as the client cost.

The audit checks, in order:

1. The receipt is bound to the auditor's key, its turn root matches its
   fields, and the node key's signature of the root verifies. Its stages
   cover every layer of the claimed GGUF from 0, in order, each with the
   leaf count its positions and layers make, and the receipt holds a token
   for every position. The layer count comes from the GGUF's metadata, not
   from the receipt.
2. The node's signature of the opening's bytes verifies: the opened matmul
   columns come from a replay and are in no tree. Every opened leaf is in
   its stage's tree, at the place its position and
   layer give. At each stage boundary, the residual the lower stage sent and
   the residual the upper stage received are the same leaf.
3. Each picked layer: weight rows and glue, described below. The rows come
   from the VRF of the opening's bytes, and each matrix is checked with the
   activation rounding profiles of the backend its stage names (CPU or
   CUDA) for its weight type. The weight types with profiles are f32, f16,
   bf16, q8_0, q4_0, q5_0, iq4_nl, iq4_xs and the K-quants q2_K to q6_K. A
   backend or weight type without a profile (q4_1, q5_1 and the other
   i-quants, for example) cannot be checked, so such a receipt gets no
   audit. For q4_K and q5_K, CUDA's prefill kernels (MMQ) apply each
   minimum of 32 weights to the f16 sum of the unquantized activation, and
   the MMA kernels round the weight's scale and minimum to f16, so the
   check recomputes those sums the same way rather than dotting the
   dequantized row.
4. The input of layer 0 at every picked position is what layer 0 of the
   claimed GGUF computes from the token the receipt names.
5. At four positions the turn sampled a token from, drawn from the same VRF
   output, the final stage opens the residual leaving the last layer and
   the logits its replay computed from it. The auditor checks those logits
   with its secret head key. The key holds four random ±1 vectors `r` over
   the vocabulary and `rᵀW` for the output matrix `W` (`output.weight`, or
   the token embedding when the model ties them). For each vector, `r · y`
   must equal `(rᵀW) · x` for the logits `y` and the normed residual `x`
   with the backend's rounding profile, within 1e-5 of the logits' norm.
   Honest Qwen3-4B Q8_0 logits on CUDA come within 1.5e-7. The node does not
   know `r`, so it cannot change a logit by more than the tie band unseen.
   The auditor also checks 16 random rows of `W` and the sampled token's row
   like any matmul. Then it checks that the sampled token is the greedy
   choice of the opened logits. A token whose logit is within 1% of the
   logits' root mean square of the best one is accepted and counted as a
   tie; more than one tie in four positions is flagged, not failed.

   The node normalizes the residual in f32, and the auditor in f64. At about
   one position in 150, a value of `x` lies within a few ulps of a q8 step
   boundary, and the node rounds it to the other step. Its logits are then
   one step times that column of `W` away, some 4e-4 of their norm, though
   it computed honestly. So the auditor lists the values of `x` that move to
   another step, or change their block's scale, when they move by 8 f32
   ulps: under one per position with q8, about ten with f16. It accepts the
   one or two whose effect on `(rᵀW) · x` explains the differences of all
   four vectors, and checks the rows with them. This gives the node no
   freedom: a flip explains the differences only when the logits moved by
   exactly that column of `W`.

   The first audit of a model builds the key in one pass over `W` and keeps
   it under `~/.cache/mesh-llm/receipts/head-keys/`, readable by its owner
   only (107 KiB for Qwen3-4B). A sparse copy does not keep the matrix. The
   check then takes about 10 ms, and the client never computes every
   logit. A model that divides its logits by a `logit_scale` (Granite) is
   checked at that scale: the auditor finds which of 1, the scale and its
   inverse the opened logits use, within 1e-3, and checks them unscaled.
   Against a node that opens no logits, or a model that soft-caps its
   logits, the auditor computes the whole head as before. So does
   `recheck-receipt`, which has no key.

   A greedy turn's token must be the best logit. A sampled turn's token
   must be one its stated filters keep: `top_k`, `top_p`, `min_p` and the
   temperature, in the stated order, after the logit biases, with the tie
   band as slack. The check does not replay the draw. A turn with
   penalties, DRY, XTC, typical or top-n-sigma sampling, mirostat, a
   grammar or a reasoning budget gets no token judgement, because those
   depend on more than the logits; its logits are still checked. Without
   opened logits, a model that scales or caps them gets none either.
   `audit-receipt` asks for greedy decoding, so there a receipt whose
   sampler is not `greedy` fails.
6. The response (its reasoning and content together) is the text of the
   sampled tokens, up to whitespace and reasoning markers, so a response that
   drops or adds text fails, and so does an empty one. The prompt is in the
   text of the tokens before them; the chat template adds text around it, so
   that is a containment check.
7. Attention, loosely. The challenge also picks two of each picked layer's
   positions and a head for each, from the VRF of the turn root. With the
   opening, the stage that holds the layer reads the K and V rows of every
   position the query attends to from its KV cache. It computes the head's
   flash-attention partials (running max, sum and weighted value) over
   blocks of 64 of those positions, in f64 on its CPU, from the committed
   q. The auditor combines the partials and compares them with the head's
   slice of the committed attention output, within 2e-2. Then it draws two
   blocks per pick from the VRF of the opening as it arrived, which the
   node cannot know when it opens the partials, and asks for their rows
   (`POST /receipts/rows`). Each row must be in the receipt: through its
   position's mixing leaf when the turn computed it, or through the cell
   digest and the prefix root when the turn read it from cache. The auditor
   recomputes each drawn block's partial from the committed q and those
   rows, in f64, and compares it with the node's, within 1e-6 of the
   partial's largest value. A node that attends over other rows, or
   computes another softmax, fails unless it changes no more than the
   tolerance allows. Causal, sliding-window and chunked attention, logit
   soft-capping and attention sinks are followed; a layer whose attention
   the check cannot follow is reported as not checked, which says nothing
   about the node.

   The rows are read from the live cache, so the opening fails once the
   cells no longer hold what the turn wrote or read. Serving frees a turn's
   cells when its response ends, but a freed cell keeps its rows until a
   ubatch writes it, and each stage pins the cells of a turn a client may
   still challenge: a later request writes other cells while it fits
   there. A turn bound to a client key is pinned for 12 seconds (the
   client's 10-second challenge window and 2 seconds of travel), and once
   opened, until the client has its rows, for at most 30 seconds. An
   unbound turn is not pinned, since no client can challenge it. A busy
   stage whose cache is full writes pinned cells anyway, and the turn can
   no longer be opened.

   A pinned cell widens the cache's attention window for every later
   request, which reads it as a masked row. On back-to-back 256-token
   Qwen3-4B requests on an RTX 5090, all of them bound, decode ran at
   117.5 tok/s against 119.5 with receipts off (1.7% slower); pinning
   every turn for 30 seconds had cost 4.5%. Unbound requests, and the
   second turn of a conversation that continues the first, ran as fast as
   with receipts off.

An honest node that serves the claimed Q8_0 GGUF, both stages on one RTX
5090 (release build):

```text
receipt 6d1cad88-8bbf-4af8-a8aa-a48daccd8fb6  model Qwen3-4B-Q8_0  positions 0..117  generation from 24
  stage 0  layers  0..18  root 1ac8469168711144…  leaves 4329  cache f16  (CUDA0)
  stage 1  layers 18..36  root c4f534abd4b072f2…  leaves 4329  cache f16  (CUDA0)
  turn root 917b98ace8feab12…  node key 15f6df776731a7c0…  client key f102d562e4c3fabd…  skippy-abi 0.1.87; …  sampler greedy
challenge: layer 11 (stage 0), positions [19, 22, 66, 96]
challenge: layer 19 (stage 1), positions [0, 4, 49, 114]
challenge: layer 22 (stage 1), positions [15, 19, 57, 87]
challenge: layer 31 (stage 1), positions [6, 8, 31, 76]
opening: 0.28 s
receipt binding and signature, shape, merkle paths and stage boundaries: ok
layer 11:
  blk.11.attn_k.weight           q8_0    4/4   pass  max err 1.8e-4  q8_32_div_dh
  blk.11.attn_output.weight      q8_0    4/4   pass  max err 2.6e-4  q8_32_div_dh
  ...
  blk.11.ffn_up.weight           q8_0    4/4   pass  max err 2.1e-4  q8_32_div_dh
  glue matmul inputs   28/28  pass  max err 1.8e-7  ok
  glue mixing inputs    4/4   pass  max err 1.5e-6  ok
  glue stored rows      8/8   pass  max err 0.0e0  ok
  glue residual out     4/4   pass  max err 0.0e0  ok
  client cost: layer load 0.61 s, glue 26.50 ms per position (4 positions, first run 106 ms), row checks 5 ms
layer 19:
  ...
weight rows, each role across the opened layers:
  attn_k.weight             16/16  pass  ok
  ...
  ffn_up.weight             16/16  pass  ok
input of layer 0 from the receipt's tokens: 15/15 pass  max err 0.0e0  ok
sampled tokens at [28, 33, 51, 65] are the greedy choice: 4/4 pass  ties 0  ok  (0.03 s, node's logits checked with the head key)
attention, per picked head:
  position 66 layer 11       head 15  67 positions  combine 2.5e-7  blocks [0, 1] 2.9e-8  ok
  position 19 layer 11       head 31  20 positions  combine 2.6e-4  blocks [0] 4.1e-8  ok
  ...
  position 76 layer 31       head 24  77 positions  combine 5.5e-7  blocks [0, 1] 2.7e-8  ok
  rows opened in 0.02 s
response is the text of the sampled tokens: ok
prompt is in the receipt's tokens: ok
AUDIT PASS
```

A MoE layer (granite-3.1-3b-a800m) adds the router, which is checked on every
row, the experts each matmul used, and the router's selection. The expert
matrices are checked at 16 (position, expert) pairs per layer, drawn with the rows' VRF output
from the pairs the opened positions routed to, so the
row check reads 16 experts' rows instead of every routed expert's. This output
is from an earlier build that opened one layer at 16 positions:

```text
  blk.5.ffn_down_exps.weight     q8_0  128/128 pass  max err 3.6e-4  q8_32_div_dh  ok
  blk.5.ffn_gate_inp.weight      f32    16/16  pass  max err 2.7e-7  f16  ok  (router, all rows)
  ...
glue (the layer computed here on the opened values):
  matmul inputs  128/128 pass  max err 1.6e-7  ok
  expert ids      48/48  pass  max err 0.0e0  ok
  mixing inputs   16/16  pass  max err 3.2e-6  ok
  stored rows     32/32  pass  max err 0.0e0  ok
  selections      16/16  pass  max err 0.0e0  ok
  residual out    16/16  pass  max err 9.4e-8  ok
```

The same Qwen3-4B split serving Q6_K while the receipts claim Q8_0:

```text
challenge: layer 0 (stage 0), positions [17, 22, 32, 36]
challenge: layer 7 (stage 0), positions [5, 15, 45, 48]
challenge: layer 18 (stage 1), positions [13, 17, 23, 30]
challenge: layer 20 (stage 1), positions [3, 9, 23, 32]
receipt binding and signature, shape, merkle paths and stage boundaries: ok
layer 0:
  ...
  glue residual out     4/4   pass  max err 0.0e0  ok
...
weight rows, each role across the opened layers:
  attn_k.weight              0/16  pass  FAIL
  ...
  ffn_up.weight              0/16  pass  FAIL
input of layer 0 from the receipt's tokens: 0/13 pass  max err 1.9e-2  FAIL
response is the text of the sampled tokens: ok
prompt is in the receipt's tokens: ok
AUDIT FAIL
```

The Merkle paths and the glue checks still pass in the second case: the node
committed to what it computed, and its replay reproduced it. The row checks
fail because those values do not come from the claimed weights, and the
input of layer 0 fails because the Q6_K token embeddings are not the claimed
Q8_0 ones. The row
tolerance is 1e-3; honest CUDA and CPU errors are about 1e-4 to 1e-3. The glue
tolerance is 1e-4, relative to the largest value of the reference column;
honest errors are about 1e-7 to 3e-6.

### What the glue checks bind

A replay recomputes the challenged layer, so its matmuls would pass the row
checks even if different weights served the turn. The glue check ties the
replayed matmuls to the committed values instead. The auditor builds the
challenged layer with its own llama.cpp from the claimed GGUF, and replaces
every weight matmul output, token-mixing output and selection with the opened
value, and the residual entering the layer with the opened one. Then it
computes the layer and compares:

- each weight matmul's input, and the experts of each MoE matmul;
- each token-mixing input, such as q after RoPE;
- each stored KV row. The opened row is in the cache's type; it may be no
  further from the auditor's f32 value than 1.5 times the auditor's own
  rounding of that value into the cache type, plus 1e-4 of the row's norm;
- each selection (a top-k must hold the highest scores, an argsort must be in
  order), against scores computed from the opened router output;
- the residual leaving the layer.

The node must also show that its replay reproduces every committed leaf of the
challenged positions. The served token-mixing values are committed, so the row
checks of the matmuls that produce them test the served values. The glue
check does not look inside attention or another token-mixing op; check 7
checks attention loosely. The audit first checks that the opening has the
plan of the auditor's graph: the same weights by index and name, the same
routers, the same number of token-mixing values and selections, and every
weight matmul at every position that keeps its row. Merkle paths must fit
the tree the receipt names, and only the model's final layer may leave out a
position's residual out.

The row fail rule judges each weight role (`attn_q.weight`,
`ffn_down_exps.weight`, ...) on its columns in every opened layer together.
A role fails when more than `2 + n/32` of its `n` columns fail the row check
(2 of 16), and a router fails on any failing column. Four positions per
layer are too few for the rule on their own: two failures of four would
pass, and a node that cheats on half of the positions would pass most
layers. Honest backends fail at most 0.34% of column checks.

## Recheck evidence

`--evidence FILE` makes `audit-receipt` write the audit's evidence: the signed
receipt, the challenge with the VRF proof that drew it, the opening's bytes
with the node's signature of them, and the VRF proof that drew the checked
rows. It cannot be combined with `--layer`, because a challenge picked by hand
is not evidence. It also holds the attention rows the node sent after the
opening, with the node's signature of them, so a recheck runs the attention
check again.

Anyone with the evidence and the claimed GGUF can check the audit again.
`--claimed-model` takes the same values as in `audit-receipt`. Without it,
the command reads the model the receipt names, if that is a Hugging Face
file. Then it needs nothing but the evidence file and access to Hugging
Face. A Mesh node names its model that way when the GGUF came from Hugging
Face.

```bash
skippy recheck-receipt \
  --evidence evidence.json \
  --claimed-model /models/Qwen3-4B-Q8_0.gguf
```

The command first checks that the file is evidence: the node's signatures of
the receipt and of the opening verify, and the client key's VRF draws exactly
the challenge from the turn root and the rows from the opening's bytes. A file
that fails this is not evidence, and the command exits with an error. So is
evidence this machine could not have audited (see above), such as a receipt of
another `skippy-abi` version. Then it runs the audit's checks again and reports the verdict, pass or fail, with a
zero exit status. The response text is not part of the evidence, because the
node does not sign it, so check 6 runs only in the original audit.

## HTTP API

Every stage serves these routes on its receipts address. Errors return HTTP 400
with `{"error": "..."}`, except for a receipt that does not exist (yet), which
returns HTTP 404. Stage 0 commits a turn before it sends the last chunk of the
response, so the receipt is there by the time the response ends.

| Route | Served by | Purpose |
|---|---|---|
| `GET /receipts/{id}` | Stage 0 | The receipt for a response. `id` is the request's `x-request-id`. Stage 0 keeps the last 256 receipts |
| `POST /receipts/open` | Any stage | Opens a challenge `{receipt_id, picks: [{layer, positions}], head, proof}` on this stage and every later one, and returns one opening per stage. `head` lists the positions whose final residual the last stage opens. For a bound receipt, stage 0 signs the response body with its node key in the `x-receipt-opening-signature` header. For a bound receipt, stage 0 requires `proof`, the hex VRF proof of the turn root under the client key, and opens only one distinct challenge |
| `POST /receipts/rows` | Any stage | After an opening, returns the K and V rows of the attention blocks the client drew, `{receipt_id, picks: [{position, layer, blocks}], proof}`, with their proofs, from this stage and every later one. Each pick must be one the stage opened partials for, with at most two distinct blocks. For a bound receipt, stage 0 requires `proof`, the hex VRF proof of the opening's bytes under the client key, and signs the response body in the `x-receipt-opening-signature` header |
| `POST /receipts/commit` | Stages after stage 0 | Commits a session's positions. The previous stage calls this at the end of a turn |

A program that embeds the runtime can serve receipts without a receipts
address, and carry two routes on its OpenAI address instead, for clients that
reach the node only there (a Mesh peer). Those routes open bound receipts only,
so nobody but the client reads a turn's activations:

| Route | Served by | Purpose |
|---|---|---|
| `POST /v1/receipts/open` | Stage 0, on the OpenAI address | `POST /receipts/open` for a bound receipt. The body may also name the `model`, which a Mesh node routes the request by. An unbound receipt returns HTTP 403 |
| `POST /v1/receipts/rows` | Stage 0, on the OpenAI address | `POST /receipts/rows` for a bound receipt, with the same `model` field |

The receipt id is the request's `x-request-id`. A client can send its own UUID
in that header, or read the `x-request-id` header of the response.

## Cost and limits

- Receipts cost some decode and prefill throughput. Each ubatch copies the
  committed tensors, including the stored KV rows, to pinned host memory. A
  worker thread hashes them while the next ubatch computes, and a commit or
  opening waits for it to finish. Two-stage Qwen3-4B Q8_0 split on one RTX
  5060 Ti, release build, median of five runs: decode 71.7 → 69.5 tok/s (3%
  slower), prefill 2292 → 1661 tok/s (1.38x slower).

- An opening replays the challenged sequence from its first position, so its
  cost grows with the length of the conversation. Each stage replays once
  per challenge, for all of its picked layers, and the stage that holds
  layer 0 always replays to open its input. Each replayed step stops after
  the stage's last picked layer (the last layer, on the final stage, when
  the challenge names head positions), so the later layers and the output
  head are skipped. The table below was measured with one layer per challenge. The same two-stage split on the RTX 5060 Ti, challenging the last
  16 positions (the worst case):

  | Conversation (prompt + generated) | Serving | Open layer 5 (stage 0) | Open layer 30 (stage 1) |
  |---|---|---|---|
  | 86 + 256 | 3.9 s | 0.67 s | 1.27 s |
  | 88 + 1024 | 14.3 s | 2.02 s | 4.30 s |
  | 2859 + 32 | 2.1 s | 0.39 s | 0.66 s |
  | 2857 + 512 | 9.3 s | 1.32 s | 2.76 s |

- The auditor's cost per layer of 16 positions, release build on the CPU
  (measured before challenges spread over four layers): loading the layer
  takes 0.5 to 1.3 s, the first glue evaluation
  76 to 160 ms (it creates the glue context), and each later evaluation of
  the same layer 0.03 to 0.27 ms per position. The row checks take 20 ms to
  1 s, depending on the page cache and, for MoE, the full router check.

- A stage keeps the openings it made, up to 128 MiB, and answers a repeated
  challenge from them in milliseconds. They are removed with their turn.
- A bound receipt is opened for one challenge, and an unbound one for at
  most 4 distinct challenges. A replay costs about as much as serving the
  turn did, so this bounds the work that one receipt can ask for.
- Replays run one at a time in a second context of the stage model, which
  needs its own KV cache. The context holds only the KV cells the replay
  uses, at least one batch and at most the stage's context, so it grows with
  the conversation and not with `--ctx-size`. It is created by the first
  replay, created again larger when a replay needs more cells, and freed
  after 60 seconds without a replay.
- An unbound receipt is fetched after the response, not attached to it.
  A receipt names the claimed model by name, not by digest. The parent turn is the one the
  client names; the node does not check it, the client does.
- The head check does not replay a sampled turn's draw: it checks only
  that the token is one the stated filters keep. Penalties, grammars and
  the other samplers are not judged.
- Attention is checked loosely, at two positions per picked layer and one
  head each, within tolerances an f64 recomputation sets; a node may change
  attention within them unseen. Other token-mixing ops (recurrent state,
  DeltaNet) are not checked inside. The check follows an output that
  passes through layout changes, the rotation of a quantized cache, and
  an output gate (Qwen3.5) before it leaves the token-mixing region; it
  refuses other ops there. With a quantized K cache the kernel rounds q
  to q8_0 blocks of 32 before its dots with K, and the check rounds q the
  same way.
- A hybrid stage builds its reserve graph with tokens of two sequences,
  which the receipt classifier cannot follow. The first graph that
  computes receipts then plans the scheduler's allocation again, once. A
  hybrid ubatch that holds more than one token of each of several
  sequences gets no receipt for its turns.
- On the last layer of the final stage, the FFN runs only for the output rows,
  so the FFN matmuls and glue checks cover fewer positions.
- The auditor's llama.cpp must build the layer as the node's does. A model
  whose graph changes between llama.cpp versions can fail glue checks until
  both sides use the same pin.
- The receipt tensors are graph outputs while receipts are on. This keeps them
  in memory and can disable some kernel fusions. An opening's replay reports
  more tensors, so it fuses fewer kernels than the turn did; when that
  changes a committed value, the replay is anchored to the committed values
  instead (`mesh/docs/design/RECEIPTS.md`, Node: commitment and replay).
