# Inference receipts

An inference receipt lets a client check that a serving node ran the model
that it claims to run. This feature is experimental and off by default.

This guide is for operators who serve with receipts and for users who audit
a node. The design is in
[RECEIPTS.md](../../mesh/docs/design/RECEIPTS.md) and
[RECEIPTS_GLUE.md](../../mesh/docs/design/RECEIPTS_GLUE.md). The status of
each milestone is in
[RECEIPTS_PLAN.md](../../mesh/docs/design/RECEIPTS_PLAN.md). The limits
that are still open are in
[Open limitations](../../mesh/docs/design/RECEIPTS.md#open-limitations).

Do not confuse an inference receipt with the generation receipt in
`skippy-serving`. A generation receipt is a digest of the generated token
ids for local generation hooks. It does not show which model computed them.

## How a receipt works

1. The client names a VRF key in its request (`x-receipt-client-key`).
2. Each stage hashes the values of every layer while it serves the turn.
   At the end of the turn, each stage commits those hashes with a BLAKE3
   Merkle root.
3. Stage 0 binds the roots, the tokens of the turn and the client key into
   a turn root. It signs the turn root with its node key. This signed
   record is the receipt, and it is in the response.
4. The VRF of the turn root draws the challenge: four layers, four
   positions in each layer, and four sampled tokens. The client and the
   node cannot choose them.
5. Each stage replays its part of the turn in a scratch context and opens
   the challenged values. The node signs this opening.
6. The client checks the opening against the receipt and against the
   claimed GGUF, on its own CPU.

A node that serves a different quantization, or a different model, fails
these checks.

## Outcomes and exit codes

`skippy audit-receipt` and `skippy recheck-receipt` end with one of four
outcomes. The last line of the text report is the outcome.

| Last line | Exit code | JSON `outcome` | Meaning |
|---|---|---|---|
| `AUDIT PASS` | 0 | `pass` | Every check ran and passed |
| `AUDIT FAIL` | 1 | `fail` | The receipt or the answer of the node is defective, or it contradicts what the client sent or received |
| `NO AUDIT` | 2 | `no_audit` | This machine could not run the audit. This says nothing about the node |
| `NO ANSWER` | 3 | `unanswered` | The node gave no usable signed answer to the challenge or to the request for rows |

Each failure is one of two types:

- A provable failure is in something that the node signed. `skippy
  recheck-receipt` fails the evidence in the same way, on all machines.
- A failure that this client alone saw has the mark `(seen by this client
  alone)` in the report. The response text and the request are of this
  type. The node did not sign them, so they convince nobody else. Logits
  that only the head key refused are also of this type, because nobody
  else has the key. `skippy recheck-receipt` checks the same logits on
  every row of the output matrix, and it fails them when they are that
  far from the claimed model.

A command error that is not an outcome, for example an evidence file that
cannot be written, also exits with 1. Read the last line of the report to
tell the two apart.

## What a pass proves

A pass is a statement about the picked layers, positions and sampled
tokens of one turn. For those, a pass shows that:

- the node used the claimed weights, with the claimed quantization, in
  every weight matmul of the four picked layers.
- the operations between the matmuls are those of the claimed model.
- the input of layer 0 is the embedding of the token that the receipt
  names.
- at four sampled positions, the logits come from the claimed output
  matrix, and the sampled token is one that the stated sampler can give.
- the response is the text of the tokens that the receipt names.
- at two positions per picked layer, the attention of one head agrees with
  the committed K and V rows, within loose tolerances.
- when the turn continued a cached prefix, the same things hold at two
  positions of the prefix, except the check of the sampled token.
- the prompt is part of the text of the tokens before the response, the
  tokens of a cached prefix included.

A pass does not show these things:

- The positions and layers that the challenge did not pick. The check is a
  sample.
- Most of a cached prefix. The challenge picks two of its positions. A
  node that changed one token of a long prefix is caught with a small
  probability.
- A sampling draw near a boundary. The audit replays the draw of a sampled
  turn within the draw band, 0.01% of the root mean square of the logits.
  A node can steer a draw only when its uniform is that close to the
  boundary between two tokens.
- The end tag of a reasoning budget. The audit replays which tokens the
  budget forced, with the tags that the node states. It does not derive
  the tags from the chat template. The report shows them.
- A seed that the node chose. When the request fixed no seed, the replay
  shows only that the node followed the seed that it states.
- Exact attention. A change within the tolerances passes.
- The file hash. The audit reads the bytes that Hugging Face serves for
  the named commit over TLS.
- Privacy. The node reads the prompt and the response.

The full list, with the threat model, is in
[What a pass proves](../../mesh/docs/design/RECEIPTS.md#what-a-pass-proves).

## Requirements

**Serving.**

- Standalone Skippy issues receipts on binary split serving
  (`skippy serve --stage-transport binary`). A Mesh node issues them for
  the models that it serves alone and for the splits that it coordinates,
  when `[receipts] enabled` is on.
- The weights are F32, F16, BF16, Q8_0, Q4_0, Q5_0, IQ4_NL, IQ4_XS or a
  K-quant from Q2_K to Q6_K. Dense and MoE models are supported.
- The backend is CPU, CUDA, or experimentally Metal on the exact device
  `Apple M5 Max`. Metal requires every matrix in the actual loaded GGUF
  artifacts (including embeddings and the output head) to be Q8_0, full
  offload to one unambiguously selected GPU, no host experts, and no
  disabled op offload. Implicit multi-shard GGUF loading is refused;
  explicitly supplied parts are all checked. This checks artifact metadata
  and placement configuration, not a resident-tensor attestation. No other
  Apple GPU or Metal matrix type is qualified.
- Unqualified stages, including Vulkan and ROCm, serve normally without
  receipts. `skippy serve` prints `Receipts stay off for this model stage`
  with the reason; Mesh derives its receipts flag from the attached recorder,
  not the requested setting. The auditor refuses missing backend/type
  profiles. Vulkan lost its profiles on 2026-11-10: an honest Vulkan node
  with Q4_K_M weights on an integrated GPU passed 2 of 16 audits.
- The model uses flash attention. A stage with a transposed V cache gets
  no receipt.
- The KV cache stores f16 rows, which is the default. A stage with a
  quantized KV cache (`cache_type_k` or `cache_type_v` of q8_0 or q4_0)
  issues no receipts, and a Mesh node does not advertise receipts for it.
  A layer package can name a quantized cache as its default. Set both
  cache types to f16 to get receipts.
- The KV cache can be unified, or it can keep one stream per sequence.
- The receipt classifier must accept the graph. See
  [Troubleshooting](#a-turn-gets-no-receipt) for the reasons that it can
  give.

**Stages that issue receipts with a limit.**

- A stage with a native MTP head issues receipts. The trunk checks every
  draft, so the receipt covers the trunk and not the head.
- A hybrid model, for example Qwen3.5, issues receipts. See
  [Hybrid memory](#hybrid-memory).
- On a hybrid stage, a turn that continues a cached prefix gets no
  receipt.
- A turn that continues a cached prefix gets no receipt when the stage
  cannot replay a position of the prefix. See
  [The cached prefix](#the-cached-prefix).
- A turn with image tokens gets no receipt.
- A model that has recurrent memory and no attention issues no receipts. A
  standalone stage with such a model stops at startup when receipts are
  on.

**Auditing.**

- The auditor needs the GGUF that the node claims: a local file, or a file
  on Hugging Face that it reads by range.
- The auditor loads layer 0 and each challenged layer on the CPU, with the
  native Skippy runtime.
- The auditor and the node must run the same Skippy ABI and the same
  llama.cpp build. See [Upgrade and compatibility](#upgrade-and-compatibility).

## Quick start

This procedure serves a two-stage split on one machine and audits one
turn.

1. Make the stage configs. The `--worker` addresses are the internal stage
   listeners. The output directory must not exist.

   ```bash
   skippy plan-split --model-path /models/Qwen3-4B-Q8_0.gguf \
     --worker 127.0.0.1:9400 --worker 127.0.0.1:9401 --output-dir plan
   ```

2. Start stage 1 in one terminal. `--receipts` turns receipts on without a
   receipts API.

   ```bash
   skippy serve --stage-transport binary --worker-only \
     --config plan/stage-1.json --receipts
   ```

3. Start stage 0 in a second terminal. `--receipts-listen` turns receipts
   on and serves the receipts API.

   ```bash
   skippy serve --stage-transport binary --config plan/stage-0.json \
     --bind-addr 127.0.0.1:9337 --receipts-listen 127.0.0.1:9501
   ```

4. Audit one turn. Against standalone `skippy serve`, `--receipts` is
   necessary, because only a Mesh node opens receipts on its OpenAI
   address.

   ```bash
   skippy audit-receipt \
     --api http://127.0.0.1:9337 \
     --receipts http://127.0.0.1:9501 \
     --claimed-model /models/Qwen3-4B-Q8_0.gguf \
     --evidence evidence.json
   ```

5. Read the last line of the report. `AUDIT PASS` and exit code 0 show
   that the audit passed.

6. Check the evidence again. This step needs only the evidence file and
   the claimed GGUF.

   ```bash
   skippy recheck-receipt \
     --evidence evidence.json \
     --claimed-model /models/Qwen3-4B-Q8_0.gguf
   ```

## Serve with receipts

Start stage 0 with `--receipts-listen`, which serves the receipts API that
clients and auditors use. Start each later stage with `--receipts`. Each
stage reaches the receipts of the next stage over the binary stage
connection. That connection is a dedicated one to the binary address of
the next stage. It carries only receipts messages and holds no lane.

A later stage can also serve its own receipts API with `--receipts-listen`.
If you start the stage before it with `--receipts-downstream URL`, that
stage uses the API and not the stage connection.

| Flag | Purpose |
|---|---|
| `--receipts` | Turn receipts on without a receipts API. Use it on a later stage |
| `--receipts-listen ADDR` | Turn receipts on and serve the receipts API on `ADDR` |
| `--receipts-allow-remote` | Permit a `--receipts-listen` address that is not loopback |
| `--receipts-downstream URL` | The receipts API of the next stage. Commits and openings for its layers go there |
| `--receipts-claimed-model NAME` | The model that each receipt names. The default is the stage model id |
| `--receipts-signing-key FILE` | A file with the Ed25519 seed that signs turn roots, as 64 hex digits. Without it, the stage makes a key at startup |

Use `--receipts` or `--receipts-listen`, not both. The other four flags
need one of them.

When receipts are on, the stage log has the line `receipts on:` with the
API address, the stage index and the layers. The stage also logs the node
key that signs its receipts.

### Bound and unbound receipts

A request gets a bound receipt when it names the VRF key of the client in
the `x-receipt-client-key` header (64 hex digits). Stage 0 binds these
things into the turn root and signs it:

- the client key.
- the parent turn root, if the request names one in `x-receipt-parent`.
- the sampler statement, with the seed. A sampled turn draws with the seed
  of the request body, else with the seed that the request names in
  `x-receipt-seed` (a decimal number above 0), else with a random seed.
- the runtime: the `skippy-abi` version, the llama.cpp commit and a hash
  of the patch queue.
- the sampler statement.
- its node key, and every field of the receipt.

The sampler statement is `greedy` when every token was the best logit.
That is true with sampling off, or with temperature 0 and no penalties,
logit biases, grammar or reasoning budget. Otherwise the statement is
`sampled` and a JSON object with the parameters and the seed of the
sampler. A thinking model gets a default reasoning budget from the server.
So a request with temperature 0 gets a `sampled` statement unless it also
sends `"reasoning_budget": -1`.

A turn with a reasoning budget states the budget, and what the runtime
built the sampler of the budget from, in the `reasoning` field: the tag
that starts the reasoning, the tags that end it, and the text that the
chat template placed after the prompt. For a model that reasons between
`<think>` and `</think>`, the two fields look like this example:

```json
"reasoning_budget": 512,
"reasoning": {
  "start": "<think>",
  "ends": ["</think>"],
  "prefill": "<|im_start|>assistant\n"
}
```

The sampler forces the end tag when the reasoning uses the budget up. An
auditor replays the sampler from these texts and the tokens of the turn,
so it knows which tokens were forced. The rule is in
[Reasoning budget](../../mesh/docs/design/RECEIPTS.md#reasoning-budget).

Today the runtime builds that sampler only for a chat that has a grammar.
A chat without a grammar enforces no budget, and its statement has
`"reasoning": null`. The audit then judges every token of the turn by the
usual rule.

A sampled turn that gets a receipt draws each token by the rule
`indexed-v1`, and the statement names the rule in `draw`. The uniform of a
token is a function of the seed and of the position of the token, and the
sampler takes the kept candidates in token id order. So a client that
knows the seed and the logits can replay the draw. A turn without a
receipt keeps the `dist` sampler of llama.cpp. The rule is in
[Sampling draw](../../mesh/docs/design/RECEIPTS.md#sampling-draw).

A bound receipt is in the response: in the `receipt` field of a chat or
text completion body, and of the last chunk of a stream.

A request without a client key gets an unbound receipt on a split. An
unbound receipt is only on the receipts API, and the auditor rejects it. A
single stage in a Mesh node commits only bound turns.

The recorder runs on every ubatch while receipts are on, for bound and
unbound requests.

No shipped client keeps the chain of parent turn roots. The node binds the
parent that the client names and does not check it.

### Keep the receipts API private

The receipts API refuses an address that is not loopback unless you also
pass `--receipts-allow-remote`.

CAUTION: Do not expose the receipts API to parties that must not read the
prompts. An opening contains activations that come from the prompt.

A bound receipt opens only for the client that holds its VRF key. An
unbound receipt opens for all callers of the receipts API.

### What a stage keeps

A stage keeps leaf hashes and a replay log. It does not keep activations.

- Per position, the stage keeps `2 · layers + 1` hashes and the Merkle
  tree above them.
- Per ubatch, the log holds the tokens, the positions, the output rows,
  the KV cells and `n_kv`. On a stage after the first, it also holds the
  stage input rows: one hidden-state row per token.
- Committed turns use at most 1 GiB per stage. The stage removes the
  oldest turns first, and a removed turn can no longer be opened.
- A bound receipt is kept for the 12-second challenge pin, and for 30
  seconds after it is opened.
- The stage keeps at most 256 receipts that no client can still challenge,
  and at most 64 MiB of receipts. It removes the oldest of them first.
- When the receipts that can still be challenged fill the 64 MiB, a new
  turn gets no receipt.
- If a new turn uses a request id again, it replaces the record of the old
  turn. It does not when the old turn belongs to another client and can
  still be challenged. Then the new turn gets no receipt.

A replay runs only the turn, on top of the KV cells that the turn read.
The receipt binds those cells by a prefix root, and an opening copies them
from the live cache. So an opening fails after another request wrote one
of those cells.

### The cached prefix

A challenge also picks two positions of the cached prefix. The stage did
not compute them in the turn. For every KV cell it keeps the leaf hashes
of the position and the recorded ubatch that wrote the cell. It opens a
picked position by a replay of that one ubatch, on top of the cells below
it. Another request, or a cache sequence, may have computed the prefix.
The stage that holds layer 0 sends the tokens of the whole prefix with the
opening, and the receipt binds their digest.

What this costs an operator:

- Memory. The leaf hashes are about 2.3 kB per position for a stage of 36
  layers, less than 2% of the K and V rows. A stage after the first also
  keeps the input rows of each ubatch, at most 1 GiB. Past that it drops
  the oldest ubatch.
- Time. An opening of a cached turn also replays up to two ubatches of the
  prefix. On a CPU stage that can take seconds for a ubatch of 512 tokens.
  The time is not measured yet.

A turn gets no receipt when the stage cannot replay its prefix:

- after a state load, a context shift or a compaction;
- when a ubatch that the recorder could not record wrote a cell of the
  prefix;
- when the prefix holds cells of two requests;
- when the ubatch of a prefix cell is past the 1 GiB of input rows;
- on a stage with recurrent state.

The log line names the cause after `the turn's cached prefix is unknown`.

Each stage pins the cells of a bound turn for 12 seconds, so a later
request writes other cells while the cache has room. After the opening,
the pin stays until the client has its rows, for at most 30 seconds. A
stage with a full cache writes pinned cells anyway, and the turn can then
no longer be opened. An unbound turn is not pinned.

With receipts on, a repeated greedy request does not reuse the tokens that
an earlier request generated. The turn computes them again.

### Hybrid memory

A recurrent layer (Mamba, the gated delta net) keeps one state per
sequence, and each ubatch writes over it. A replay cannot build that state
again from the KV cache.

- Before the first ubatch of a turn, the stage reads the state from its
  recurrent memory.
- The stage commitment names the BLAKE3 digest of the state in
  `recurrent_state`.
- The stage keeps the state while a client can challenge the turn. A
  replay starts from that state.
- Each ubatch also commits the state that it read, in the mixing leaf of
  its first position.

For the first attention pick of a stage on a recurrent layer, the auditor
runs the token-mixing ops of the layer again on the CPU. It compares the
outputs and the states at both ends of the opened region.

A hybrid stage restores a cached prefix with a KV page import and a state
read. The recorder cannot replay those, so a turn that continues a cached
prefix gets no receipt on a hybrid stage.

## Audit a response

`skippy audit-receipt` makes a new VRF key and sends one chat request that
names it. It gets the receipt, draws the challenge from the VRF of the
turn root, and checks the opening.

```bash
skippy audit-receipt \
  --api http://127.0.0.1:9337 \
  --receipts http://127.0.0.1:9501 \
  --claimed-model /models/Qwen3-4B-Q8_0.gguf
```

| Flag | Default | Purpose |
|---|---|---|
| `--api URL` | `http://127.0.0.1:9337` | The OpenAI base URL of stage 0 |
| `--receipts URL` | None | The receipts API of stage 0. Without it, the command opens the receipt on the OpenAI API, which only a Mesh node serves |
| `--claimed-model MODEL` | Required | The GGUF that the node claims: a local file, or `repo@revision/file` on Hugging Face |
| `--prompt TEXT` | `Explain in two sentences why the sky is blue.` | The prompt of the request |
| `--max-tokens N` | 32 | The response length |
| `--layer N` | None | Challenge this layer alone. The report then says that the challenge is not evidence |
| `--timing-runs N` | 1 | Compute each challenged layer N times for the cost report |
| `--evidence FILE` | None | Write the evidence of the audit to this file. You cannot use it with `--layer` |
| `--temperature T` | 0 | Sample at this temperature. At 0 the node decodes greedily. Above 0 the audit sets a seed and replays the draw of each checked token |
| `--top-p P` | The default of the node | Top-p of a sampled turn. Needs `--temperature` |
| `--top-k K` | The default of the node | Top-k of a sampled turn. Needs `--temperature` |
| `--min-p P` | The default of the node | Min-p of a sampled turn. Needs `--temperature` |
| `--seed N` | Random | The seed of a sampled turn, above 0. Needs `--temperature` |
| `--reasoning-budget N` | No limit | The reasoning tokens that the request allows a thinking model. Where the runtime enforces the budget, the node forces the end of the reasoning when they run out, and the audit replays which tokens it forced |
| `--output json` | Text | Print the report as JSON |

The command sends the request to the first model that `GET /v1/models`
lists. It asks for greedy decoding: temperature 0 and
`"reasoning_budget": -1`. With `--reasoning-budget N` it sends that
budget instead, and the statement of the node is then `sampled` also at
temperature 0.

Without `--receipts`, the command takes the receipt from the response and
opens it with `POST /v1/receipts/open` on the OpenAI address. The challenge
names the model, and it names the node key of the receipt in
`x-mesh-target`. A Mesh node thus sends the challenge to the peer that
signed the receipt.

### The claimed model

`--claimed-model` takes a local GGUF or a Hugging Face file as
`repo@revision/file`, for example
`unsloth/Qwen3-4B-GGUF@main/Qwen3-4B-Q8_0.gguf`. Name a split GGUF by its
first shard. The command reads every shard.

The command does not download a Hugging Face file. It reads only the bytes
that the checks use, with range requests, into a sparse copy. Those bytes
are the header, the small tensors of the challenged layers, the checked
rows, the embedding rows of the challenged tokens, and the output matrix.

- The sparse copy is in the platform cache directory, at
  `~/.cache/mesh-llm/receipts/claimed/` on Linux. It is keyed by the commit
  that the revision resolves to.
- Later audits fetch only the bytes that the copy does not hold.
- The output matrix is read one time, to build the head key. The copy does
  not keep it.
- `HF_ENDPOINT` and `HF_TOKEN` apply. The token goes only to the Hugging
  Face host, not to the CDN.
- The header is read before the request, so it does not delay the
  challenge.

The command reads the GGUF header strictly, for a local file too. It
refuses a file over its limits, and the result is then `NO AUDIT`. The
limits are in [Parameters](../../mesh/docs/design/RECEIPTS.md#parameters).

### The checks, in order

Before the challenge, the command decides from the receipt alone whether
this machine can audit it. See [NO AUDIT](#no-audit). Then it does these
checks:

1. **Receipt.** The receipt is bound to the key of the auditor. The turn
   root matches the fields, and the signature of the node key is correct.
   The stages cover every layer of the claimed GGUF from 0, in order. The
   layer count comes from the metadata of the GGUF, not from the receipt.
2. **Request.** The receipt has the id of the request, and the sampler
   statement does not contradict the request. Without `--temperature`
   and `--reasoning-budget`, `audit-receipt` asks for greedy decoding, so
   a receipt whose sampler is not `greedy` fails. With them, the statement
   must name the temperature, the filters, the seed and the reasoning
   budget of the request.
3. **Opening.** The signature of the node covers the bytes of the opening
   and the challenge that it answers. The opening holds exactly the stages,
   layers and positions of the challenge, with nothing twice and nothing
   extra. Every opened leaf is in the tree of its stage. Where one stage
   hands over to the next, both stages open the same residual leaf. A
   picked position of a cached prefix is in no tree of the turn. The stage
   opens every leaf hash that it committed for the position, and those
   must be in the prefix tree of the receipt. The tokens of the prefix
   must hash to the digest in the receipt.
4. **Values.** Every opened value is finite and has the length that the
   claimed GGUF gives. The checks below read only values that passed
   checks 3 and 4.
5. **Weight rows.** For each matmul of a picked layer, the command reads 16
   random rows of the weight from the claimed GGUF and computes them again.
   The rows come from the VRF of the bytes of the opening. Each matrix is
   checked with the rounding profiles of the backend that its stage names.
   A router is checked on every row.
6. **Glue.** The command builds each picked layer with its own llama.cpp
   from the claimed GGUF, and computes it on the opened values. See
   [What the glue checks bind](#what-the-glue-checks-bind).
7. **Input of layer 0.** The command computes the input of layer 0 from
   each token in the receipt, with layer 0 of the claimed GGUF. The result
   must be the opened input at every picked position.
8. **Output head.** The challenge names four positions from which the turn
   sampled a token. At each, the final stage opens the residual that leaves
   the last layer and the logits of its replay. The command checks the
   logits with its secret head key. It also checks 16 random rows of the
   output matrix and the row of the sampled token. Then it checks that the
   sampled token is the greedy choice.
9. **Attention.** At two positions per picked layer, the node opens the
   attention of one head as partial sums over blocks of 64 context
   positions. The partials must combine to the committed attention output.
   The tolerance is 2e-2 for a head that attends to at most about 420
   positions. For more positions it increases with the square root of the
   context: 6.3e-2 at 4,096 positions and 1.8e-1 at 32,768. Then the
   command draws two blocks per pick and asks
   for their K and V rows (`POST /receipts/rows`). It computes those blocks
   again from the committed q, within 1e-6.
10. **Response.** The response (reasoning and content together) is the text
    of the sampled tokens, up to whitespace and reasoning markers. An empty
    response fails. A response that the node cut at the token limit must be
    the start of that text. It can leave out only an unfinished character
    or the start of a marker. For a turn that starts at position 0, the
    prompt must be part of the text of the tokens before them.

Notes on the checks:

- A column that fails check 5 is repaired when an activation near-tie
  explains it. The report counts repaired columns (`repaired_tokens` in
  JSON). The rule is in
  [Fail rule](../../mesh/docs/design/RECEIPTS.md#fail-rule).
- In check 8, the tie band is 1% of the root mean square of the logits. A
  token whose logit is inside the tie band of the best logit is accepted
  and counted as a tie. More than one tie in four positions is flagged, not
  failed.
- In check 8, a sampled turn must take a token that its stated `top_k`,
  `top_p`, `min_p` and temperature keep. A turn with penalties, DRY, XTC,
  typical or top-n-sigma sampling, mirostat or a grammar gets no token
  judgement. Its logits are still checked.
- In check 8, the audit replays the draw of a sampled turn whose statement
  names the `indexed-v1` rule. The report shows how many draws pick the
  sampled token, and how many do so only within the draw band. The draw
  band is 0.01% of the root mean square of the logits that the node
  opened. More than one such draw in four positions is flagged, not
  failed. A draw that does not pick the sampled token fails the audit, and
  the evidence shows it. A turn with a dynamic temperature or
  `ignore_eos`, and a model with suppressed tokens, get no replay. A turn
  at temperature 0 draws nothing.
- In check 8, the audit replays the reasoning budget of a turn that states
  one. A token that the budget forced must be the forced token: the end
  tag of the reasoning, one token at a time. Every other token is judged
  as usual. The report has a line `reasoning budget of N tokens, from
  ... to ... as the node states them: forced M of the checked tokens`,
  and the JSON report has the stated tags in `head.verdict.reasoning` and
  the forced token of a position in `forced`. The tags are the statement
  of the node. Compare them with the chat template of the model if you
  know it. A statement with a tag of more than 256 bytes, more than 8 end
  tags or a prefill of more than 1,024 bytes leaves the tokens unjudged.
- When the request set no seed, the report has a note: the draws were
  replayed with the seed that the node chose.
- The head key is a file that only its owner can read. When a check with
  the key refuses the logits of a node, the command deletes the key, and
  the next audit builds a new one. If the node opens no logits, that is a
  failure of the node, in the audit and in a recheck.
- In check 9, a layer whose attention the check cannot follow gives a
  `could not check:` line. If nothing failed, the outcome is then
  `NO AUDIT`. That says nothing about the node.
- When a turn continues a cached prefix, the challenge picks two positions
  of the prefix in every picked layer, and the prompt check reads the
  tokens of the prefix and of the turn. The report line is `prompt is in
  the tokens of the cached prefix (N positions) and the receipt`.

### A pass

This report is from an audit on 2026-11-09. The node is honest and serves
the claimed Q8_0 GGUF, with both stages on the CPU. Lines with `...` are
removed lines.

```text
response: The harbor at dawn is a quiet, golden-hued sanctuary where the first light of day paints the water in soft hues

receipt 2481a665-206d-436e-8d38-9038fed18da0  model Qwen3-4B-Q8_0  positions 0..44  generation from 21
  stage 0  layers  0..18  root ddc64b2f324d2ac6…  leaves 1628  cache f16  (CPU)
  stage 1  layers 18..36  root d7cea91ab4fae165…  leaves 1628  cache f16  (CPU)
  turn root 9a641adc549e561c…  node key 4ea448670cd034ae…  client key 5177fa51677edb4d…  skippy-abi 0.1.81; llama.cpp 372dc5ca157276beb6c67b2e92477e8d662e964a patches 22c4606fa8d1400aca80e531baad6424d83af05e0daab3b033dd7b5b03615d99  sampler greedy
challenge: layer 17 (stage 0), positions [12, 20, 39, 40]
challenge: layer 22 (stage 1), positions [7, 13, 28, 35]
challenge: layer 23 (stage 1), positions [6, 20, 25, 42]
challenge: layer 28 (stage 1), positions [1, 14, 24, 40]
opening: 1.02 s
receipt binding and signature, shape, opening form, merkle paths, stage boundaries and values: ok
layer 17:
  blk.17.attn_k.weight           q8_0    4/4   pass  max err 1.4e-7  cpu_q8_0
  blk.17.attn_output.weight      q8_0    4/4   pass  max err 2.5e-7  cpu_q8_0
  ...
  blk.17.ffn_up.weight           q8_0    4/4   pass  max err 1.8e-7  cpu_q8_0
  glue matmul inputs   28/28  pass  max err 0.0e0  ok
  glue mixing inputs    4/4   pass  max err 0.0e0  ok
  glue stored rows      8/8   pass  max err 0.0e0  ok
  glue residual out     4/4   pass  max err 0.0e0  ok
  client cost: layer load 0.39 s, glue 6.55 ms per position (4 positions, first run 26 ms), row checks 4 ms
layer 22:
  ...
weight rows, each role across the opened layers:
  attn_k.weight             16/16  pass  ok
  attn_output.weight        16/16  pass  ok
  ...
  ffn_up.weight             16/16  pass  ok
input of layer 0 from the receipt's tokens: 14/14 pass  max err 0.0e0  ok
sampled tokens at [24, 27, 28, 35] are the greedy choice: 4/4 pass  ties 0  ok  (0.02 s, node's logits checked with the head key)
attention, per picked head:
  position 20 layer 17       head 2   21 positions  combine 3.5e-4  blocks [0] 2.2e-8  ok
  position 40 layer 17       head 22  41 positions  combine 6.1e-4  blocks [0] 2.9e-8  ok
  ...
  position 40 layer 28       head 13  41 positions  combine 1.6e-4  blocks [0] 2.8e-8  ok
  rows opened in 0.01 s
response is the text of the sampled tokens: ok
prompt is in the receipt's tokens: ok
AUDIT PASS
```

A MoE layer adds lines for the router, which is checked on every row, for
the expert ids, and for the selections.

### A failure

This report is from 2026-11-09. The node serves another quantization of
Qwen3-4B on the CPU, and its receipts claim Q8_0.

```text
challenge: layer 14 (stage 0), positions [7, 9, 25, 28]
challenge: layer 16 (stage 0), positions [4, 10, 32, 33]
challenge: layer 23 (stage 1), positions [11, 18, 35, 37]
challenge: layer 27 (stage 1), positions [2, 5, 28, 37]
opening: 0.76 s
receipt binding and signature, shape, opening form, merkle paths, stage boundaries and values: ok
layer 14:
  blk.14.attn_k.weight           q8_0    0/4   pass  max err 4.3e-2  cpu_q8_0
  ...
  glue matmul inputs   28/28  pass  max err 0.0e0  ok
  glue mixing inputs    4/4   pass  max err 0.0e0  ok
  glue stored rows      8/8   pass  max err 0.0e0  ok
  glue residual out     4/4   pass  max err 0.0e0  ok
  ...
weight rows, each role across the opened layers:
  attn_k.weight              0/16  pass  FAIL
  ...
  ffn_up.weight              0/16  pass  FAIL
input of layer 0 from the receipt's tokens: 0/14 pass  max err 1.9e-2  FAIL
sampled tokens at [25, 28, 34, 35] are the greedy choice: 0/4 pass  ties 0  FAIL  (0.03 s, node's logits checked with the head key)
  position 25: key error 4.228e-2, row error 1.445e-2
  ...
  position 25: sampled 264, best 264, margin 4.010e0 (band 4.134e-2)
  ...
attention, per picked head:
  position 25 layer 14       head 30  26 positions  combine 1.8e-4  blocks [0] 3.8e-8  ok
  ...
  rows opened in 0.01 s
response is the text of the sampled tokens: ok
prompt is in the receipt's tokens: ok
failed: weight rows of attn_k.weight: 16/16 fail
failed: weight rows of attn_output.weight: 16/16 fail
...
failed: weight rows of ffn_up.weight: 16/16 fail
failed: input of layer 0
failed: output head  (seen by this client alone)
AUDIT FAIL
```

The Merkle paths, the glue checks and the attention check pass. The node
committed to what it computed, and its replay gave the same values. The
weight rows fail because those values do not come from the claimed
weights. The input of layer 0 fails because the token embeddings are not
the claimed Q8_0 ones. The output head fails against the head key of this
client, which nobody else has, so it has the mark `(seen by this client
alone)`. A recheck shows the same failure without the key.

The error line on standard error repeats the failures, after the native
log of the auditor:

```text
error: the receipt audit failed: weight rows of attn_k.weight: 16/16 fail; ...
```

### NO AUDIT

`NO AUDIT` means that this machine could not run the audit. It is not a
failure, and it says nothing about the node. A node whose receipts no
machine can audit gets no passes.

This report is from a command with a `--claimed-model` that does not
exist:

```text
receipt : the claimed model /nonexistent/model.gguf is neither a file here nor repo@revision/file on Hugging Face: it is not repo@revision/file
NO AUDIT
```

The command also writes this line to standard error and exits with 2:

```text
error: the receipt was not audited: the claimed model /nonexistent/model.gguf is neither a file here nor repo@revision/file on Hugging Face: it is not repo@revision/file
```

See [Troubleshooting](#no-audit-causes) for the causes.

### NO ANSWER

`NO ANSWER` means that the node gave no usable signed answer. Nothing was
checked, and the cause is on the side of the node or of the network.

This report is from a command whose `--receipts` address refuses
connections:

```text
receipt 8e3c78ce-7f74-4792-a0e6-64a7d46122d5: the node did not answer /receipts/open: error sending request for url (http://127.0.0.1:9/receipts/open): client error (Connect): tcp connect error: Connection refused (os error 111)
NO ANSWER
```

The command also writes this line to standard error and exits with 3:

```text
error: the node did not answer the audit: error sending request for url (http://127.0.0.1:9/receipts/open): client error (Connect): tcp connect error: Connection refused (os error 111)
```

See [Troubleshooting](#no-answer-causes) for the causes.

### The JSON report

Use `--output json` for a report that a program can read. Every report has
`outcome` and `pass`. This is a part of the report of a pass on
2026-11-09, with fields removed:

```json
{
  "outcome": "pass",
  "pass": true,
  "failures": [],
  "unchecked": [],
  "notes": [],
  "binding_error": null,
  "receipt_error": null,
  "opening_error": null,
  "request_errors": [],
  "anchors": {
    "verdict": { "check": "anchors", "tokens": 13, "failing_tokens": 0, "max_error": 0.0, "pass": true }
  },
  "tokens": {
    "verdict": { "response_ok": true, "prompt_ok": true }
  },
  "hand_picked": false,
  "open_seconds": 0.958884305
}
```

| Field | Content |
|---|---|
| `outcome` | `pass`, `fail`, `no_audit` or `unanswered` |
| `pass` | `true` only for the outcome `pass` |
| `failures` | A list of `{check, provable}`. `check` is the text of a `failed:` line. `provable` is `false` for a failure that this client alone saw |
| `unchecked` | The checks that could not run on this machine. These are the `could not check:` lines |
| `notes` | What the audit does not cover for this turn. These are the `note:` lines |
| `binding_error` | Why the receipt is not bound to this client or not signed |
| `receipt_error` | Why the signed receipt does not describe a turn of the claimed model |
| `opening_error` | Why the signed opening is defective |
| `request_errors` | How the receipt contradicts the request |
| `anchors`, `head`, `tokens` | Each is one of `{"verdict": ...}`, `{"node_fault": "..."}` or `{"unavailable": "..."}` |
| `tokens.verdict.prompt_ok` | Whether the prompt is in the text of the tokens before the response: the tokens of the cached prefix, then the prompt tokens of the turn |

A `NO AUDIT` report has `outcome`, `pass`, `receipt_id` and `reason`. A
`NO ANSWER` report also has `request`, which is `open` or `rows`.

The text report prints the `note:`, `failed:` and `could not check:` lines
immediately before the outcome line. It escapes each string that the node
chose and cuts it at 512 characters.

### What the glue checks bind

A replay computes the challenged layer again. So its matmuls pass the row
checks even if different weights served the turn. The glue check ties the
replayed matmuls to the committed values. The auditor builds the
challenged layer with its own llama.cpp from the claimed GGUF. It replaces
every weight matmul output, token-mixing output and selection with the
opened value, and the residual that enters the layer with the opened one.
Then it computes the layer and compares:

- the input of each weight matmul, and the experts of each MoE matmul.
- each token-mixing input, such as q after RoPE.
- each stored KV row. The opened row is in the type of the cache. The
  auditor rounds its own f32 value into the cache type and measures that
  distance. The opened row can be no further from the f32 value than 1.5
  times that distance, plus 1e-4 of the norm of the row.
- each selection, against scores from the opened router output. A top-k
  must hold the highest scores, and an argsort must be in order.
- the residual that leaves the layer.

The node must also show that its replay gives every committed leaf of the
challenged positions. The glue check does not look inside attention or
another token-mixing op. Check 9 checks attention loosely.

The glue tolerance is 1e-4, relative to the largest value of the reference
column. The row tolerance is 1e-3. A weight role (`attn_q.weight`,
`ffn_down_exps.weight`, and the others) is judged on its columns in all
opened layers together. A role fails when more than `2 + n/32` of its `n`
columns fail, which is 2 of 16. A router fails on one failing column.

## Recheck evidence

`--evidence FILE` makes `audit-receipt` write the evidence of the audit. It
writes the file for a pass and for a failure. The file holds:

- the signed receipt.
- the challenge, with the VRF proof that drew it.
- the bytes of the opening, with the signature of the node.
- the VRF proof that drew the checked rows.
- the attention rows that the node sent after the opening, with their
  signature.

An audit that refuses the receipt sends no challenge and writes no
evidence. An audit with `--layer` writes none either.

Anyone with the evidence and the claimed GGUF can check the audit again:

```bash
skippy recheck-receipt \
  --evidence evidence.json \
  --claimed-model /models/Qwen3-4B-Q8_0.gguf
```

`--claimed-model` takes the same values as in `audit-receipt`. Without it,
the command reads the model that the receipt names. That works only when
the name is `repo@commit/file` on Hugging Face with a full commit. A Mesh
node names its model in that way when the GGUF came from Hugging Face.

The command first makes sure that the file is evidence:

- The node signed the receipt, the opening for this challenge, and the
  rows.
- The VRF of the client key draws exactly this challenge from the turn
  root, and these rows from the bytes of the opening.

A file that fails this is not evidence, and the result is `NO AUDIT`. Then
the command does the checks of the audit again and ends with the same four
outcomes and exit codes. Three things differ from the audit:

- The response is not part of the evidence, because the node does not sign
  it. Check 10 does not run.
- The recheck has no head key. It reads the full output matrix and checks
  the opened logits on every row, with the rounding of the backend of the
  node. The tolerance is the tolerance of the head key, 1e-5 of the norm
  of the logits. Then it judges the sampled token on the same logits. So
  check 8 gives the result that it gave in the audit. Logits that the head
  key refused fail again when they are more than the tolerance from the
  claimed model.
- For a model that soft-caps its logits, the audit and the recheck both
  compute the logits from the output matrix. They do not have the
  rounding of the node, so a token inside a near-tie can fail.
- Evidence without attention rows reports the rows as missing. That is
  `NO AUDIT`, not a failure.

This report is the recheck of the evidence of the failure above, on
2026-11-09:

```text
weight rows, each role across the opened layers:
  attn_k.weight              0/16  pass  FAIL
  ...
  ffn_up.weight              0/16  pass  FAIL
input of layer 0 from the receipt's tokens: 0/14 pass  max err 1.9e-2  FAIL
sampled tokens at [25, 28, 34, 35] are the greedy choice: 0/4 pass  ties 0  FAIL  (0.10 s, node's logits checked on every row of the output matrix)
  ...
attention, per picked head:
  ...
  rows opened in 0.00 s
response: not part of the evidence (the node does not sign it)
failed: weight rows of attn_k.weight: 16/16 fail
...
failed: weight rows of ffn_up.weight: 16/16 fail
failed: input of layer 0
failed: output head
AUDIT FAIL
```

## Receipts in Mesh

This section is a walkthrough. The policy, with the reason for each rule,
is in
[Audit policy in Mesh](../../mesh/docs/design/RECEIPTS.md#audit-policy-in-mesh)
and [TRUST_TIERS.md](../../mesh/docs/design/TRUST_TIERS.md#receipts).

### Configure the serving node

1. Add this table to the config of the node that serves the model:

   ```toml
   [receipts]
   enabled = true
   ```

2. Run `mesh-llm config validate`. It reports a key that the node does not
   know under `[receipts]`.
3. Start the node again. `enabled` is read when the runtime of a model
   starts.

The node then signs a receipt with its node key for each turn that names a
client key. The node key is the Ed25519 key of its mesh identity. The node
serves no receipts address. A client takes the receipt from the response
and opens it on the OpenAI address, with `POST /v1/receipts/open`.

When the GGUF is a Hugging Face file, the receipt names the model as
`repo@commit/file`. A client can then audit the turn without a copy of the
model. A split GGUF is named by its first shard.

Serve the model with a Hugging Face identity. A node that serves a file
from a local path (`--gguf <path>`) names the model
`local-gguf/sha256-...`. No client can fetch what that name stands for.
So every audit of such a model ends as `no audit`, the node does not
advertise the `receipts` flag for it, and a request with
`x-mesh-trust: receipts` does not reach it. Use one of these instead:

- Name the model by its Hugging Face reference, so that the node serves
  it from its Hugging Face cache:
  `--model unsloth/Qwen3-4B-GGUF@<commit>/Qwen3-4B-Q8_0.gguf`.
- If you have the file and it is not in a cache, put it in a Hugging Face
  cache layout, point `HF_HUB_CACHE` at it, and name the model by the
  same reference. The node reads the repository and the commit from the
  path. This layout worked in the two-node test of 2026-11-10:

  ```text
  hub/models--unsloth--Qwen3-4B-GGUF/
    blobs/<sha256 of the file>
    snapshots/<commit>/Qwen3-4B-Q8_0.gguf -> ../../blobs/<sha256>
    refs/main        (holds the commit)
  ```

  The bytes must be the bytes of that commit on Hugging Face. An auditor
  reads the weights from Hugging Face, so other bytes fail the audit.

On a split, the coordinator asks every later stage to record its part. It
commits and opens that part over the stage connection, across peers.

### Keep the defaults of a serving node neutral

An auditing peer compares the sampler statement of a receipt with the
request, and the response with the sampled tokens. Defaults on the
serving node can thus give an honest node strikes.

- Do not set defaults that add a penalty or a logit bias, or that drop a
  sampler filter. The sampler statement then contradicts a request that
  set none of them.
- Do not set default stop strings. They cut the text, and the response
  check of the peer fails.

### Make sure that the serving node advertises receipts

A node advertises a model with receipts only when all of these are true:

- the loaded runtime issues receipts, because its recorder started. The
  recorder does not start on a stage with a quantized KV cache, or on a
  backend outside the [qualified serving requirements](#requirements).
- the backend/type pair has rounding profiles: CPU, CUDA, or qualified
  Metal Q8_0.
- the model serves chat completions.
- every instance of the model issues receipts.
- the receipts name a file of a Hugging Face repository at a commit.
- no stage of the model runs on another node.

If `enabled` is on and the node advertises a model without receipts, the
log of the serving node has one warning when the model loads. The warning
starts with `[receipts] enabled is set, but` and gives the cause.

No status field shows the `receipts` flag of a peer. To make sure that a
peer advertises receipts, send it a request that requires them:

1. Get the provider id of the peer from the client node:

   ```bash
   curl -fsS http://127.0.0.1:9337/mesh/v1/models | jq '
     .data[] | {id, providers: [.providers[] | .provider_id]}'
   ```

2. Send a chat completion with the trust policy and the target:

   ```bash
   curl -sS http://127.0.0.1:9337/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -H 'x-mesh-trust: receipts' \
     -H "x-mesh-target: $PROVIDER" \
     --data "$(jq -nc --arg model "$MODEL" '{
       model: $model,
       messages: [{role: "user", content: "Reply with a short greeting."}]
     }')"
   ```

3. Read the status. HTTP 200 shows that the peer advertises receipts for
   the model. HTTP 409 with the code `trust_policy_unsatisfied` shows that
   it does not, or that this node cannot audit it. HTTP 409 without that
   code is a routing error, for example a peer that is absent or blocked.

### Configure the client node

Use one of these keys on the node that sends requests for its clients.
The node reads them from its loaded configuration for each request. A
change that you apply through owner control takes effect at the next
request. An edit of the TOML file takes effect when you start the node
again: the node does not watch the file.

```toml
[receipts]
audit = true     # audit the turns that peers serve
require = false  # true: route only to runtimes that issue receipts
```

- With `audit`, the node names a new receipt key on each chat completion
  that it forwards to a peer, unless the client named one. It reads the
  receipt from the response as it relays it. The VRF of the turn root
  draws about one turn in ten for an audit.
- With `require`, each chat completion of a local client goes only to a
  runtime that issues receipts: this node's own, or a peer that advertises
  them. The node audits those turns even when `audit` is off. One request
  can ask for the same policy with the header `x-mesh-trust: receipts`.
- With `offline = true`, the node does no audit and does not contact
  Hugging Face for receipts. No peer then satisfies a policy that requires
  receipts. Only this node's own runtime can serve such a request. The
  node watches no turn, so its log has no line for a turn.

A request under the policy that no runtime can serve fails with HTTP 409
and the code `trust_policy_unsatisfied`. So does each inference endpoint
other than chat completions, because receipts cover chat completions
only. See
[Require verifiable receipts](../../mesh/docs/PROVIDER_SELECTION.md#require-verifiable-receipts).

An audit runs on one worker thread, off the request path. It opens the
challenge on the peer over the mesh, reads the claimed Hugging Face GGUF
by range, and checks the opening on the CPU.

### Name the model so that the node can audit it

The node audits a turn only when it can tie the model name of the request
to a Hugging Face file without the word of the peer. The forms below are
the names that the auditor can tie to a file. They are not all names that
a node routes. In a two-node test on 2026-11-10, a request through a
second node reached the peer only with the id that the peer advertises,
`repo@commit:selector` (for example
`unsloth/Qwen3-4B-GGUF@<commit>:Q8_0`). `GET /v1/models` on the client
node lists that id. The other forms route when this node knows the name,
for example from its loaded catalog. A receipt names the model in one
more way, `repo@commit/file`.

The forms that the auditor can tie to a file, in the `model` field of the
request:

| Form | Example |
|---|---|
| The file | `unsloth/Qwen3-4B-GGUF@main/Qwen3-4B-Q8_0.gguf` |
| A name in the loaded catalog of this node: the name, the exact ref, the file or the file stem | `Qwen3-4B-Q4_K_M` |
| A repository with a selector | `unsloth/Qwen3-4B-GGUF:Q4_K_M` |
| A repository at a revision with a selector | `unsloth/Qwen3-4B-GGUF@main:Q4_K_M` |

These names give `no audit` for each drawn turn:

- a runtime name of the peer that the catalog of this node does not list
- an alias
- `local-gguf/sha256-...`
- `owner/repo` without a selector
- a direct URL

A request that you type as `repo:selector` can also give `no audit`. This
occurs when the node rewrites the name to the bare runtime name of a peer
before it forwards the request.

### Read the verdicts

The node writes one log line for each watched turn. `<id>` is the receipt
id, or `(none)` when the response had no receipt. `<peer>` is the short id
of the peer.

| Log line | Level | Meaning |
|---|---|---|
| `receipt <id> from <peer>: audit passed` | info | Every check ran and passed |
| `receipt <id> from <peer>: no audit: <reason>` | info | The node did not audit the turn, or was not able to. This says nothing about the peer |
| `receipt <id> from <peer>: strike N of 3 within 24 h: <reason>` | warn | One strike against the peer |
| `receipt <id> from <peer>: audit failed: <failures>` | warn | A defect in something that the peer signed |
| `blocked peer <peer> for seven days: 3 strikes within 24 h` | warn | The third strike blocked the peer |
| `blocked peer <peer> for seven days: a receipt audit failed` | warn | A failure blocked the peer |
| `receipt <id> from <peer>: evidence not kept: <cause>` | warn | The node was not able to keep the evidence. The block still occurs |

The node escapes each text from a peer and cuts it at 240 characters
before it logs the text.

**Strikes.** The node checks the receipt of every watched turn, not only
of the turns that the VRF draws. A peer that advertises receipts for the
model gets a strike for each of these:

| Reason starts with | Cause |
|---|---|
| `its response carried no receipt` | The peer sent no receipt |
| `its response held no receipt this node could read` | A line or the body was larger than 16 MiB |
| `its receipt is signed by the key` | The signer is not the identity key of the peer |
| `its receipt does not decode` | The receipt is malformed |
| `its receipt is not bound to this request's key and signed` | The receipt is unbound |
| `its receipt does not verify` | The binding or the signature is wrong |
| `its receipt is for the request` | The receipt has another request id |

A peer gets a strike for this reason also when it does not advertise
receipts:

| Reason starts with | Cause |
|---|---|
| `the receipt names the model` | The receipt names another model than the request asked for |

A turn that the VRF draws can add these strikes:

| Reason starts with | Cause |
|---|---|
| `no usable signed answer to /v1/receipts/open` | The peer did not answer the challenge in the window, or its answer was not signed for the challenge |
| `no usable signed answer to /v1/receipts/rows` | The peer did not answer the request for attention rows |
| `failed checks only this node saw` | The response text, the sampler statement, or the logits against the head key |

Three strikes in 24 hours block the peer for 7 days. The node keeps
strikes in memory, so a restart clears them. A strike ends after 24 hours.
A pass does not clear a strike.

**No audit.** These are the reasons that the log can give:

| Reason contains | Cause |
|---|---|
| `the VRF did not draw the turn` | About nine turns in ten |
| `the peer advertises no receipts for this model` | The peer owes no receipt |
| `this node cannot tie the model name` or `names a repo and no file of it` | See [Name the model so that the node can audit it](#name-the-model-so-that-the-node-can-audit-it) |
| `which is not repo@commit/file on Hugging Face` | The receipt names a bare name or a local path |
| `which is not a commit and can move to other bytes` | The receipt names a branch or a tag |
| `the receipt is version` | The peer runs another release |
| `the audit worker is busy` | Two drawn turns are in the queue |
| `this node's audit worker is off` | The worker ended too many times, or its thread did not start. Start the node again |
| `this node is shutting down` | The node stops |
| `past the 10 s challenge window` | The node was not ready to send the challenge in time. The first audit of a model can cause this |
| `the attention rows were not checked` | The node was ready to ask for the rows later than 28 s after the challenge |
| `this node could not open the claimed model` | Hugging Face is out of reach, or the file is over a limit |
| `checks could not run here` | A check was not able to run on this machine |
| `the response did not end whole`, `the peer answered with status`, `the peer's answer reported an error` | The response was not a whole turn |
| `no rounding profiles for backend`, `the node ran` | See [NO AUDIT causes](#no-audit-causes) |

**Evidence.** After a failure, the node keeps the evidence under
`receipt-evidence/` in its identity state directory.

- The file name is `<20-digit time in ms>-<32 hex digits>.json`. The log
  line and the block reason give the path.
- On Unix, the directory has mode 0700 and each file has mode 0600.
- The node keeps at most 64 files and 1 GiB. It removes the oldest files
  first.
- A node without an absolute state directory keeps no evidence.
- Check a file again with `skippy recheck-receipt --evidence FILE`.
- A file with `{"refused_receipt": ...}` holds a signed receipt that the
  audit refused before a challenge. `skippy recheck-receipt` does not read
  that form.

**The worker.** If the audit worker ends unexpectedly, the node logs an
error and starts it again, at most three times. Then the log has `receipt
audits are off until this node restarts`.

### Read and lift a block

A block from an audit has `"requested_by": "receipts"`. Its reason has one
of two shapes:

```text
{"receipt_audit": {"receipt_id": "...", "model": "...", "failures": [...], "evidence": "<path>"}}
{"receipt_audit": {"strikes": [{"at_ms": N, "receipt_id": "...", "reason": "..."}], "strike_limit": 3, "strike_window_hours": 24}}
```

The management API is on the console port (3131 by default) and accepts
only loopback connections.

1. List the active blocks:

   ```bash
   curl -s http://127.0.0.1:3131/api/peer-blocks | jq
   ```

2. To lift a block, post the full 64-hex endpoint id of the peer:

   ```bash
   curl -s -X POST http://127.0.0.1:3131/api/peer-blocks/unblock \
     -H 'Content-Type: application/json' \
     --data '{"peer": "<64-hex endpoint id>"}'
   ```

A block is local. The node does not gossip it. A failed audit does not
withhold a payment: that part of the design is not built.

## HTTP API

A stage that you start with `--receipts-listen` serves these routes on its
receipts address. A stage that you start with `--receipts` alone serves no
HTTP API.

| Route | Served by | Purpose |
|---|---|---|
| `GET /receipts/{id}` | Stage 0 | The receipt for a response. `id` is the `x-request-id` of the request |
| `POST /receipts/open` | Each stage | Opens a challenge on this stage and on every later stage. Returns one opening per stage |
| `POST /receipts/rows` | Each stage | After an opening, returns the K and V rows of the attention blocks that the client drew, from this stage and every later stage |
| `POST /receipts/commit` | Stages after stage 0 | Commits the positions of a session. The stage before calls it at the end of a turn |

The receipt id is the `x-request-id` of the request. A client can send its
own UUID in that header, or read the `x-request-id` header of the
response. Stage 0 commits a turn when its response ends. If
`GET /receipts/{id}` returns 404 immediately after the response, send the
request again.

**Challenge body** (`POST /receipts/open`):

```text
{
  "receipt_id": "...",
  "picks": [{"layer": N, "positions": [...]}],
  "head": [...],
  "attention": [{"position": N, "layer": N, "head_draw": N}],
  "proof": "<hex VRF proof of the turn root>"
}
```

A stage admits a challenge before it replays anything:

- at most 4 layers, in increasing order.
- 1 to 4 increasing positions per layer, inside the turn.
- before them, at most 2 positions of the cached prefix, the same in every
  layer.
- at most 4 head positions.
- at most 2 attention picks per layer, each at a picked position.

A challenge outside these bounds gets HTTP 400. For example, a challenge
of 36 layers gets `the challenge picks 36 layers; a draw picks at most 4`.

For a bound receipt, `proof` is necessary, and each stage opens one
challenge. An unbound receipt opens for at most 4 different challenges.

**Rows body** (`POST /receipts/rows`):

```text
{
  "receipt_id": "...",
  "picks": [{"position": N, "layer": N, "blocks": [...]}],
  "proof": "<hex VRF proof of the opening's bytes>"
}
```

Each pick must be an attention pick of the admitted challenge, with at
most 2 blocks in increasing order. For a bound receipt, `proof` is
necessary.

For a bound receipt, stage 0 signs the response body of both routes with
its node key, in the `x-receipt-opening-signature` header.

**Commit body** (`POST /receipts/commit`): `receipt_id`, `session`,
`position_start`, `position_end`, and `client_key` for a bound turn.

Between stages over HTTP, a forwarded open of a bound receipt has the
header `x-receipt-turn-root`. A forwarded rows request has the header
`x-receipt-opening-digest`.

**Status codes** on `/receipts/open`, `/receipts/rows` and
`/v1/receipts/*`:

| Status | Meaning |
|---|---|
| 200 | The answer |
| 400 | A refused or malformed request, and every case that is not below |
| 403 | The route opens only bound receipts, and this receipt is unbound |
| 404 | The stage keeps no record of the receipt id. The id is unknown, or the record was removed |
| 409 | The stage can no longer open the turn: `cells_overwritten`, `cells_unknown` or `state_dropped` |
| 410 | The turn left the opening window: `expired` |
| 503 | The stage was unloaded |

The body of an error is `{"error": "..."}`. The body of a 409 or a 410 also
has `cannot_open` with `receipt_id`, `stage_index`, `reason` and `detail`.
For a bound receipt, the node signs that body in the
`x-receipt-opening-signature` header.

`GET /receipts/{id}` returns 404 when no receipt has that id, and 400 with
the reason when the turn got no receipt.

A program that embeds the runtime can serve receipts without a receipts
address. It then mounts two routes on its OpenAI address, for clients
that reach the node only there. A Mesh node does this. Standalone
`skippy serve` does not.

| Route | Purpose |
|---|---|
| `POST /v1/receipts/open` | `POST /receipts/open` for a bound receipt. The body can also name the `model`, by which a Mesh node routes the request. An unbound receipt returns 403 |
| `POST /v1/receipts/rows` | `POST /receipts/rows` for a bound receipt, with the same `model` field |

These routes open bound receipts only. So only the client reads the
activations of a turn.

## Troubleshooting

### A turn gets no receipt

A stage that cannot issue a receipt for a turn serves the turn normally.
The response then has no `receipt` field. Find the reason in one of these
places:

- the log of stage 0, in a warning that starts with `receipt <id> not
  committed:`.
- `GET /receipts/{id}`, which returns HTTP 400 with `turn <id> got no
  receipt:` and the reason.
- the report of `skippy audit-receipt --receipts URL`, which ends with
  `NO AUDIT` and quotes that answer.

| Reason contains | Cause | What to do |
|---|---|---|
| `the response carried no receipt` | Receipts are off on the node, or the turn got none, and the command has no `--receipts` | Start the stages with the receipts flags. Pass `--receipts` to the audit |
| `op ... is not in the receipt op table` | The graph uses a ggml op that the classifier does not know | Serve this model without receipts. Report the op |
| `the V cache is stored transposed; receipts need flash attention` | Flash attention is off | Turn flash attention on |
| `no audit covers a quantized KV cache` | A KV cache type of the stage is q8_0 or q4_0. Such a stage also logs `receipts stay off for this stage` when it starts, and then no response carries a receipt | Set both KV cache types of the stage to f16 |
| `an audit covers CPU, CUDA and Metal alone` | The stage runs on Vulkan, ROCm or another backend without rounding profiles | Serve on a qualified backend |
| `Metal GPUs measured` | The selected Metal device is not exactly `Apple M5 Max`, or selection is ambiguous | Serve without receipts or use a qualified device |
| `Metal placement is unqualified` | Partial offload, host experts, split GPU placement, or disabled op offload | Use full offload to one GPU with op offload enabled |
| `Metal tensor coverage is unqualified` | A loaded matrix is not Q8_0, metadata cannot be read, or GGUF shards are implicit | Use qualified Q8_0 artifacts and explicitly enumerate loaded parts, or serve without receipts |
| `which nothing commits` | A layer reads a value from another layer that is not the residual stream | Serve this model without receipts |
| `more than receipts leave unchecked` | Token-mixing ops read more than 1% of the weight bytes of the layer directly | Serve this model without receipts |
| `token mixing reaches the layer output without a weight matmul` | The classifier cannot bound a token-mixing region | Serve this model without receipts |
| `a token whose positions are not a text token's cannot be receipted` | The turn has image tokens | None. Such a turn gets no receipt |
| `a ubatch of embeddings cannot be receipted` | Stage 0 got embeddings and not tokens as its input | None |
| `a token shared by several sequences cannot be receipted` | One ubatch token belongs to more than one sequence | None |
| `the sequence's KV cells were loaded from saved state` | The stage restored a state. A hybrid stage does this for a cached prefix | None. The next turn that starts at position 0 gets a receipt |
| `the sequence's positions were shifted` | A context shift moved the positions of the sequence | None |
| `the KV cache was compacted` | A compaction moved the cells | None |
| `the sequence took KV cells from another sequence mid-turn` | A copy between sequences occurred after the turn started | None |
| `the turn's cached prefix is unknown` | The stage does not know what a cell of the prefix holds, or cannot replay the ubatch that wrote it. The text after it gives the cause | None. The next turn that starts at position 0 gets a receipt. See [The cached prefix](#the-cached-prefix) |
| `the ubatch that wrote position N of the cached prefix is no longer held` | A stage after the first dropped the input rows of that ubatch, past 1 GiB | None |
| `a stage with recurrent state cannot open the cached prefix of a turn` | A hybrid stage continued a sequence that it kept | None |
| `the recurrent state the turn started from is unknown` | A hybrid stage could not read the state | Read the stage log for the native error |
| `the turn recorded nothing` | No ubatch of the turn was recorded. The text after it gives the cause | Use the cause in the text |
| `the stage keeps ... bytes of receipts that can still be challenged` | The receipts that clients can still challenge fill the 64 MiB | Send fewer bound requests at one time |
| `names a turn of another client that can still be challenged` | The request used the request id of another client's open turn | Use a new `x-request-id` for each request |
| `the next stage did not answer`, `the next stage is not asked` | A later stage did not commit in 10 seconds, or it failed before and is not asked for a time | Make sure that the later stage runs. Wait 30 to 60 seconds |
| `this stage issues no receipts` | A later stage runs without `--receipts`. It is not asked again for 10 minutes | Start every stage with a receipts flag |

Two errors stop a stage at startup:

- `receipts need a plain KV cache`: the model has no attention KV cache.
- `receipts API on ... would expose opened activations to the network`:
  the `--receipts-listen` address is not loopback. Use a loopback address,
  or pass `--receipts-allow-remote`.

### NO AUDIT causes

| Report contains | Cause | What to do |
|---|---|---|
| `is neither a file here nor repo@revision/file on Hugging Face` | `--claimed-model` names no file and no Hugging Face file | Correct the path or the name |
| A connection error, or an HTTP status, before a receipt line | The command could not send the chat request, or Hugging Face is out of reach | Make sure that `--api` is correct and that the network works |
| `the node lists no model` | `GET /v1/models` on `--api` is empty | Wait until the model is loaded |
| `got no receipt:` | The turn got no receipt | See [A turn gets no receipt](#a-turn-gets-no-receipt) |
| `the receipt is version N, and this verifier checks version 7` | The node and the auditor are different releases | Use the same release on both sides |
| `the node ran ..., and this verifier runs ...` | The Skippy ABI or the llama.cpp build differs | Use the same release on both sides |
| `ran on backend ..., whose rounding no profile models` | A stage ran on Vulkan, ROCm or another backend without profiles | None. Such a stage cannot be audited. A current node issues no receipts on such a backend |
| `no rounding profile for ... weights` | A stage multiplies a weight type without a profile, for example q4_1 or q5_1 | None |
| `stored KV rows of type ...: an audit covers a cache of f16 or f32 rows` | A stage has a quantized KV cache | None. A current node issues no receipts with such a cache |
| `no reference decoding of ... output weights` | The auditor cannot decode the type of the output matrix | None |
| `this client cannot audit this model: ...; that is not a verdict on the node` | The claimed GGUF is over a limit of the auditor, or it is not a GGUF that the runtime loads | None |
| `names ..., not repo@commit/file on Hugging Face` | `recheck-receipt` has no `--claimed-model`, and the receipt does not name a pinned Hugging Face file | Pass `--claimed-model` |
| `not evidence of an audit:` | The file is not evidence. A signature or a VRF proof is wrong | Get the evidence file again |
| A `could not check:` line before `NO AUDIT` | One check could not run on this machine. The line gives the check and the cause | Use the cause in the line |

`recheck-receipt` without `--claimed-model` and the Mesh auditor take the
model from the receipt. A receipt that names its model with a branch, a
tag or a short hash then gives no audit.

### NO ANSWER causes

The report names the request without an answer (`/receipts/open` or
`/receipts/rows`) and the cause.

| Cause | What to do |
|---|---|
| A transport error, for example `Connection refused` | Make sure that `--receipts` names the receipts API of stage 0 |
| An error status against standalone `skippy serve`, from a command without `--receipts` | Pass `--receipts`. Only a Mesh node serves `/v1/receipts/open` |
| `404` with `no receipt` | The stage removed the record, or the id is unknown. Audit a new turn |
| `403` | The receipt is unbound. Send the request with `x-receipt-client-key` |
| `409` or `410` with `cannot_open` | The stage can no longer open the turn. The cache was full, a state load changed the cells, or the window ended. Audit a new turn |
| `503` | The stage was unloaded |
| `400` with `was already challenged` | A bound receipt opens for one challenge |
| `the node did not sign its opening`, or a signature that is not correct | The answer cannot be held against the node. Read the stage log |
| A body that does not decode | The node sent no opening |

## Files and their sensitivity

| File | Location | Content | Sensitivity |
|---|---|---|---|
| Evidence file | The path that you give to `--evidence`. On a Mesh node, `receipt-evidence/<time>-<digest>.json` in the identity state directory, with mode 0600 in a directory of mode 0700 on Unix. The node keeps at most 64 files and 1 GiB | The receipt with the token ids of the full turn, the opened activations, and the K and V rows | High. The token ids give the prompt and the response. Share the file only with parties that can read the conversation. `skippy audit-receipt --evidence` writes its file with the default mode of your shell |
| Head key | `receipts/head-keys/` in the cache directory of `mesh-llm`. Only the owner can read it | The secret vectors that check the logits of one model | Secret. A node that knows the key can change logits without detection. You can delete the file. The next audit builds a new key |
| Sparse GGUF copy | `receipts/claimed/<owner>--<name>/<commit>/` in the cache directory of `mesh-llm`, with a `<file>.lock` | Public weight bytes that the audits read | Low. The directory shows which models you audit. You can delete it. Later audits fetch the bytes again |
| Signing key | The file that you give to `--receipts-signing-key` | The Ed25519 seed of the node key, as 64 hex digits | Secret. A party that has it can sign receipts as this node |
| Receipt blocks | `peer_blocks_receipts.json` in the state directory of a Mesh node | The blocked peers and the reasons | Low |
| Replay logs | Memory of each stage, at most 1 GiB | The tokens and the stage input rows of recent turns | They are not written to disk |

The cache directory is the platform cache directory. On Linux it is
`~/.cache/mesh-llm/`. The state directory of a Mesh node is `~/.mesh-llm/`
for the default node key.

An evidence file for a Qwen3-4B turn was about 14 MB on 2026-11-09.

Without `--receipts-signing-key`, a standalone stage makes a new node key
at each start. A receipt then shows which key signed it, but not who holds
that key. A Mesh node signs with the key of its mesh identity.

## Upgrade and compatibility

- **Receipt version.** This build writes and checks receipt version 7. An
  auditor gives `NO AUDIT` for another version.
- **Runtime.** A bound receipt names the runtime: the `skippy-abi` version
  (0.1.81 in this build), the llama.cpp commit and a hash of the patch
  queue. The auditor must run the same runtime, because its glue check
  must build the layer as the node does. If the runtimes differ, the
  result is `NO AUDIT`. Upgrade the node and the auditor to the same
  release.
- **Rounding profiles.** The weight row check uses rounding profiles, and
  each profile models the kernels of one llama.cpp pin. A new pin can
  change a kernel, and the old profile then fails honest nodes. After each
  change of the llama.cpp pin, run honest audits and cheating audits again
  for each backend and weight type that you serve, on real hardware. CI
  has no GPU and cannot find a changed kernel. The CUDA K-quant profile
  changed in this way on the current pin. See
  [CUDA K-quant profile](../../mesh/docs/design/RECEIPTS_MEASUREMENTS.md#cuda-k-quant-profile).
- **Old evidence.** A newer build gives `NO AUDIT` for evidence with
  another receipt version or another runtime. Keep the release that made
  the evidence if you must check it again.
- **Stages of different versions.** A stage that does not issue receipts
  refuses receipts messages. The turns of that split then get no receipt.
  The stage before does not ask it again for 10 minutes.
- **Mesh peers.** A peer from a release without receipts does not advertise
  the `receipts` flag and does not satisfy the tier. A peer from a release
  without split receipts records nothing, and the turns of that split get
  no receipt.
- **Blocks.** A Mesh node saves the blocks from its audits in
  `peer_blocks_receipts.json`. An older release thus reads the operator's
  `peer_blocks.json` as before.
- **Caches.** You can delete the head keys and the sparse copies at any
  time.

## Cost and limits

All numbers in this section are from 2026-11-09, on one RTX 5090 with both
stages of a two-stage Qwen3-4B Q8_0 split, release build. The complete
tables, and the numbers of earlier builds, are in
[RECEIPTS_MEASUREMENTS.md](../../mesh/docs/design/RECEIPTS_MEASUREMENTS.md).

**Serving.**

- The serving budget is decode at most 3% slower and prefill at most 10%
  slower than with receipts off.
- On 2026-11-10, on one RTX 5090, decode was about 1.5% slower with
  receipts on: 200.3 tok/s on, against 203.4 tok/s off, over 14 rounds.
  That is inside the run-to-run noise of the arm without receipts. The
  decode rate depends on the idle states of the processor, so measure it
  on your hardware.
- Prefill was about 3% slower in the same rounds: 14,880 tok/s with
  receipts on, against 15,308 tok/s with receipts off.
- On CUDA, the recorder hashes the committed values on the GPU and copies
  digests. Other backends copy the values to the host.
- A stage with a recorder uses the dense-mask attention and not the
  per-sequence row lists. A stage that serves more than one sequence from
  a unified cache pays for this in decode.
- A Vulkan stage does not record. On 2026-11-10, prefill on Vulkan was 2.4
  to 2.5 times slower with a recorder, before Vulkan lost its receipts.
- Metal and more than one lane were not measured on this build.

**Openings.**

- An opening replays only the ubatches that hold a challenged position.
  It does not replay from the first position of the conversation.
- On an RTX 5090, an opening takes 0.2 to 0.3 s for a short turn, and 1.6
  to 2.3 s for a turn of 30,000 fresh tokens. A turn that continues a
  cached prefix also replays up to two ubatches of the prefix. It opens
  in 0.2 to 0.3 s at a prefix of 1,900 positions, and in 1.2 to 1.5 s at
  28,700.
- On a CPU stage, a short turn opens in about 1 s, and a turn that
  continues a prefix of 1,900 positions in 5 to 7.5 s. That is inside the
  10-second window, with less room than a fresh short turn has.
- Replays run one at a time in a second context of the stage model. The
  context holds only the KV cells that the replay uses. The stage frees it
  after 60 seconds without a replay.
- A stage keeps the openings that it made, up to 128 MiB, and answers a
  repeated challenge from them.
- A bound receipt opens for one challenge, and an unbound one for at most
  4. This bounds the work that one receipt can ask for.
- A stage serves at most 8 receipts connections from the stage before it,
  and closes one after 120 seconds without a request.
- A commit to the next stage has 10 seconds. A commit that is late gives
  no receipt for that turn, and it does not delay the response. An open
  has 120 seconds.

**Audits.**

- With a local GGUF, `skippy audit-receipt` takes about 4.5 s for a short
  turn, 8.4 s at 15,000 tokens and 12 s at 31,000 tokens. These times
  include the time to generate the turn.
- Most of the time for a short turn is the load of the challenged layers.

**Limits.**

- The limits that are still open are in
  [Open limitations](../../mesh/docs/design/RECEIPTS.md#open-limitations).
  The most important ones for an operator: CPU, CUDA and narrowly qualified
  experimental Metal Q8_0 stages are audited, attention is a loose check,
  an audit checks two positions of a
  cached prefix, and the tags of a reasoning budget are the statement of
  the node.
- `GET /v1/models` on a Mesh client node still lists the model of a peer
  that the node blocked. A request for it then gets 404.
- The attention combination check is looser at long contexts, because the
  CUDA prefill kernel accumulates in f16. The check of the two drawn
  blocks stays exact.
- On the last layer of the final stage, the FFN runs only for the output
  rows. So the FFN matmuls and the glue checks cover fewer positions
  there.
- The receipt tensors are graph outputs while receipts are on. This keeps
  them in memory and can turn off some kernel fusions. When that changes a
  committed value in a replay, the node anchors the replay to the
  committed values. See
  [Node: commitment and replay](../../mesh/docs/design/RECEIPTS.md#node-commitment-and-replay).
