# Inference Receipts

Status: built and experimental. Receipts are off by default. Skippy serves
and audits receipts of wire version 7, and a Mesh node issues them, audits
its peers and routes by the `receipts` trust tier.

These parts of the design are not built:

- a consequence for payments
- key delegation
- a client that keeps the chain of turns
- the phone verifier

[Open limitations](#open-limitations) is the complete list.

Related:

- [skippy/RECEIPTS.md](../../../skippy/docs/RECEIPTS.md): the operator and
  user guide
- [RECEIPTS_GLUE.md](RECEIPTS_GLUE.md): how a stage finds the receipt
  tensors, and how the client checks the glue
- [RECEIPTS_PLAN.md](RECEIPTS_PLAN.md): the status of each milestone
- [RECEIPTS_MEASUREMENTS.md](RECEIPTS_MEASUREMENTS.md): every measurement,
  with its date and build
- [TRUST_TIERS.md](TRUST_TIERS.md): the `receipts` tier
- [`Mesh-LLM/ln-payments-plugin`](https://github.com/Mesh-LLM/ln-payments-plugin):
  the `pay-lightning` payment plugin
- [../NODE_REP.md](../NODE_REP.md)

## Summary

A client can check on its own CPU that the serving node ran the model that
it claims. "The model" means the bytes of a GGUF file on Hugging Face at
one commit. The client does not download that file.

The node commits to its computation while it serves each turn. It signs
the commitment and sends it with the response as a receipt. After the
turn, a verifiable random function draws four layers, four positions in
each layer, and four sampled tokens. The client asks the node to open
them. Then the client checks the opened values against random weight rows,
which it reads from Hugging Face with range requests. The node is bound by
its commitment before it can know the layers, the positions or the rows.

If the audit fails in something that the node signed, a Mesh node blocks
the peer locally for 7 days and keeps the evidence. Everything in the
evidence is signed by the node or is public, so anyone can check it again.

## Goals

- The client that receives the inference does the check. No trusted third
  party, stake, or mesh-wide reputation is necessary.
- The design works for every model that llama.cpp can serve, without
  parameters that someone writes for each model.
- The design works for agentic traffic: long contexts, many short turns,
  and a small increase in latency.
- The download and the storage of the client stay small.
- A failure is provable.

## Non-goals

- Privacy. Receipts say nothing about who can read a prompt. See
  [TRUST_TIERS.md](TRUST_TIERS.md).
- Exact attention. A bit-for-bit check of attention needs an emulation for
  each GPU, kernel and llama.cpp pin. The lab showed that this is possible,
  but the maintenance cost is too high. Attention gets a loose check.
- Detection of a node that cheats on few turns. The chance to catch it per
  turn is the audit rate times the share of turns on which it cheats.
- Mesh-wide punishment. Consequences stay local to the client.

## Decisions

| Topic | Decision | Status |
|---|---|---|
| Who checks | The client, on its own CPU | Built |
| Model sizes | All sizes | Built for the weight types in [Parameters](#parameters). End-to-end audits ran on models up to Qwen3-30B-A3B. Larger models were measured for row fetch only |
| Devices | Desktop first. The parameters must fit a phone | Desktop built. Phone not built |
| Attention | Loose check only | Built |
| Consequences | Block the peer locally. No gossip, no stake | Built |
| Payments | Do not pay the final invoice of a failed turn | Not built |
| Node identities | Free. A blocked node can join again with a new key | Accepted risk |
| Failures | Provable from signed and public data | Built for the checks that the node signs |
| Audit rate | 1 in 10 turns by default; `[receipts] audit_one_in` sets 1 in N | Built in Mesh |
| Audit shape | Four layers and four sampled tokens, drawn after the node commits | Built |
| Agentic | Commit the positions that the turn computed, and the cached prefix by the leaves of the turns that computed it. Challenge immediately, in a short window | Built. A challenge picks two positions of the cached prefix |
| Parameters | From the GGUF header and tables per backend, not per model | Built |

## Protocol

This section describes receipt wire version 7 as the code implements it.

### Model identity

A receipt names its model as a string, and the turn root binds that string.

- A Mesh node names a Hugging Face GGUF as
  `owner/name@<commit>/path.gguf`. The commit is 40 lowercase hex digits.
  A split GGUF is named by its first shard.
- A standalone Skippy stage names the model that the operator gives, or the
  stage model id.

An auditor reads a model that a receipt names only when the name has this
form with a full commit. A name with a branch, a tag or a short hash gives
no audit. A user of `skippy audit-receipt` can name a local file or a
branch in `--claimed-model`, because the user and not the node chose it.

A Mesh node goes further. It does not open a repository because a receipt
names it. It derives the file from the model name of its own request, and
the receipt must name that file. See
[What the node holds a peer to](#what-the-node-holds-a-peer-to).

A commit alone does not show that the owner of the repository published
the bytes. Hugging Face serves a file at any commit of a repository, and
the commits of an open pull request are included. Anyone can open a pull
request on a public repository. So when the request names a branch, or no
revision, the auditor checks that the commit of the receipt is in the
history of that branch (`main` when the request names none). It reads the
commit list of the branch, `GET /api/models/{repo}/commits/{branch}`, in
pages of 1,000 commits, for at most 4 pages, before it reads a byte of the
file. A commit that it finds stays accepted for the life of the process. A
commit that is not in a complete list gets the same result as a receipt
that names another file. A list that does not end within 4 pages, or that
Hugging Face does not give, gives no audit. A request that names a commit
needs no check. A user of `skippy audit-receipt --claimed-model` names the
revision, and the auditor trusts it. `skippy recheck-receipt` without
`--claimed-model` takes the commit from the receipt, and its report says
that the commit was not checked against a branch.

The commit pins the bytes, so later range requests read the same file. The
receipt holds no LFS SHA-256, and the client does not check a hash on range
reads. The commit and TLS to Hugging Face are the only bond to the bytes.
The client checks that each range answer is for the bytes it asked for
(`Content-Range`), and it reads no more than it asked for.

The client reads the GGUF header one time per model with range requests
and keeps it in a sparse copy. The header gives everything else that the
client needs: architecture, layer count, tensor shapes, quantization types,
byte offsets and expert counts. The client parses the header strictly and
applies the limits in [Parameters](#parameters).

The sparse copies are under the user's cache directory, never under the
shared temporary directory. Without a cache directory, the audit does not
run. When the copies hold more than 8 GiB, opening a copy removes the
copies that were used longest ago until they fit. It never removes a copy
that another audit has open.

The native loader of llama.cpp stops the process on some metadata values
(`GGML_ASSERT`). So before the loader runs, the auditor checks those values
in Rust, and a model that fails gives no audit. The checks cover the layer
count, the expert counts and gating, the head counts per layer, RoPE
(scaling type, dimension counts and M-RoPE sections), the pooling and
classifier options, the steps that the graph divides by, the context
length, the epsilon values, and the vocabulary (tokenizer model, duplicate
tokens, the byte tokens of an SPM vocabulary, and the rows of
`token_embd`). The token ids that the auditor gives the tokenizer must be
in the vocabulary. A layer that reads the KV cache of another layer
(Gemma 3n and Gemma 4 shared KV layers) gives no audit, because a stage of
one layer does not hold that cache. See
[Open limitations](#open-limitations), item 19, for what is not covered.

### Commitment

For every position `t` that a turn computes, a stage commits
`2 · layers + 1` leaves:

```text
residual leaf(t, l) = H( x_l(t) )            for each layer l of the stage
mixing leaf(t, l)   = H( mix_l(t) )          for each layer l of the stage
output leaf(t)      = H( x_out(t) )          the residual that leaves the stage
```

- `x_l(t)` is the residual stream that enters layer `l`, as the backend
  holds it. On the stage that holds layer 0, the residual leaf of layer 0
  also hashes the token id at `t`. Thus the receipt binds the tokens of the
  turn.
- `mix_l(t)` holds the values of the token-mixing operations of the layer:
  their inputs (for attention, q after RoPE), the K and V rows as the cache
  stores them, and their outputs. The stored rows are hashed apart from the
  other values, so an attention check can open rows alone.
- At the first position of each ubatch, the mixing leaf of a recurrent
  layer also holds the state that the ubatch read.

A leaf binds each value by its ggml type, its length and `H` of its bytes.
A node can thus hash the values on the GPU that computed them and copy 32
bytes per value to the host. An auditor that opens a value hashes it again
before it checks the path.

`H` is BLAKE3. The leaves of a stage form one Merkle tree per turn, ordered
by position and then by slot. Stage 0 puts every stage root into the turn
root:

```text
turn root = H( receipt version || receipt id || model string
               || first and last position || first generated position
               || final token || every token of the turn
               || per stage: index, layers, tree root, leaf count,
                  backend name, cache types, first position of each ubatch,
                  prefix root, length and token digest,
                  recurrent state digest
               || client VRF key || parent turn root || runtime
               || sampler statement || node key )
```

The node signs the turn root with its Ed25519 node key. A Mesh node signs
with the key of its mesh identity. The receipt is in the `receipt` field of
the response body, or of the last chunk of a stream.

The root names these things in this way:

- **Client key.** The request names the VRF key of the client in
  `x-receipt-client-key`. The client is thus bound to its key before it
  learns the root. A receipt without a client key is unbound, and an
  auditor rejects it.
- **Runtime.** The Skippy ABI version, the llama.cpp commit, and a hash of
  the patch queue that built the native library.
- **Backend.** The name of the backend device of each stage, for example
  `CPU` or `CUDA0`. The root does not name a rounding profile. The auditor
  tries each profile that the backend family has for the weight type and
  takes the best one.
- **Sampler statement.** `greedy`, or `sampled` with a JSON object of the
  sampler parameters and the seed. A grammar is named by its BLAKE3 hash.
  A sampled turn that gets a receipt draws by the rule `indexed-v1`, and
  the statement names it in `draw` (see [Sampling draw](#sampling-draw)).
  A turn with a reasoning budget states the budget and, in `reasoning`,
  what the runtime built the sampler of the budget from (see
  [Reasoning budget](#reasoning-budget)).
- **Parent.** The turn root that the client names in `x-receipt-parent`.
  The node binds the value and does not check it.
- **Prefix.** A stage commits a prefix root exactly when the turn starts
  after position 0. The root is over one leaf for each position below the
  turn:

  ```text
  prefix leaf = H( position || KV cell index || cell digest
                   || position digest )
  cell digest     = Merkle root of the stored-row leaf of each layer
  position digest = H( every leaf of the position, in slot order )
  ```

  The leaves of a position are the ones that the stage committed when it
  computed the position, in an earlier turn or for another request. The
  cell digest binds the K and V rows that the turn read. The position
  digest binds the residuals and the mixing values behind those rows, so a
  client can challenge a position of the prefix as it challenges a
  position of the turn.
- **Prefix tokens.** The stage that holds layer 0 commits the BLAKE3 digest
  of the tokens of the prefix. The node sends the tokens with the opening.

One thing that the first design named is not built. No shipped client
keeps the chain of parents: `turn::verify_chain` exists, and nothing calls
it.

### Challenge

The challenge comes from a verifiable random function (VRF), not from the
free choice of the client. The suite is ECVRF-EDWARDS25519-SHA512-TAI
(RFC 9381). The VRF gives one output per key and input, and anyone with the
public key can check it:

```text
(beta_c, pi_c) = VRF( client key, "mesh-llm/receipts/challenge" || turn root )
(beta_r, pi_r) = VRF( client key, "mesh-llm/receipts/rows" || opening bytes )
```

`skippy audit-receipt` and a Mesh node make a new client key for each
request. A delegation from the mesh node key to a receipts key is not
built.

`beta_c` decides whether a Mesh node audits the turn, at a rate of 10%. It
also picks:

- **Layers.** Four layers without replacement. The chance of each layer is
  proportional to its weight bytes, from the header.
- **Positions.** Four positions of the turn per layer. Half of them are
  generated positions when the turn generated tokens.
- **Prefix positions.** Two positions of the cached prefix, when the turn
  starts after position 0. They are the same in every picked layer, so a
  stage replays at most two ubatches of the prefix. A prefix of one
  position is picked whole.
- **Head positions.** Four positions from which the turn sampled a token.
- **Attention picks.** Two of the positions of each picked layer, with a
  head draw for each.

The client sends `pi_c` with the challenge. Every stage checks the proof
against the client key and the turn root before it replays. The node thus
opens only the one challenge that the key draws.

`beta_r` picks what the client checks in the opening: 16 rows of each
matrix, 16 (position, expert) pairs of each MoE layer, and two blocks of 64
context positions for each attention pick. The first block is drawn in
proportion to the weight that the opened partials give it in the
combination, and the second block is drawn uniformly. The node cannot
compute `beta_r` without the secret key of the client. So it knows nothing
about the rows
before it signs the opening. The client sends `pi_r` to the node with the
request for the attention rows. At that time the opening is signed.

A plain Ed25519 signature of the root is not sufficient. A signer can
choose its nonce and make many valid signatures of one message. With the
VRF, the client cannot draw again until the rare outliers of an honest node
fail the audit.

The challenge window is 10 seconds. A stage keeps the turn openable for 12
seconds, and for 30 seconds after the opening so that the client can get
its rows.

### Opening

Each stage replays the turn in a scratch context and opens its part. The
answer holds one opening per stage:

- For each picked layer and position: the residual leaf that enters the
  layer, the mixing leaf, and the residual that leaves the layer, each with
  its Merkle path.
- For each picked position: the input of layer 0 with its token and path.
- At each stage boundary: the leaf of the residual that one stage sent and
  the leaf that the next stage received.
- For each picked layer: the full output column of every weight matmul.
  For a MoE layer, also the router output, the expert ids and the
  selections.
- For each head position: the residual that leaves the last layer, and the
  logits that the replay computed from it.
- For each attention pick: the flash-attention partials (running maximum,
  sum, and weighted value) of one head over blocks of 64 context positions.
  The stage computes them in f64 on its CPU from the committed q and the K
  and V rows in the live cache. The sum of each block is at least 1 and at
  most the number of positions in the block, because the running maximum
  is the largest logit of the block.
- For the first attention pick of a stage on a recurrent layer: the mixing
  leaves of the full ubatch that computed the pick, and of the position
  after it.
- For each picked position of the cached prefix: the hash of every leaf
  that the stage committed at the position, the stored-row leaf of each
  layer of the stage, the index of the KV cell, and the path of the
  position in the prefix tree. The stage that holds layer 0 also sends the
  tokens of the whole prefix.

A leaf of a prefix position is in no tree of the turn, so it has no Merkle
path of its own. The client checks it against the leaf hashes that the
stage opened for the position.

The node signs the bytes of the answer with its node key. The signature
covers the digest of the challenge, so an opening is evidence only of the
challenge that it answers. The opened matmul columns come from a replay and
are in no Merkle tree. Only the signature shows that the node sent them.

After the opening, the client asks for the K and V rows of the blocks that
`beta_r` draws. Each row has a path. When the turn computed the row, the
path goes through the mixing leaf of its position. When the turn read the
row from the cache, the path goes through the cell digest and the prefix
root, and the node names the position digest that the prefix leaf binds
with the cell. The node signs the rows, and the signature covers the
blocks that the client asked for.

A stage that can no longer open a turn says so. The answer names the
stage, a reason (`expired`, `cells_overwritten`, `cells_unknown` or
`state_dropped`) and a detail. For a bound receipt, the node signs this
answer too.

### Verification

Before it sends a challenge, the client decides from the receipt alone
whether it can audit the turn. If it cannot, the result is no audit. The
node cannot change this decision after it sees the challenge.

Then the client does these checks, in this order:

1. **Receipt.** The receipt is bound to the key of the client. The turn
   root matches the fields, and the signature of the node key is correct.
   The stages cover every layer of the claimed GGUF from 0, in order. Each
   stage has the leaf count that its positions and layers give. Each stage
   commits a prefix exactly when the turn starts after position 0, and
   only the stage that holds layer 0 commits the tokens of the prefix.
   Every token is one that the model embeds. The layer count comes from
   the metadata of the GGUF, not from the receipt.
2. **Request.** The receipt has the id that the client asked for. The
   sampler statement does not contradict the sampling parameters that the
   client sent. Each number that the client set must be stated with its
   exact value: the seed and `top_k` as integers, the others as the same
   f32. A statement may leave the tokens or the draw unjudged only for a
   cause in the request: a grammar, a penalty or another sampler that the
   client asked for. These are contradictions: a value outside the ranges
   that a request can set, a sampled turn without the `indexed-v1` draw,
   `ignore_eos` that the client did not send, a `reasoning` sampler on a
   turn without a grammar, and a filter stated off that the client left
   at a default that is on. A sampler statement whose fields do not have
   the types that an honest node writes is a provable failure (step 4 of
   the receipt check).
3. **Form of the opening.** The signature of the opening is correct for
   this challenge. The opening holds exactly the stages of the receipt in
   order, exactly the picked layers one time each, and exactly the picked
   positions. Nothing is there twice, and nothing is extra. Every opened
   leaf is in the tree of its stage at the place that its position and
   layer give. The boundary leaves of adjacent stages are equal. For a
   picked position of the cached prefix, the opened leaf hashes and the
   stored-row leaves must give a leaf of the prefix tree, and every opened
   value of the position must hash to one of those leaves. The stored rows
   in an opened mixing leaf must be the rows that the cell digest covers.
   The tokens of the prefix must hash to the committed digest.
4. **Values.** Every opened value is finite, has the length that the
   shapes of the claimed GGUF give, and has ids in range. A stored KV row
   has a type that its stage names in the receipt, f32 or f16, and the
   correct byte length. The set of matmuls is the set in the plan of the client's own
   graph. A value that fails this step is a failure of the node. A check
   reads only values that passed steps 3 and 4.
5. **Weight rows.** The client reads 16 rows of each opened matrix from the
   GGUF. It computes those rows of each matmul output on the CPU: reference
   dequantization, the rounding profiles of the backend family for the
   weight type, and f64 accumulation. It compares the result with the
   opened column, by the [fail rule](#fail-rule). A router is checked on
   every row.
6. **Glue.** The client builds each picked layer with its own llama.cpp
   from the claimed GGUF and computes it on the opened values. It compares
   the matmul inputs, the expert ids, the token-mixing inputs, the stored
   rows, the selections and the residual that leaves the layer. See
   [RECEIPTS_GLUE.md](RECEIPTS_GLUE.md).
7. **Input of layer 0.** The client computes the input of layer 0 from
   each token with the claimed embedding and compares it with the opening.
8. **Output head.** The client checks the opened logits with its secret
   head key (a Freivalds check over the full vocabulary), 16 rows of the
   output matrix, and the row of the sampled token. Then it checks the
   sampled token. A greedy turn must take the best logit. A sampled turn
   must take a token that its stated `top_k`, `top_p`, `min_p` and
   temperature keep. When the statement names the `indexed-v1` draw, the
   token must also be the one that the draw picks (see
   [Sampling draw](#sampling-draw)). The rule follows the runtime: an
   empty chain is the default chain, a `top_p` of 0 keeps the best token,
   and the tokens that the model suppresses (`tokenizer.ggml.suppress_tokens`)
   and, for a turn with `ignore_eos`, the end-of-generation tokens are out
   of every candidate list before the rule applies. A token that the
   opened logits refuse is a failure that evidence shows, whatever the
   key found of those logits.
9. **Attention.** The sum of each block must be between 1 and the number
   of positions in the block. The partials of each picked head must
   combine to its slice of the committed attention output. The client
   draws one block in proportion to its weight in that combination, and
   one block uniformly. The rows of the drawn blocks
   must be the committed stored rows. The client recomputes the partials of
   those blocks from the committed q in f64. For a recurrent layer, the
   client runs the token-mixing ops of the layer over the opened region and
   compares the outputs and the states.
10. **Response.** The response text must be the text of the sampled tokens,
    up to whitespace and reasoning markers. A response that the node cut at
    the token limit (`finish_reason` is `length`) must be the start of that
    text. It can leave out only an unfinished character or the start of a
    marker, at most 16 characters. The message that the client passes,
    usually its last user message, must end in the text of the prompt
    tokens of the turn. A short message could otherwise match an earlier
    turn in the cached prefix. When the prefix holds all of the prompt but
    its last token, as for a request that a client sends again, the
    message must end the prefix, with at most 128 characters of template
    text after it. The system prompt and the history are not compared, and
    the report says what was compared. The node signs neither text, so
    only this client sees a failure here.

A comparison that gives a non-finite error fails.

The client keeps one secret: the head key of each model. A node that knows
the key can change logits without detection. The key has 16 secret sign
vectors. A node that lowers one logit and raises another by the same
amount goes unseen only when the two rows have the same sign in all 16:
1 in 65,536 for rows that the node picks without the key. A key file has
a version, and a file of another version is built again, never read. A
report shows whether the key accepted the logits and how far from its
tolerance they were in powers of ten, never the error itself: a node that
read the exact errors of its probes could learn which rows share signs.

When a check with the key refuses the logits of a node, the client
deletes the key, and the next audit builds a new one. It also checks the
logits against every row of the output matrix, as a recheck does, and
reports that verdict. So the audit and a recheck of its evidence agree,
and the failure is one that evidence shows.

A recheck of evidence has no key. It reads the full output matrix and
compares the opened logits with every row, with the rounding profiles of
the backend of the node. It fails when the difference over all rows is
more than 1e-5 of the norm of the logits, which is the tolerance of the
head key. It also does the row check of the audit, on the same rows. Then
it judges the token on the opened logits, as the audit does. So the audit
and the recheck give one verdict on the output head. The recheck does not
compute the logits with its own rounding and judge the token on those:
without the rounding of the node, logits were up to 4% of their root mean
square from the logits of an honest CPU node, which is more than the tie
band.

A model that soft-caps its logits is the exception. Its opened logits are
not the product of the output matrix, so no check can tell them from
other logits. Logits that the client computes lack the rounding of the
node, and judged on them a token of an honest node could fail. So the
tokens of such a model are not judged, and the report says so. The audit
and the recheck still check that the opened final residuals fit the
output head.

### Sampling draw

The `dist` sampler of llama.cpp draws from a `std::mt19937` stream, over
the candidates in the order that the earlier samplers left. Neither is the
same on every platform, so a client cannot replay it. A sampled turn that
gets a receipt uses another last sampler, the rule `indexed-v1`:

- The uniform of the token at position `i` of the sequence is a function
  of the seed and of `i`: the 53 high bits of SplitMix64 over
  `seed << 32 | i`, divided by 2^53.
- The sampler takes the candidates that the chain kept, in token id order.
  It weighs each one by the exponential of its logit after the
  temperature, and picks the first candidate whose running weight is more
  than the uniform times the total.

The node states the seed in the sampler statement. A request can fix the
seed in its body (`seed`) or in its receipt request (`x-receipt-seed`, a
decimal number above 0). The seed of the body comes first, then the seed
of the header, then a default seed of the operator. The header does not
turn sampling on for a request that the node decodes greedily. When the
request fixes no seed, the node takes a random one and states it.

The replay without the band follows the arithmetic of the runtime: its
filters work in f32, and an f32 `top_p` can cut the candidates one token
before an f64 sum would. A temperature so small that the weights overflow
leaves the runtime the last candidate in token id order, and the replay
picks that one too.

The client replays the draw at each head position, on the logits that the
node opened. It finds the interval of uniforms that pick the sampled
token, and the token passes when the uniform of its position is in the
interval. The client widens the interval by the draw band: each logit can
be 0.01% of the logit root mean square higher or lower, and a candidate
at the edge of a filter can be kept or dropped. A draw that passes only
in the widened interval is counted. More than one such draw in four
positions is flagged, not failed.

The draw band is 100 times narrower than the tie band of the greedy
check. The node drew from the logits that it opened, so the replay
follows the node exactly. Only the last bits of `exp` can differ between
the libraries of the node and of the client. On 2026-11-10, every honest
draw was exact with no band: about 130 draws on CPU, 120 on CUDA with
Q8_0 and Q4_K_M weights, and 160 on Vulkan before Vulkan lost its
profiles. A wider band gives a node room to steer the draws near a
boundary and gives an honest node nothing.

A draw outside the interval is a failure that evidence shows. The
statement, the tokens and the opened logits are all signed by the node, so
`skippy recheck-receipt` replays the same draw on the same logits and
gets the same result.

The replay shows that the node did not choose the token only when the
client fixed the seed. With a seed that the node chose, the replay shows
that the node followed one seed. The report says which case applies.

The client does not replay the draw of a turn with a dynamic temperature.
Such a turn keeps the kept-token check alone, and the client must have
asked for the dynamic temperature. A turn at a temperature of 0, or with
a `top_k` of 1, draws nothing: its chain keeps the best logit alone, and
the kept-token check with the tie band is the whole judgement.

### Reasoning budget

A thinking model gets a reasoning budget on an ordinary request. The
sampler of the budget counts the tokens after the tag that starts the
reasoning. When the count reaches the budget, the sampler forces the tag
that ends the reasoning, one token at a time: the forced token is the
only candidate that it leaves. At every other token, the sampler changes
nothing.

A forced token is not the choice of the logits. So the client must know
which tokens the sampler forced. The state of the sampler follows from
two things:

- What the runtime built the sampler from. The node states this in the
  `reasoning` field of the sampler statement: the start tag (`start`),
  the end tags (`ends`), and the text that the chat template placed after
  the prompt (`prefill`). The sampler reads the prefill first. The
  statement names `reasoning` only when the runtime builds the sampler:
  the turn has a budget, and the chat metadata names the tags.
- The tokens that the turn sampled. The receipt names them.

The client turns the stated texts into tokens with the tokenizer of the
claimed model. Then it runs the same state machine over the prefill and
the sampled tokens. The machine is a port of `common/reasoning-budget.cpp`
of the pinned llama.cpp (`head::reasoning_budget` in `skippy-receipts`).
At a head position where the machine forces a token, the sampled token
must be the forced token. Any other token is a failure that evidence
shows, and `skippy recheck-receipt` finds the same failure. At every
other head position, the usual rule applies: the greedy choice, the kept
tokens and the draw.

The client reads the statement as hostile input. A start or end tag has
at most 256 bytes, the statement names at most 8 end tags, and the
prefill has at most 1,024 bytes. A statement outside these limits leaves
the tokens of the turn unjudged, and the report gives the reason. The
report shows the stated tags and the number of checked tokens that the
sampler forced.

The client does not derive the tags from the chat template. So a node
can state an end tag of its own, and its tokens then pass as forced
tokens. The stated tags are in the signed statement and in the report. A
turn with a lazy grammar stays without a token judgement.

The runtime builds the sampler for a turn with a finite budget, with a
grammar or without, when the template names both tags and did not close
the reasoning block itself (`chat_sampling_metadata_for_native` in
`skippy-runtime`). The statement reads the chat metadata through the same
function, so it names a sampler exactly when the runtime built one. A turn
with no finite budget, or with reasoning turned off, states `reasoning` as
`null`, and the client judges every token of it by the usual rule.

An audit holds the statement against the request in two ways. A
`reasoning` sampler without a stated finite budget is a contradiction. The
stated budget may not be larger than the budget of the request, and a
request that allowed any number of reasoning tokens contradicts every
stated budget. A request that named no budget gets the default of the
node, so any stated budget is consistent with it. The stated prefill must
be the end of the text of the prompt tokens, whitespace aside. When it is
not, the forced tokens are not judged, and the report says why. A recheck
has no request, so it reads the statement as the node wrote it.

Before 2026-11-11 the runtime built the sampler only for a chat with a
grammar, and the client held a `reasoning` sampler on a request without a
grammar as a contradiction. The runtime now enforces the budget for every
chat, so that rule is gone.

### Fail rule

For each opened matrix and position, the client computes the relative
error over the checked rows:

```text
e = || y[R] - y_ref[R] || / max( || y_ref[R] ||, || x || * || W[R] || / sqrt(K) )
```

`y` is the opened column, `R` is the set of checked rows, `y_ref` is the
output of the client, `x` is the opened activation after the rounding
profile, `W[R]` is the checked rows of the weight, and `K` is the row
length. Every term of the scale comes from the client: its own weights and
the activation, which the glue check binds to the computation of the
layer. No opened output changes the scale. The second term is the norm
that the checked rows give for an activation of the same size in no
particular direction. Rounding moves an output by an amount in proportion
to it, but the reference of a row can be small by cancellation. Without
the second term, 4 of 46 honest `attn_v` columns of layer 0 failed, on
CPU and CUDA stages. See
[the measurement](RECEIPTS_MEASUREMENTS.md#fail-rule-2026-11-11).

Until 2026-11-11 the scale was `|| y || * sqrt(|R| / M)`, over every opened
output. A node could then put one value of 1e6 in a row that the client
did not draw and hide any error in the checked rows. A later op cancels
such a value: a gate of -1e6 gives silu 0, so its up value is free too and
the input of `ffn_down` does not change. That let a node serve `ffn_gate`,
`ffn_up`, `attn_q` and `attn_k` from other weights.

A weight role (`attn_q`, `ffn_down`, and the others) fails when more than
`2 + n/32` of its `n` opened columns, across all opened layers, have
`e > 1e-3`. That is 2 of 16 columns. It also fails when one column has
`e > 1e-2`, whatever the allowance: the allowance is for honest columns a
little past the tolerance, and the largest honest `e` measured is 1.7e-3.
A router of f32, f16 or bf16 weights fails on one failing column. A
quantized router has the allowance of the other roles: at the measured
honest rate of 1 in 8,350 failing columns, it would otherwise fail an
honest node about 1 in 400 audits. Stock `llama-quantize` keeps
`ffn_gate_inp` in f32. The audit fails when one role fails. The rule is
per weight role, so the substitution of one tensor type fails too.

A majority rule was rejected. It lets a node cheat on up to half of the
opened positions, for example on every decode token of a turn with a long
prompt.

**Near-tie repair.** Most honest column failures are near-ties in the
activation. The backend rounds the activation to the grid of its profile.
The value that it rounds can differ from the opened one by a few ulps. A
fused kernel computes the value again, or it divides with a fast
reciprocal. A value on a rounding boundary then rounds the other way. The
output of the column moves by one step of that value times its weight
column.

Before a column fails, the client moves each value of the opened
activation by 8 f32 ulps. It lists the values that then change their
rounding, or the scale of their block. It rounds at most two of them the
other way. It
keeps the result only when the error of the column decreases to a quarter
of the tolerance on every checked row. This gives the node no freedom. A
flip changes the reference by a fixed quantity that the node cannot know
before the client draws the rows. The values that can flip are in the
opened activation, which the glue check binds to the computation of the
layer. The output head uses the same repair against its key.

The allowance of `2 + n/32` absorbs what the repair does not explain. The
honest failure rates are in
[RECEIPTS_MEASUREMENTS.md](RECEIPTS_MEASUREMENTS.md#honest-failure-rate-and-near-tie-repair).

### Evidence

The evidence of an audit holds:

- the signed receipt
- the challenge with `pi_c`
- the bytes of the opening with the signature of the node
- `pi_r`
- the attention rows with their signature

`skippy recheck-receipt` first checks that the file is evidence:

- The node signed the receipt, the opening for this challenge, and the
  rows for the drawn blocks.
- The challenge is the one that the VRF of the turn root draws.
- The rows are the ones that the VRF of the opening draws.

Then it does the checks again, without the request check and the response
check. The weight rows are public, and both VRF
proofs show that the client did not choose the challenge or the rows.
Evidence without attention rows is incomplete, not failed. A challenge
that a user picked by hand keeps no evidence.

### Node: commitment and replay

The node does not store residuals. It keeps leaf hashes and a replay log
for each turn, and it computes an opening again in a scratch context.

**What a bit depends on.** The bits of a lane depend on its own tokens and
on the geometry of each ubatch that held it: the ubatch size, the column
offsets, the KV cells, `n_kv` and the graph flags. They do not depend on
the values in the other lanes. A replay keeps the recorded geometry and
puts dummy tokens in the columns of the other lanes.

**The replay log.** Per stage, the log holds:

- Per ubatch: tokens, sequence ids, positions, output flags, the KV cell
  indices and `n_kv` of each cache, and the graph flags. Native MTP turns
  on `embeddings_nextn` for every prompt row, which changes the shapes and
  the rounding of the last layer.
- The memory operations at the KV cache and the recurrent state: remove,
  copy, keep, shift, clear, state read, compaction.
- For a stage after the first: the imported activation rows. Raw f32 on
  the wire is lossless, so these rows are the committed outputs of the
  stage before.

**Replay of the turn only.** A replay from the first token of a
conversation increases with the conversation, and a 10-second window
cannot absorb it. So the node replays only the turn, on top of the KV
cells that the turn read. It mirrors which cell holds which position of
which sequence. When it opens a turn, it stops the stage for a moment. It
makes sure that those cells still hold what the turn read. Then it copies
their K and V rows into the same cells of the scratch context.

When a turn wrote each position one time and in order, the node replays
only the ubatches that hold a picked position. Each of them starts from a
copy of the cells that were written before it. Other turns replay in full.
This is not weaker for the client. The client checks the opened values
against leaves that the node committed before the challenge, and the
method of the recomputation does not matter.

**Replay of a prefix position.** The turn did not compute its prefix, so a
replay of the turn does not recompute it. For every KV cell, the node
keeps the leaf hashes of the position that the cell holds and the recorded
ubatch that wrote it. When a challenge picks a position of the prefix, the
node replays that one ubatch, from a copy of the prefix cells below it. It
compares the result with the leaf hashes that it kept, as it compares a
replay of the turn with the turn's tree.

Another sequence may have computed the prefix: a cache sequence, or a
request whose cells the turn shares. The replay runs in the sequence of
the turn. The columns of the writing sequence take the number of the
turn's sequence, and the other columns are dummies.

A copy of the cells below a ubatch is what that ubatch read only when
those cells did not change since. The node tracks this with a stamp per
sequence. A ubatch that appends to a sequence notes the stamp that it
finds and leaves a new one. A cut at the end of a sequence and a copy of a
sequence from position 0 give the stamp of the cells that remain. Any
other change leaves the sequence without a stamp until it is empty. A turn
gets a receipt only when the stamps of its prefix follow each other from
position 0.

The leaf hashes cost 32 bytes per leaf: about 2.3 kB per position for a
stage of 36 layers, less than 2% of the K and V rows of the cell. A stage
after the first also keeps the input rows of each ubatch, one residual row
per token. It keeps at most 1 GiB of them, and drops the oldest ubatch
first. A turn whose prefix reads a cell of a dropped ubatch gets no
receipt.

**Pins.** Serving frees the cells of a turn when its response ends. A
freed cell keeps its rows until a ubatch writes it. Each stage pins the
cells of a bound turn for the challenge window, so a later request writes
other cells while the cache has room. When a challenge of the turn
arrives, the stage pins the turn again until the opening ends, for at
most 20 s. The opening reads the cells after the replay, and the replay
of a long prompt on a CPU takes seconds. Without this pin, a request that
runs during the replay can write the cells of a turn whose 12 s pin ran
out, and the stage answers "cannot open" to its own honest turn. A full
cache writes pinned cells anyway, and the turn can then no longer be
opened.

Pins are capped, because every pinned cell slows the requests after it.
The pinned turns of one client key hold at most half of the cells of a KV
stream, and all pinned turns together at most three quarters. A new
bound turn always gets its pin and its receipt. Past a share, older pins
give way: first the pins of turns that no client has challenged, the
oldest first, then the pins of turns that wait for their rows request. A
turn that lost its pin can still be opened until a ubatch writes its
cells. A turn that is past a share alone is pinned alone.

**What invalidates a turn.** A shift of positions, a load of saved state
or a compaction makes the affected turns unopenable. The stage then
answers a challenge with "cannot open". It does not open wrong values.

A ubatch that the recorder cannot record, such as a ubatch of image
embeddings, costs only the turns that it could touch. The native side
does not say which sequences or cells such a ubatch wrote. So the
recorder frees the cells that no sequence held and no pin held, and marks
the pinned free cells as exposed. Before it opens a pinned turn, the stage
reads the rows of each exposed cell again and checks them against the
digest that it committed. A cell that still holds the rows opens. A cell
that the ubatch wrote gives "cannot open". A turn that was running when
the ubatch ran gets a receipt only from its own start, and only when it
recorded every position.

**What gives no receipt.** A turn gets no receipt when the node cannot
replay a position of its prefix. That is the case after a shift, a state
load or a compaction, for a cell that an unrecorded ubatch wrote, for a
prefix that holds cells of two requests, for a ubatch past the input
budget, and on a stage with recurrent state.

**Anchored replay.** A replay reports more tensors than the recorder: the
matmuls and the selections of the picked layers. A reported tensor cannot
be fused away, so a backend with op fusion computes those layers with
fewer fused kernels than the turn did. The residuals of the replay can then
drift from the committed ones. When a full replay misses a committed leaf,
the node replays two more times. The first replay reports only what the
recorder reports, and it reproduces the committed values. The second
replay writes the committed residual that enters each picked layer, and the
committed token-mixing outputs, into the columns of the picked positions.
The matmuls then follow from the committed values.

**KV streams.** A KV cache that is not unified keeps a stream of cells per
sequence. The mirror then knows a cell by its stream and its index. A
replay puts each token in the stream of its sequence, and a pin names the
stream.

**Stages.** Each stage keeps its own log and replays its own layers.
Stages can batch differently. A stage commits and opens the next stage
over a dedicated receipts connection on the binary stage protocol. In a
Mesh split, that connection crosses peers.

**Hashing.** On CUDA, the recorder hashes each committed value on the GPU
and copies the 32-byte digests. Other backends copy the values to the
host, and a worker thread hashes them while the next ubatch computes. The
leaves are the same in both cases. A GPU that hashes wrongly fails every
audit, because the auditor hashes the opened values on its CPU.

The node keeps these things for the window:

- the replay log of the turn
- the cells of the prefix, each with its position digest and its token
- the pinned KV cells
- for a hybrid model, the recurrent state at the start of the turn
- a scratch context

**Limits of a serving node.** Receipts must not slow ordinary serving, and
a node must not hold a receipt that it cannot open. So each stage limits
what turns and challenges can hold, and it refuses new work before it
drops a turn that a client can still challenge.

| Limit | Value | When it is reached |
|---|---|---|
| Records of committed turns and logs of uncommitted turns, with their ubatches, position hashes, prefix cells and recurrent state | 1 GiB per stage | A turn that does not fit beside the turns inside their challenge window gets no receipt. Only turns past their window are dropped for room. A live log that outgrows the window is shed: its turn gets no receipt, and it keeps nothing more for that turn |
| Input rows for prefix replays | 1 GiB per stage after the first | The rows of a turn inside its window are kept. Newer traffic loses its rows first, and a later turn whose prefix needs them gets no receipt |
| Pinned cells | Half of a KV stream per client key, three quarters in total | The oldest pins give way. The new bound turn keeps its receipt |
| Openings and rows requests at work | 1 per stage | The others wait |
| Requests that wait | 8 in total, 2 per client key, 8 s each | HTTP 503 with `Retry-After: 2`. The challenge is not admitted, so the client can send it again |
| Rows requests that read rows | 4 per receipt | A repeated request gets the kept answer. A fifth new request is refused |
| Stage connections | 8 | A new connection closes the one that waited longest for a request |
| Detail of a later stage's "cannot open" | 256 bytes, no control characters | Longer text is cut. An answer for another receipt or an earlier stage is not signed |

When several requests wait, the client key that used the least time since
the stage was last idle goes next. So one key that sends many challenges
cannot keep the challenge of another key waiting. The turns bound to no
key count as one key.

The busy answer is not signed. The receipt format has no busy answer that
a node signs, and a signed one would need a new reason in the wire
format. A request that the stage before forwards over the stage
connection is not queued again on a later stage, so a later stage does not
answer busy.

`GET /receipts/{id}` needs only the receipt id. For a bound receipt it
returns the receipt without its tokens and final token. The client of a
bound turn gets the whole receipt with its response.

## What a pass proves

A pass is a statement about the picked layers, the picked positions and
the picked sampled tokens of one turn. For those, a pass shows:

- The node used the claimed weights, including the claimed quantization,
  in every weight matmul of the four picked layers.
- The operations between the matmuls (norms, RoPE, activations, residual
  adds, MoE routing and combine) are those of the claimed model in the
  client's llama.cpp.
- The input of layer 0 is the embedding of the token that the receipt
  names, and the stages agree at their boundaries.
- At the four head positions, the logits come from the claimed output
  matrix, and the sampled token is one that the stated sampler can give.
- The response text is the text of the tokens that the receipt names.
- At two positions per picked layer, the attention of one head agrees with
  the committed rows within the loose tolerances.
- At two positions of the cached prefix, in the four picked layers, the
  same statements hold as for a position of the turn. The output head is
  not checked there, because the turn sampled no token from the prefix.
- The prompt is part of the text of the tokens of the prefix and the turn.

The node cannot know the picks in advance. A node that cheats on every
token fails almost every audit.

A pass does not show these things:

- **The other positions and layers.** The check is a sample. A node that
  changes a small part of a turn is caught with a small probability.
- **Most of the cached prefix.** The check of the prefix is a sample of two
  positions. A node that computed the whole prefix with another model
  fails at both. A node that changed one token of a long prefix is caught
  with a small probability.
- **A draw near a boundary.** The client replays the draw of a sampled
  turn within the draw band, 0.01% of the logit root mean square. A node
  can steer a draw only when its uniform is that close to the boundary
  between two tokens.
- **A seed that the node chose.** When the request fixed no seed, the
  replay shows only that the node followed the seed that it states.
- **Samplers beyond the filters.** A turn with penalties, DRY, XTC, typical
  or top-n-sigma sampling, mirostat or a grammar gets no token judgement.
  A turn with a dynamic temperature gets no replay of its draw. The
  client must have asked for each of them. A recheck has no request, so
  it cannot tell, and it notes the gap.
- **Soft-capped models.** Their tokens get no judgement.
- **The end tag of a reasoning budget.** The client replays which tokens
  the budget forced, with the tags that the node states. It does not
  derive the tags from the chat template. The report shows them.
- **Exact attention.** A change of attention within the tolerances passes.
- **The file hash.** The client trusts Hugging Face to serve the bytes of
  the named commit over TLS.
- **The full prompt.** The prompt check finds the message that the client
  passes at the end of the prompt of the turn. The chat template adds
  text, and the check does not show that the node added no other text,
  or that the system prompt and the history are the ones that the client
  sent.
- **Turns without an audit.** A Mesh node audits 10% of turns.
- **Privacy.** The node reads the prompt and the response.

## Threat model

**The client trusts** its own machine, its own Skippy and llama.cpp build,
Hugging Face to serve the bytes of a commit, and the people who can write
to the branch that the request names. It does not trust the serving node,
its operator, the text in a receipt or an opening, or the author of a pull
request. The auditor reads every string and every size from the node as
hostile input, and it bounds every read of a claimed GGUF. It reads the
metadata of a claimed GGUF as hostile input too: a node can name a commit
of a repository that it controls.

**A node can do these things without a failed audit:**

- Cheat on the turns that no client audits, and on clients that do not
  audit.
- Refuse to issue receipts, or run a backend or weight type that no
  verifier covers. Its turns then get no audit, not a pass.
- Change attention, K and V rows, or a router output within the
  tolerances.
- Choose between two tokens when the uniform of a draw is within the draw
  band of the boundary between them.
- State an end tag of its own for a reasoning budget, of at most 256
  bytes. The tokens of that tag then pass as forced tokens at the places
  where the budget runs out. The report shows the tag.
- Choose the seed of a sampled turn when the request fixed none.
- Choose a token among those that the stated filters keep, in a turn whose
  draw the client cannot replay.
- Change at most 1% of the weight bytes of a layer that token-mixing
  operations read directly.
- Change the attention of a block whose weight it understates in the
  partials. Only the uniform draw then finds it, with a chance of 1 in
  the number of blocks for each pick.
- Join again with a new key after a block.

**A node cannot do these things without a high chance of a failed audit:**

- Serve another quantization or another model on every token.
- Skip a layer, a matmul or an FFN.
- Change a logit by more than the tie band.
- Return text that is not the text of the tokens in the receipt.
- Continue a turn from a cached prefix that it computed with another model
  or another quantization, or for another prompt.
- Hide wrong weights behind a large value in a row that the client does
  not draw. The error of a column is relative to a scale that the opened
  activation sets, and one column far past the tolerance fails its role.
- Open matmul outputs or selections that the picked layer does not
  compute, or leave some out. The client compares the opening with its
  own plan of the layer, and a value that the evaluation refuses fails
  the audit.
- Change the attention of the block that has most of the weight in the
  combination. The client draws blocks in proportion to that weight, and
  a block sum that no block can give fails the audit.

**A node cannot do these things to a Mesh auditor without strikes:**

- Advertise receipts and leave out the receipt of a turn.
- Sign a receipt with a key that is not its identity key.
- Give no answer to challenges.
- Receipt a smaller model than the request asked for.
- Receipt a commit of a pull request, or of another branch, when the
  request named a branch or no revision.

**A client can do these things:**

- Read the activations of its own turn at the picked positions. It cannot
  read the turn of another client, because a bound receipt opens only with
  a VRF proof for its client key.
- Ask each stage for one replay per bound turn. The challenge must be one
  that a draw makes, so the work is bounded.
- Show a provable failure to anyone. It cannot make one for an honest
  node, because it does not choose the challenge or the rows.

A client cannot prove a failure that only it saw: the response text or the
request. The audit also does not count logits that only its head key
refused as proven, because the key is secret. Such a failure counts
locally. A recheck of the evidence compares those logits with every row
of the output matrix, and anyone who runs it sees logits that are more
than the tolerance from the claimed model.

One attack on a split is open. A later stage holds a bound turn to the
first turn root for which it saw a correct client proof. A party with
access to the stage port and a captured proof can thus lock out the real
challenge. The correction needs authentication between stages.

## Parameters

| Parameter | Value | Source |
|---|---|---|
| Receipt wire version | 7 | `skippy-receipts`, `wire::RECEIPT_VERSION` |
| Audit rate in Mesh | 1 in 10 by default, drawn by the VRF; `[receipts] audit_one_in` from 1 to 10,000 | Decision |
| Layers per challenge | 4, weighted by weight bytes, without replacement | GGUF shapes and types |
| Positions per layer | 4, half of them generated positions | Calibration |
| Prefix positions | 2 of the cached prefix, the same in every picked layer | Bounds the replay work: one ubatch per position |
| Input rows kept for prefix replays | 1 GiB per stage after the first, oldest ubatch dropped first | Memory |
| Head positions | 4 | Decision |
| Attention picks | 2 per picked layer, 2 blocks of 64 context positions each | Calibration |
| Rows per matrix | 16 | Calibration |
| MoE pairs per layer | 16 (position, expert) pairs | Fetch time |
| Expert router | Read whole, checked on every row | A 2D tensor whose row count is `expert_count` |
| Row tolerance | `e > 1e-3` on more than `2 + n/32` of the `n` columns of a role, or `e > 1e-2` on one | Calibration |
| Glue tolerance | 1e-4, relative to the largest value of the reference column | Calibration |
| Q and stored-row tolerance | The glue tolerance plus two f32 epsilons per position | RoPE error increases with position |
| Stored KV rows | At most 1.5 times the client's own rounding into the cache type, plus 1e-4 of the norm of the row | [RECEIPTS_GLUE.md](RECEIPTS_GLUE.md) |
| Attention combine tolerance | 2e-2 for a head that attends to at most about 420 positions. For more positions, `sqrt(positions) / 1024`, which is 6.3e-2 at 4,096 positions and 1.8e-1 at 32,768 | The CUDA prefill kernel accumulates in f16, and the error increases with the square root of the context. See [the measurement](RECEIPTS_MEASUREMENTS.md#attention-combine-error-on-long-contexts) |
| Attention block tolerance | 1e-6 of the largest value of the partial | f64 recomputation |
| Head key tolerance | 1e-5 of the norm of the logits, 4 key vectors | Calibration |
| Head tolerance of a recheck | 1e-5 of the norm of the logits, for the difference over every row | The head key tolerance. Honest logits were within 4.3e-7 on CPU and 1.6e-7 on CUDA |
| Tie band | 1% of the root mean square of the logits | Calibration |
| Draw band | 0.01% of the root mean square of the logits that the node opened. The tie band when the client computed the logits | Every honest draw measured was exact: about 410 draws on CPU, CUDA and Vulkan |
| Reasoning sampler statement | At most 256 bytes per tag, 8 end tags and 1,024 bytes of prefill | Bounds a hostile statement |
| Recurrent tolerance | 1e-3 | A backend can sum the scan in other chunks |
| Unchecked weights inside token-mixing ops | At most 1% of the weight bytes of a layer | Guard 2 of [RECEIPTS_GLUE.md](RECEIPTS_GLUE.md) |
| Challenge window | 10 s. The stage pins a bound turn for 12 s, while a challenge of it waits and is opened (at most 20 s), and for 30 s after the opening | Agentic latency |
| Challenges per receipt | 1 for a bound receipt, 4 for an unbound one | Bounds the replay work |
| Challenge and row draw | ECVRF-EDWARDS25519-SHA512-TAI, a new client key per request | Decision |
| Commitment hash | BLAKE3 | Throughput |
| Block length in Mesh | 7 days | The standard peer block |
| Serving budget | Decode at most 3% slower and prefill at most 10% slower than with receipts off | UX |

The weight types with rounding profiles are f32, q8_0, q4_0, q5_0,
iq4_nl, iq4_xs and the K-quants q2_K to q6_K on the CPU and CUDA, and f16
and bf16 on the CPU. The backend families with profiles are the CPU and
CUDA. Vulkan, Metal and ROCm have no profiles. A stage on a backend
without profiles issues no receipts, and a receipt that names one is not
audited.

A stage names the backend its own matrix products run on, and issues no
receipts when they can run on two. That is the case for a partial
offload: llama.cpp offloads the last `n_gpu_layers` of the model's
`n + 1` layers, the output layer last, so a stage from layer `s` is fully
offloaded when `n_gpu_layers` is -1 or at least `n + 1 - s`, and 99
covers any stage of a 36-layer model. It is also the case for a stage
with no layer offloaded in a runtime with a GPU: llama.cpp's op offload,
on by default, runs the matrix products of a batch of 32 tokens or more
on the GPU while the weights stay on the CPU. Such a stage prefills on
the GPU and decodes on the CPU. On Qwen3-4B Q8_0 it measured errors of
about 1/3,000 in every weight against the CPU profiles, and about
1/2,500,000 with op offload off. So a stage with no layer offloaded is a
CPU stage only when op offload is off or the runtime has no GPU.

The CPU multiplies f16 weights with the f32 activation as it is, so its
profile is the raw activation. CUDA has no profile for f16 or bf16
weights. For f16, decode sums the products in half precision
(`mmvf.cu`), and prefill runs cuBLAS with a 16-bit compute type. No
profile of the activation models either, and an honest CUDA node with
f16 weights failed every audit: 278 of 1,120 columns were past 1/1,000,
the largest at 1/340. bf16 was never measured on CUDA.

A stage reads the weight types from the tensor directory of its GGUF
files when it starts. When a matrix of its layers, or the output head on
the last stage, has a type without a profile on its backend, the stage
issues no receipts, as on a backend without profiles. A Mesh node then
does not advertise receipts for the model. A receipt of such a stage
from an older node is not audited: the verifier finds the type before it
sends a challenge, for the head on the final stage's backend too.

The auditor refuses to read a claimed GGUF over these limits. A refusal is
no audit and says nothing about the node.

| Limit of the auditor | Value |
|---|---|
| Header bytes, over all shards | 256 MiB |
| Tensors, and metadata entries | 65,536 each |
| Name length | 1,024 bytes |
| Metadata string | 64 MiB |
| Depth of metadata arrays | 2 |
| Shards | 128 |
| Values in one decoded row | 16,777,216 |
| Tensor outside the layers that is read whole | 256 MiB |
| Bytes fetched per audit, the header included | 3 GiB |
| Sparse copies kept under one cache root | 8 GiB, the copies used longest ago removed first |
| Commit list read for the branch check | 4 pages of 1,000 commits, 4 MiB per page, 30 s per request |

The fetch budget comes from the largest output matrix that the auditor
reads whole: Qwen3.5-397B-A17B has a vocabulary of 248,320 and a width of
4,096, which is 1.89 GiB in bf16 or f16, 1.01 GiB in q8_0 and 0.78 GiB in
q6_K. A header takes about 12 MiB, and the rows and layers of one audit
took 10 to 14 MiB in measurements, with expert routers of up to 8 MiB
each. That is about 1.9 GiB, and the budget adds about half again. A model
with an f32 output matrix of that size gives no audit.

Two tables are maintained by hand: the rounding profiles per backend and
weight type, and the tolerances. Neither is per model. A missing entry
gives no audit for that backend and type. It does not give a pass.

A receipt that names a quantized KV cache type (q8_0 or q4_0) gives no
audit either. The attention check has no model of the error of quantized
rows at prompt positions (see [Open limitations](#open-limitations),
item 8). The client reads the cache types from the receipt, so it decides
this before it sends a challenge.

**Rounding profiles follow the llama.cpp pin.** A profile models what one
kernel of one llama.cpp commit does. A new pin can change a kernel, and
the profile then fails honest nodes. This occurred for the CUDA K-quant
profile `q8_32_mul_dh_sums`: llama.cpp #31247 changed the block sum to
which the MMQ kernels apply the minimum of a weight. The sum is now the
f32 scale of the block times the sum of its quants, stored as f16. Before,
it was the sum of the unquantized activation. The verifier models only the
rule of the current pin, because a receipt names its runtime and a
verifier of another runtime does not audit it.

It occurred again for f16 weights on the CPU: llama.cpp 12ab59798 made
the f16 matmul take the activation in f32 (`vec_dot_type` of f16 is now
f32). The profile rounded the activation to f16, and an honest CPU node
with f16 weights failed 9 of 9 audits at the output head, with errors
from 1/10,500 to 1/2,150 against a limit of 1/100,000. The profile is
now the raw activation.

Maintenance rule: after each change of the llama.cpp pin, run honest
audits and cheating audits again for each backend and weight type, on real
hardware. CI has no GPU and cannot find a changed kernel. The results for
the current pin are in
[RECEIPTS_MEASUREMENTS.md](RECEIPTS_MEASUREMENTS.md#cuda-k-quant-profile).

The parameters of the Mesh audit are in
[Audit policy in Mesh](#audit-policy-in-mesh).

## Audit policy in Mesh

The code is in `mesh-llm-host-runtime`, `network/receipt_audit/`.

### Which turns a node watches

A Mesh node watches a turn when all of these are true:

- `[receipts] audit` is on, or the trust policy of the request requires
  receipts.
- `[receipts] offline` is off.
- The request is a chat completion that the node forwards to a peer for a
  local client. A `/v1/messages` or `/v1/responses` request that the node
  sends to the peer as a chat completion counts too. The node audits the
  chat completion of the peer, before it translates the answer for the
  client, for streamed and whole answers.
- The client did not name a receipt key of its own.

The node names a new client key on the request, and a new receipt id for
each attempt in `x-receipt-id`. The node removes an `x-receipt-id` that the
client sent. It reads the response as it relays it. It reads a request body
with chunked framing as its entity. When a request that requires receipts
cannot be read, the node refuses it with 409 and does not send it
unwatched. Every watched turn ends in one of four verdicts, and the node
writes one log line for each.

| Verdict | Meaning | Action |
|---|---|---|
| Passed | Every check ran and passed | An info line |
| Failed | A defect is in something that the peer signed | Keep the evidence. Block the peer for 7 days |
| Strike | The peer did something that only this node saw, or it gave no usable answer | Count one strike against the peer |
| No audit | The node did not audit the turn, or it was not able to | An info line with the reason |

### What the node holds a peer to

The node holds a peer only to what the node knows without the peer:

- the request that it sent, with its id and its sampling parameters. When
  the request fixes no seed, the node names a random one in
  `x-receipt-seed`, and holds the peer to it
- the identity key of the peer
- the Hugging Face file that the model name of the request stands for

The audit does not open a repository because a receipt names it. That
rule prevents two attacks. A peer that was asked for a large model can run
a small model and receipt it honestly. A peer can choose which repository
this node resolves, fetches and loads with the token of the user. The
gossip descriptor of the peer is used only for its `receipts`
flag. Only the announcement of the peer itself sets that flag. A node
clears it on a descriptor that a bridge relayed, so a bridge cannot send
turns that require receipts to another peer, or have that peer struck for
receipts that it never offered.

A serving node sets that flag for a model only when its receipts name a
file of a Hugging Face repository at a commit. This is the test that an
auditor applies before it fetches anything. A model that a node serves
from a local path has the name `local-gguf/sha256-...`. No client can
fetch what that name stands for, so the node does not set the flag for
it, and a request with `x-mesh-trust: receipts` does not reach it.

The node also sets the flag only for a model that clients route by a name
that the node can tie to a Hugging Face file, as described below. So the
flag is not set for a GGUF of the Hugging Face cache that a node serves
by its path. Its receipts name the Hugging Face file, but clients route it
by the name `local-gguf/sha256-...`, and an auditor ties the name of the
request, not the name in the receipt.

On the routing side, a request with `x-mesh-trust: receipts` reaches a
peer only when the routing node can tie the model name of the request to
a Hugging Face file. For any other name it gets
`409 trust_policy_unsatisfied`, because no audit of the turn could run.
A peer of an older version that sets the flag for such a name gets no
request under the tier either.

The node ties a model name to a file in this order:

1. The name is the file: `owner/repo@revision/path/file.gguf`.
2. The loaded model catalog of this node lists the name: by name, exact
   ref, file or file stem.
3. The name is a model reference with a selector: `owner/repo:selector` or
   `owner/repo@revision:selector`. Each GGUF of the repository that the
   selector matches qualifies.

A `#profile` suffix is ignored. A revision is part of the identity only
when it is a commit. These names cannot be audited: a runtime name that
the catalog does not list, an alias, `local-gguf/sha256-...`, `owner/repo`
without a selector, and a direct URL.

The receipt must then name that file as `repo@commit/file`, with a full
commit. When the request named a branch or no revision, the commit must be
in the history of that branch, or of `main`. The node checks this before
it fetches a byte of the file, on every drawn turn. A commit that is not
on the branch counts as a strike, the same as a receipt that names another
file.

### Every watched turn

A peer that advertises receipts owes one for every turn. A missing receipt
is a strike and not "no audit". If a missing receipt has no cost, a
cheating peer leaves out the receipt of each turn on which it cheated. So
the node checks the receipt of every watched turn, in this order, before
the VRF decides:

1. The response has a receipt that the node was able to read. A 200
   response whose body has an `error` key counts as a response without a
   receipt.
2. The signer of the receipt is the identity key of the peer to which the
   node sent the request.
3. The receipt has the version that this node audits. Another version is
   [unauditable](#unauditable-receipts).
4. The receipt decodes, is bound to the key that this request named, and
   its signature of the turn root is correct.
5. The receipt has the id that this node sent in `x-receipt-id`. A peer
   that does not know that header issues the receipt under the
   `x-request-id` of the request, and the node accepts that id too. When
   the request had no `x-request-id`, an older peer chooses the id, and
   the node accepts any id.
6. The receipt states nothing that no audit covers: a KV cache type other
   than f16 or f32, or a backend that has no rounding profile. Such a
   receipt is [unauditable](#unauditable-receipts).
7. The receipt does not name another model than the request asked for.

A failure of step 1, 2, 4 or 5 is a strike when the peer advertised
receipts for the model at the time of the request. For a peer that did not
advertise them, it is no audit. A failure of step 7 is a strike in both
cases when the request pinned a repository and a commit
(`owner/repo@commit/file.gguf`, or a selector with a commit), because the
peer signed that it ran something else. When the request did not pin
them, the peer chose the file, and another model is no audit. If that
request required the `receipts` tier and the receipt names another
repository, the peer also loses the tier on this node, as for an
unauditable receipt.

Then the VRF of the turn root draws one turn in ten, or one in
`[receipts] audit_one_in` (an integer from 1 to 10,000). A turn that it
does not draw is no audit.

### Unauditable receipts

> **Decision for the owner.** This rule was chosen in the review of the
> receipts change, and the owner can overrule it. The code is in
> `network/receipt_audit/auditability.rs`.

A peer chooses what its receipt states: its version, runtime, backend,
cache types and the model. Some of these this node cannot audit, and such
a turn says nothing about the peer. But a peer that knows on which turns
it cheats could make exactly those turns unauditable. So:

- A peer whose receipt this node cannot audit, for a reason in the receipt
  itself, does not satisfy the `receipts` tier on this node for 24 hours.
  The demotion is in memory, the log says it once, and nothing is
  blocked. Ordinary routing to the peer does not change.
- A peer that is demoted again within 7 days of a demotion that ended
  stays demoted for 7 days. Without this rule, a peer could take one
  request under the tier each day, answer it with a receipt that this
  node cannot audit, and never be audited.
- A peer whose receipt for a model this node audited within 24 hours,
  and that then sends a receipt for the same model that this node cannot
  audit, gets a strike. An honest peer does not change its runtime between
  turns.
- A version skew from a peer that this node never audited is a demotion
  only.

The reasons in a receipt are its version, a cache type, a backend, a
runtime or weight types that no audit covers, and a model name that does
not name a file. A reason of this node, such as a model that it cannot
open, is no audit and does not demote.

### Commits per peer

A request that names a model without a commit lets the receipt choose the
commit, and the first audit of a commit reads its GGUF from Hugging Face.
A peer that names a new commit on each turn makes each drawn turn a cold
fetch, so the node opens at most 2 commits per peer and requested model in
24 hours. A drawn turn past that is no audit, and the log says why. The
budget is in memory.

### Drawn turns

One worker thread audits the drawn turns off the request path. A drawn
turn can add these strikes:

- The peer gave no usable signed answer to the challenge, or to the
  request for rows.
- A check failed that only this node saw: the response text, the sampler
  statement against the request, or logits that only the head key refused
  where the auditor could not check them against every row.
- The answer has a shape that the request did not allow. When the request
  had no tools and asked for one choice, an answer that calls tools, has
  more than one choice, or ends for another reason than `stop` or
  `length` is a strike. `tool_calls: []` counts as no tool calls.
- The receipt is unauditable, and this node audited a receipt of the peer
  for the same model in the last 24 hours (see
  [Unauditable receipts](#unauditable-receipts)).

A drawn turn is no audit in these cases:

- The node cannot tie the model name of the request to a Hugging Face
  file.
- The receipt names no Hugging Face file at a commit: a bare name, a local
  path or a branch.
- The worker is busy, off, or the node shuts down.
- The challenge was not ready in the challenge window.
- The node was too late to ask for the attention rows. The opening was
  checked, and the rows were not.
- The node was not able to open the claimed model, or a check was not able
  to run on this machine.
- The peer named more commits of the model than the
  [budget](#commits-per-peer) allows.
- The audit panicked on this node. The log says so.

A response that did not end whole is no audit for drawn and undrawn turns.
A response that reports an error is a turn: a peer that advertises
receipts owes one for it.

### Timing

The node must not take its own delay for the silence of a peer.

| Step | Rule |
|---|---|
| Challenge window | 10 s after the receipt. The node checks the window when it sends the challenge, after the claimed model is open. A challenge that is too late is not sent |
| Wait for the opening | 15 s from each send of the challenge, also when the node sends it again to a busy peer. The time that a drawn turn waited in the queue of the node does not shorten it |
| Request for rows | Not sent later than 28 s after the challenge |
| Wait for the rows | 10 s |

The first audit of a model opens the claimed model and builds the head
key. That time does not count against the peer: a late step is no audit.

A peer can hold one audit for at most 35 s: the window when it answers
"busy", the wait for the opening and the wait for the rows. The time to open the claimed model and to check the
turn is the node's own, and has no bound. Each read from Hugging Face
has a timeout of 120 s, and a turn that waited in the queue past the
window is not opened. When one audit holds the worker for more than 35 s,
the log says so.

### Strikes

Three strikes in 24 hours block the peer for 7 days. The rules are:

- The node keeps strikes in memory, per peer. A restart clears them.
- A strike ends by age only. A pass does not clear a strike. With the
  opposite rule, a peer answers one audit in two and stays unblocked.
- The third strike blocks the peer, and the count starts again after the
  block.
- The node keeps strikes for at most 4,096 peers.

"No answer" is thus a strike and not a failure. One lost answer does not
block an honest peer, and a peer that does not answer many times is
blocked.

### The response check

The response check runs only when the response of an honest peer must be
the text of the sampled tokens. It does not run in these cases:

- The request has stop strings, tools, a response format, more than one
  choice, log probabilities, an echo, or content that is not text.
- The request continues an assistant message, or asks for a reasoning
  format that the check does not know.
- The answer calls tools, has more than one choice, or does not end as
  plain text.
- The sampled tokens hold control markers other than `<think>`.
- The answered text is longer than 4 MiB.

A response that the peer cut at the token limit must be the start of the
text of the sampled tokens. It can leave out only an unfinished character
or the start of a marker.

The prompt check runs only for a prompt that is one text of ASCII
characters.

### Bounds on what a peer sends

- The node reads an opening up to twice the size that its own challenge
  can produce for the claimed model, and never more than 128 MiB. An
  opening of a stage with recurrent state has only the 128 MiB bound.
- The node keeps a response line or a JSON body up to 16 MiB. A receipt in
  a larger line is not seen.
- The node escapes each text that a peer chose and cuts it at 240
  characters before it logs the text or keeps it in a block reason.

### The worker

The node owns the worker thread. It starts with the first drawn turn and
is joined at shutdown. The queue holds two turns. A drawn turn that finds
the queue full is no audit, and the log gives the number of such turns
since the node started. A panic in one audit does not end the worker: the
turn is no audit, and the log says so. If the thread ends, the node starts it again, at
most three times. Then audits are off until the node starts again, and
the log says so. The worker keeps two claimed models open and releases a
model after 10 minutes without an audit.

### Effects on honest serving nodes

A serving node can get strikes from its own defaults:

- Its defaults add a penalty or a logit bias, or drop a sampler filter,
  for a request that set none. Its sampler statement then contradicts the
  request.
- Its defaults turn on `ignore_eos`, a dynamic temperature or another
  sampler, or turn a filter off. A request that left those unset then
  contradicts the statement too. A default seed does not: a seed in the
  receipt request comes before it.
- Its defaults add stop strings. They cut the text, and the response
  check fails.
- Its runtime answers a turn with a 200 body that has an `error` key and no
  receipt, for example when generation fails during a stream. That counts
  as a missing receipt, and so as a strike.
- Its layer package defaults to a quantized KV cache. Then the runtime
  issues no receipts and the node does not advertise them. When
  `[receipts] enabled` is on, the node warns at load and names the
  setting that fixes it: `model_fit.cache_type_k` and
  `model_fit.cache_type_v` set to `f16` in the `[[models]]` entry.

An operator of a serving node with receipts must keep such defaults off.

The auditor counts an answer other than HTTP 200 as a strike, a signed
"cannot open" included. A node that declines to issue a receipt causes no
strike, because the turn has no receipt to challenge. So the serving
node's limits turn away new turns first.

A busy stage is the one answer the auditor asks for again. When the
answer is 503 with a `Retry-After`, the auditor waits that long, at least
1 s, and sends the same request again. It does so while the node still
keeps what the request needs: 10 s after the receipt for a challenge, and
28 s after the challenge for the attention rows. A stage that is still
busy then gave no answer, and that is a strike. The busy answer is not
signed, so a node could send it to every challenge it wants to avoid, and
an answer that cost nothing must not end an audit without a strike (see
[Open limitations](#open-limitations), item 25).

## Consequences

- **Block.** A blocked peer gets no requests from this node. The block is
  local and lasts 7 days. Its requester is `receipts`. The operator can
  undo it. Blocks are not gossiped.
- **Block reason.** After a failure, the reason is
  `{"receipt_audit": {"receipt_id", "model", "failures", "evidence"}}`.
  After three strikes, it is
  `{"receipt_audit": {"strikes": [...], "strike_limit": 3, "strike_window_hours": 24}}`.
  Each strike has `at_ms`, `receipt_id` and `reason`.
- **Evidence.** The node keeps the evidence of a failure under
  `receipt-evidence/` in its identity state directory. The rules of the
  store are below.
- **Payments: not built.** A failed audit does not withhold an invoice.
  Payments are in the `pay-lightning` plugin, and no plugin seam carries
  the audit verdict to it. The first design let the client withhold the
  final invoice of a failed turn, and let the seller treat a challenged
  final invoice as pending. Neither exists in this repository.

A turn without an audit has no consequence. An upgrade on one side thus
does not block an honest node.

**The evidence store.** Evidence holds the token ids of the prompt and
activations of the turn, and a peer decides how large it is. So:

- A file has the name `<time>-<digest>.json`: 20 digits of milliseconds
  and 32 hex digits. The name holds no text from the peer, and the node
  does not overwrite a file.
- The JSON is compact. On Unix, the directory has mode 0700 and each file
  has mode 0600.
- The store keeps at most 64 files and 1 GiB, and at most 384 MiB per
  file. It removes the oldest files first.
- A node without an absolute state directory keeps no evidence. The block
  still occurs.
- A signed receipt that the audit refused before a challenge is kept as
  `{"refused_receipt": ...}`. `skippy recheck-receipt` does not read that
  form.

## Accepted risks

- **Low audit rate and free identities.** At 10%, a node that cheats on
  every turn serves about 10 turns to a client before that client catches
  it. Then it can join again with a new key. Cheating can still pay.
- **Selective cheating.** A node can cheat only toward clients that do not
  audit, or only on some turns. The protection of each client is its own.
- **Loose attention.** KV cache quantization and attention approximations
  within the tolerances pass.
- **Partial-row cheats.** A node that degrades a fraction `q` of the rows
  of a matrix is caught by the check of that matrix with probability about
  `1 - (1 - q)^16`.
- **Hugging Face availability.** If the client cannot reach Hugging Face,
  the turn gets no audit.
- **Recurrent state from an earlier turn.** A turn that continues a
  sequence starts from the state that the earlier turn left. The receipt
  binds that state only by its digest. The region check compares each
  ubatch with the one before it, not with the earlier turn.

## Open limitations

This is the single list for all receipts documents.

1. **Cached prefix.** An audit checks two positions of the cached prefix,
   so most of a long prefix is not checked in one audit. A turn whose
   prefix the node cannot replay gets no receipt (item 8). The chain of
   parents is implemented (`turn::verify_chain`), but no shipped client
   keeps a chain. The Mesh auditor uses a new VRF key per request.
2. **Sampling draw.** The draw is replayed within the draw band, 0.01% of
   the logit root mean square, so a node can steer a draw whose uniform
   is within that band of a boundary. Every honest draw measured was
   exact: about 130 on CPU, 120 on CUDA and 160 on Vulkan. A turn with
   penalties, a grammar, mirostat, XTC, DRY, typical or top-n-sigma
   sampling gets no token judgement, and neither does a model that
   soft-caps its logits. A turn with a dynamic temperature gets no replay
   of the draw. A turn with a reasoning budget is judged, but the client takes
   the tags of the budget from the statement of the node and does not
   derive them from the chat template. So a node can state an end tag of
   its own, of at most 256 bytes, and the tokens of that tag pass as
   forced tokens. The report shows the stated tags.
3. **Attention.** Attention is a loose check: a tolerance of 2e-2 or more
   on the combination, and exact on two drawn blocks. The tolerance
   increases with the square root of the context, because the CUDA prefill
   kernel accumulates in f16. So the combination check is looser at long
   contexts: 6.3e-2 at 4,096 positions and 1.8e-1 at 32,768. Backends other
   than CUDA are not measured at long contexts. The first block is drawn
   by the weight that the partials claim. A node that understates the
   weight of a block it changed, within the bounds of a block sum, is found
   only by the uniform draw: 1 in the number of blocks for each pick,
   where two uniform draws gave 2 in that number. Evidence recorded before
   2026-11-11 drew both blocks uniformly. A recheck of such evidence draws
   other blocks, finds no signed rows for them, and gives no audit.
4. **Model identity.** The identity is the name with a pinned commit,
   and the commit must be in the history of the branch that the request
   names. No LFS hash is checked on range reads, so the client trusts
   Hugging Face and its CDN, over TLS, to serve the bytes of the commit.
   The client checks that a range answer has the bytes it asked for,
   by `Content-Range`, but not what those bytes are.
   - A person who can write to the branch can push a commit with other
     weights, for example a smaller model under the same name. A node can
     then serve and receipt that commit, and the audit passes, because
     the branch is the identity that the request named. A request that
     pins a commit is not open to this.
   - A person who can write to the branch can rewrite its history. A
     commit that is no longer in the history then counts as off the
     branch, and a node that still serves it gets a strike. An auditor
     that accepted the commit before the rewrite keeps it until the
     process ends.
   - A person who can write to the branch cannot make a commit that is
     already in its history serve other bytes through the auditor,
     unless Hugging Face serves them.
   - A person who cannot write to the branch, the author of a pull
     request included, cannot make a Mesh auditor accept other bytes
     when the request names a branch or no revision.
   - A commit older than the newest 4,000 of the branch gives no audit.
   - `skippy recheck-receipt` without `--claimed-model` does not check
     the branch. Its report says so.
5. **Key delegation.** A delegation of the VRF key from the node key is not
   built. In Mesh, the receipts VRF key is a new key per request.
6. **Splits with remote stages.** A split that a node coordinates with a
   stage on another node does not satisfy the `receipts` tier.
7. **Backends and models.** Only CPU and CUDA stages are audited. Metal
   is not audited. On 2026-11-11, an honest Metal node with Q4_K_M
   weights failed 12 of 12 audits on a build that let it record, and
   Q8_0 weights passed with a margin of 1.19 times. See
   [the measurement](RECEIPTS_MEASUREMENTS.md#metal-2026-11-11). The CPU
   runtime for macOS uses the Accelerate BLAS, which has no profile, so
   no stage on macOS issues receipts. ROCm has no profile. Vulkan is not audited either. On 2026-11-10, an honest
   Vulkan node with Q4_K_M weights on an integrated GPU passed 2 of 16
   audits, because the integer-dot kernels round K-quant weights in a way
   that no profile models. Q8_0 weights passed there with 1% of margin.
   See [the measurement](RECEIPTS_MEASUREMENTS.md#vulkan-on-the-final-build).
   A stage on one of these backends issues no receipts, and a Mesh
   runtime on one does not advertise the `receipts` flag. Vulkan needs a
   profile for the integer-dot kernels on K-quants, and more margin for
   Q8_0 on integrated GPUs, before it can come back. DSA models have no
   opening design. Draft-model speculation is not tested.
8. **Turns without a receipt.** A stage with a quantized KV cache (q8_0
   or q4_0) gives no receipt. An honest stage with a q8_0 cache failed the
   loose attention check on 65 of 290 turns, by up to 5.5 times the limit
   at prompt positions, and a looser limit would check nothing. A hybrid
   stage gives no receipt for a turn that continues a cached prefix, so a
   Mesh node does not advertise receipts for a model whose stage 0 keeps
   recurrent state, and warns why. A stage whose weights have a type
   without a rounding profile on its backend, such as f16 on CUDA, gives
   no receipt. A turn with image tokens gets no receipt. A turn gets no receipt when the node
   cannot replay its prefix: after a shift, a state load or a compaction,
   when an unrecorded ubatch wrote a cell of the prefix, when the prefix
   holds cells of two requests, and on a stage after the first when the
   ubatch of a prefix cell is past the 1 GiB input budget. A turn also gets
   no receipt when it is past the 1 GiB of turn records of the serving
   node (see [Limits of a serving node](#node-commitment-and-replay)). In
   Mesh a turn without a receipt is a strike, also for an honest node.
9. **Stage authentication.** The stage port does not authenticate its
   peer. That is out of scope for receipts, and an operator keeps the port
   to the stages of the pipeline. A party that reaches the port of a later
   stage can do these things:
   - It can commit a turn of a live session under a receipt id of its
     choice, if it knows the session id. The real commit then fails, and
     the turn gets no receipt.
   - A later stage cannot compute the turn root, because the root covers
     the tokens, which only stage 0 has. So it holds a bound turn to the
     first root for which it saw a correct proof under the client key.
     The client key's holder can bind a false root there first, with a
     proof over that root. It then picks its own challenge of that stage,
     and the honest challenge from stage 0 fails with "another turn
     root". In Mesh, that is a strike against an honest node. A party
     without the client key cannot bind a false root.
   - With a captured proof, it can send a challenge of other picks under
     the real root first. A later stage admits one challenge per bound
     turn, so the real challenge then fails.
   - It can hold receipts connections, but a new connection closes the one
     that waited longest for a request.
   A fix needs an authenticated stage connection, or a message that binds
   the root and that only stage 0 can send.
10. **Runtime lock.** The creation of the scratch context runs with the
    runtime lock held. The correction needs a native call that prepares
    the scratch context.
11. **Evaluator errors.** The client first evaluates the picked layer
    from the committed values alone. An error there that names an opened
    value fails the node; any other error is "could not run here". When
    that evaluation succeeds, the client checks the opening against its
    plan of the layer, and then any error of the full evaluation fails the
    node. So a transient fault of the client's own evaluator between the
    two runs is blamed on the node. The client knows the refused values
    by the text of the native error messages. A new message in the native
    evaluator for a refused value is "could not run here" until the list
    in `skippy-receipts` names it.
12. **Glue scale.** The glue tolerance is relative to the largest value of
    the reference column. That reference reads opened values that the row
    check covers only at the drawn rows. A node that inflates one of them
    also changes the committed residual, which the glue check compares, so
    no attack on this scale is known. It is not bounded by a value that
    the node cannot set, as the row check now is.
13. **Signed fields that no client checks.** `turn::verify_cannot_open`
    checks a signed "cannot open" answer, but no client calls it: a Mesh
    auditor counts any "cannot open" as an unanswered challenge, a strike,
    signed or not. The `ubatches` field of
    a stage is signed and bounds the size of a reply, but no client
    compares it with the opened ubatches. A recurrent state committed
    together with a cached prefix fails the audit.
14. **Accepted risks.** The low audit rate with free identities, selective
    cheating, partial-row cheats and Hugging Face availability stay as
    [accepted risks](#accepted-risks).
15. **Decode overhead depends on the processor.** On 2026-11-10, decode
    with receipts on is about 1.5% slower than with receipts off, on one
    RTX 5090, which is inside the noise of the arm without receipts. The recorder hashes each decode ubatch on short-lived
    threads, which keep the cores of the stage out of deep idle. Hashing
    on one thread measured 7% slower in decode on the same machine. Other
    hardware can behave differently. See
    [RECEIPTS_MEASUREMENTS.md](RECEIPTS_MEASUREMENTS.md#serving-cost-on-the-final-build).
16. **Model names that Mesh cannot audit.** A Mesh node audits a turn only
    when it can tie the model name of the request to a Hugging Face file.
    A client can type a request as `repo:selector`. If the node rewrote
    that name to the bare runtime name of a peer, the turn is usually not
    auditable. Through a second node, a two-node test on 2026-11-10 routed
    only the id that the peer advertises, `repo@commit:selector`. A model
    that a node serves from a local path (`--gguf`) has the name
    `local-gguf/sha256-...`. It gets no audit and does not satisfy the
    `receipts` tier. This is also true for a GGUF of the Hugging Face
    cache that a node serves by its path.
17. **Offline nodes log nothing per turn.** With `[receipts] offline`, a
    node does not watch turns. So its log has no line for a turn.
18. **The two-node test uses a toy peer.** An automated test joins two
    nodes over real QUIC and sends the challenge over the mesh tunnel
    (`transport_tests/receipt_audits.rs`). The serving runtime there is a
    toy backend that serves and opens the toy turn, and the auditor opens
    the toy model, not a GGUF from Hugging Face. So that test does not
    cover Skippy issuing and opening real receipts, or reads from Hugging
    Face. A manual test with two real nodes passed on 2026-11-10.
19. **A CPU stage can open too slowly.** A stage on the CPU replays a long
    prefill slowly. On 2026-11-09, a two-stage Qwen3-4B split on a desktop
    CPU opened a fresh 1,500-token turn in 7 to 11 s. A Mesh client waits
    15 s for the opening from the moment it sends the challenge, then
    counts a strike. Short turns
    opened in about 1 s. A turn from a cached prefix also replays up to two
    ubatches of the prefix: on 2026-11-10, with a prefix of 1,900
    positions, it opened in 5 to 7.5 s on the same CPU. A longer prefix on
    a CPU stage is not measured.
20. **A signed "cannot open" is a strike.** A stage that can no longer
    open a turn, for example because a full cache wrote its pinned cells,
    answers the challenge with a signed "cannot open". The auditor counts
    that as no usable answer, so as a strike, for an honest node too. If
    it did not, a node that cheats would answer "cannot open" to each
    drawn turn on which it cheated, and never be caught. The strike limit
    bounds what an honest node loses to it.
21. **Two strikes a day cost nothing.** A peer is blocked at the third
    strike in 24 hours. So a peer can take two strikes a day, for example
    two cheated turns that were drawn and not answered, with no
    consequence. A lower limit blocks honest peers that lose an answer.
22. **`x-mesh-trust` on tunnel ingress.** A request that a peer sends to a
    node over the mesh tunnel runs where it lands, under no policy. The
    node ignores `x-mesh-trust` on it. A client that reaches a node only
    through the tunnel ingress therefore cannot require the `receipts`
    tier there, and the node does not refuse such a request.
23. **Unauditable receipts demote and do not block.** A peer that sends a
    receipt that this node cannot audit loses the `receipts` tier on this
    node for 24 hours, and is struck only if this node audited it for the
    same model in that time. A peer that is unauditable from its first
    turn is never struck for it. This is a
    [decision that the owner can overrule](#unauditable-receipts).
24. **The native loader runs in the auditing process.** The auditor
    checks in Rust the metadata values on which the llama.cpp loader and
    tokenizer stop the process (see [Model identity](#model-identity)).
    Other paths are not covered: the asserts of each of the about 180
    graph builders for each architecture, the ggml kernels on hostile
    weight values, the debug `assert` calls, the stage logic of Skippy,
    and the memory that a large model needs. A GGUF of a repository that
    a node controls can thus still stop or exhaust an auditing process.
    The proposed correction is to plan the model and open the stage
    (`StageModel::open`) in a child process that the auditor can lose. A
    child that stops gives no audit. This is not built.
25. **A stage busy for the whole window is a strike.** The Mesh auditor
    asks a busy stage again until the challenge window ends, and counts a
    stage that is still busy then as a strike. A stage opens one challenge
    at a time and lets 8 wait. An honest node is struck only when
    challenges keep its queue full for the 10 s after a receipt, which
    takes a steady stream of bound turns from several client keys. A
    signed busy answer, which could count as no audit up to a limit,
    needs a change of the wire format.
26. **Per-key limits and Mesh keys.** The Mesh auditor uses a new client
    key per request. So the pin share and the queue share of one key bound
    a single turn in Mesh, not a client. A client that sends many bound
    turns is held only by the shares of all keys together: past three
    quarters of a KV stream, the oldest pins give way. A turn that lost
    its pin answers "cannot open" when a later request wrote its cells
    before the challenge came, which in Mesh is a strike.
27. **Unrecorded ubatches beside receipted turns.** The native side does
    not report which sequences or cells a ubatch that it cannot record
    wrote, for example a ubatch of image embeddings. A turn that ran beside
    it gets a receipt only from its start. A pinned turn whose cells the
    ubatch wrote answers "cannot open", which in Mesh is a strike. The
    native side would need to report the cells of such a ubatch.
28. **A full KV cache writes pinned cells.** A pin only steers where new
    ubatches go. When the cache is full, the native side writes pinned
    cells, and the stage finds that only when it opens the turn. The turn
    then answers "cannot open".
29. **A replay needs memory that no budget reserves.** A stage makes the
    scratch context of a replay when the first challenge comes, outside
    every memory budget. On a device that other processes fill, the
    allocation can fail. The stage then refuses the challenge with a
    plain error, which a client counts as no answer, and in Mesh that is
    a strike. On 2026-11-11, one of 40 parallel audits on a full RTX 5090
    ended this way. The stage would have to reserve the scratch context
    when it starts, or answer busy so that the client asks again.
30. **A split stage with a full KV cache can lose a receipt.** When two
    cached prompts fill the cache of a split, the eviction of the prefix
    cache can free fewer cells than it counts, and a decode fails. That is
    a defect of the prefix cache, with receipts on or off. The KV
    compaction that follows makes the cached prefix of the next turn
    unknown, so that turn gets no receipt, which in Mesh is a strike. On
    2026-11-11 this cost 2 of 22 turns on a split with a context of 4,096
    tokens and two cached prompts of about 1,900 tokens. A stage that
    serves a model alone was not affected.
31. **Stated reasoning tags are not checked against the template.** The
    client replays the reasoning-budget sampler from the tags that the
    node states, and does not derive them from the chat template of the
    claimed model. A node can thus state an end tag of its own, of at
    most 256 bytes, and those tokens pass as forced tokens once per
    reasoning block. The tags are in the signed statement and in the
    report. Because the runtime now enforces budgets for chats without a
    grammar, this applies to every turn whose request does not ask for
    unlimited reasoning. The correction is to check the stated tags
    against the chat template in the claimed GGUF.
32. **A split is advertised by its stage 0.** Each stage decides at
    start whether it issues receipts, from its own backend, KV cache and
    weights. A Mesh node advertises receipts for a split from what stage 0
    decided. A later stage that issues none, for example the last stage
    of a model whose output head is f16 on CUDA, or a later stage with
    recurrent state behind a stage 0 without it, leaves every turn of
    the split without a receipt, which in Mesh is a strike. The
    correction is for stage 0 to ask each later stage at load whether it
    records receipts.

## Rollout status

| Step | Status | What is left |
|---|---|---|
| Node: leaf hashing, turn trees, signed receipts, KV pins and replay in Skippy, off by default | Done | The chain of turns has no client |
| Receipt in the response body and in the last stream chunk | Done | None |
| Client: the audit in `skippy audit-receipt` and in the Mesh host runtime | Done | The prefix check is a sample of two positions, a draw is replayed within the draw band, and the tags of a reasoning budget are the statement of the node |
| Evidence and `skippy recheck-receipt` | Done | None |
| Local peer block after a provable failure | Done | None |
| Strikes for events that only the auditing node saw, and a block after three strikes in 24 hours | Done | Strikes are in memory only. The automated test over a real mesh tunnel uses a toy peer and the toy model |
| Trust tier: a request policy that requires `receipts` | Done | Splits with remote stages do not qualify |
| Payments: act on the audit verdict | Open | A plugin seam for the verdict, and the pending state in the payment plugin |
| Phones: a verifier in the embedded client | Open | The embedded client has no peer blocks and no payments |

The detail of each milestone is in [RECEIPTS_PLAN.md](RECEIPTS_PLAN.md).
