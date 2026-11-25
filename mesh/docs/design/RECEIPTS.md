# Inference receipts in Mesh

Status: built and experimental. Receipts are off by default. A Mesh node
issues receipts through its Skippy runtime, audits its peers and routes by
the `receipts` trust tier.

A receipt lets a client check on its own CPU that a node ran the model that
it claims. The protocol, the threat model, what a pass proves, the
parameters and the open limitations are designed in the Skippy repository:

- [design/RECEIPTS.md](https://github.com/Mesh-LLM/skippy/blob/main/docs/design/RECEIPTS.md): the design
- [RECEIPTS.md](https://github.com/Mesh-LLM/skippy/blob/main/docs/RECEIPTS.md): the operator and user guide of
  `skippy serve` and `skippy audit-receipt`
- [design/RECEIPTS_GLUE.md](https://github.com/Mesh-LLM/skippy/blob/main/docs/design/RECEIPTS_GLUE.md): how a stage finds
  the receipt tensors, and how the client checks the glue
- [design/RECEIPTS_MEASUREMENTS.md](https://github.com/Mesh-LLM/skippy/blob/main/docs/design/RECEIPTS_MEASUREMENTS.md):
  every measurement, with its date and build

This document covers what a Mesh node adds. A Mesh node is one client of the
protocol: which turns it watches and draws, what it holds a peer to, how it
times a challenge, and what a verdict costs a peer are its own policy.

Related:

- [../RECEIPTS.md](../RECEIPTS.md): how to turn receipts on in a Mesh node,
  read its verdicts and lift a block
- [TRUST_TIERS.md](TRUST_TIERS.md): the `receipts` tier
- [../plugins/openai-exchange-lifecycle.md](../plugins/openai-exchange-lifecycle.md#receipt-audit-evidence):
  the audit verdict that a plugin sees
- [`Mesh-LLM/ln-payments-plugin`](https://github.com/Mesh-LLM/ln-payments-plugin):
  the `pay-lightning` payment plugin
- [../NODE_REP.md](../NODE_REP.md)

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
does not draw is no audit. A plugin that waits for the verdict of the turn
is told that the turn was not drawn (see [Consequences](#consequences)).

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

Worker threads audit the drawn turns off the request path, two by default
(see [The workers](#the-workers)). A drawn turn can add these strikes:

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
- The request pinned no commit, and the commit that the receipt names is
  on the branch but does not hold the file. Hugging Face answers 404 for
  it.

A drawn turn is no audit in these cases:

- The node cannot tie the model name of the request to a Hugging Face
  file.
- The receipt names no Hugging Face file at a commit: a bare name, a local
  path or a branch.
- Every worker is busy, the workers are off, or the node shuts down. A
  busy worker is counted against the peer for payments when audits of its
  own turns hold workers (see [Consequences](#consequences)).
- The challenge was not ready in the challenge window.
- The challenge was sent more than 8 s after the receipt, and the peer
  answered that it no longer holds the turn (404, 409 or 410).
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
| Challenge window | 10 minutes after the receipt. The node checks the window when it sends the challenge, after the claimed model is open. A challenge that is too late is not sent |
| Prompt window | 8 s after the receipt. A peer keeps a turn for sure for 12 s from its commit, which leaves 4 s for the response and the challenge to travel, and then while it has room. When the node sends the challenge later and the peer answers that the turn is gone, that is no audit. Sent in time, the same answer is a strike |
| Busy peer | Asked again for 10 s from the first send of the challenge |
| Wait for the opening | 15 s from each send of the challenge, also when the node sends it again to a busy peer. The time that a drawn turn waited in the queue of the node does not shorten it |
| Request for rows | Not sent later than 28 s after the challenge |
| Wait for the rows | 10 s |

The first audit of a model opens the claimed model and builds the head
key. That time does not count against the peer: a late step is no audit.

A peer can hold one audit for at most 48 s: the 10 s in which it answers
"busy" to the challenge, the 28 s in which it answers "busy" to the request
for rows, and the wait for the rows. The time to open the claimed model and to check the
turn is the node's own, and has no bound. Each read from Hugging Face
has a timeout of 120 s, and a turn that waited in the queue past the
window is not opened. When one audit holds a worker for more than 48 s,
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

### The workers

The node owns its worker threads: two by default, and
`[receipts] audit_threads` sets from 1 to 16. A worker starts with its
first drawn turn and is joined at shutdown. Each audits one turn at a time
and has a queue of eight. A drawn turn goes to the worker that holds the
fewest turns, and to the first of those, so a node that audits one turn at
a time uses one thread. A drawn turn that finds every queue full is no
audit, and the log gives the number of such turns since the node started.
A turn may wait in a queue for the whole challenge window; the last of a
full queue still starts inside it when each audit before it takes the
48 s a peer can make it take. A waiting turn holds its response, up to
about 20 MiB, so a thread and its queue hold up to 180 MiB.

A panic in one audit does not end its worker: the turn is no audit, and
the log says so. If a thread ends, the node starts it again, at most three
times. Then that worker is off until the node starts again, and the log
says so. Each worker keeps two claimed models open and releases a model
after 10 minutes without an audit. A loaded model stays on the thread that
loaded it, so more threads hold more model data.

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
keeps what the request needs: 10 s from the first send of a challenge, and
28 s after the challenge for the attention rows. A stage that is still
busy then gave no answer, and that is a strike. The busy answer is not
signed, so a node could send it to every challenge it wants to avoid, and
an answer that cost nothing must not end an audit without a strike (see
[Open limitations](https://github.com/Mesh-LLM/skippy/blob/main/docs/design/RECEIPTS.md#open-limitations), item 25).

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
- **Payments.** The host gives the verdict of a remote turn that requires
  receipts to its
  [exchange lifecycle plugins](../plugins/openai-exchange-lifecycle.md#receipt-audit-evidence).
  A payment plugin decides what each status means for the final usage
  payment. The rule itself is in the `pay-lightning` plugin, not in this
  repository. Set `require = true` in `[receipts]`, or set `audit = true`
  and send `x-mesh-trust: receipts` on the purchase request.

| Status | Meaning | Intended payment |
|---|---|---|
| `passed` | The VRF drew the turn, and its audit passed | Pay |
| `not_drawn` | The receipt is the peer's, bound to the request and names the model asked for. The VRF did not draw the turn | Pay |
| `not_audited` | The same receipt, and the VRF drew the turn. The node did not complete the audit, for a reason that is not in the receipt: its workers were busy, Hugging Face was out of reach, or the peer no longer held a turn that the node challenged late | Withhold |
| `unverified` | The receipt supports no audit of what was asked for: it is missing or unusable, names another file, or states something the node cannot audit. Also a turn whose response did not end whole, and a verdict that did not arrive in time | Withhold |
| `failed` | The audit failed, or the peer owed a receipt or an answer to a challenge and gave none | Withhold |

**Why `not_audited` is not paid.** If a turn that was drawn and not
audited were paid, nothing would depend on the audit: a peer would gain by
keeping its audits from running, and a buyer whose audits never run would
pay for work that nobody checked. So the buyer pays for a drawn turn only
when its audit passed. The cost falls on an honest peer when the fault is
the buyer's: a final invoice that stays unpaid is a debt at the seller, and
the `pay-lightning` seller refuses a buyer that has three. A buyer whose
audits do not run thus loses each seller after three drawn turns, until the
operator of the seller forgives the debt. That is the pressure on a buyer to
keep its audits running: enough `audit_threads`, and a path to Hugging
Face. The status is told apart from `unverified` so that the operator of
the buyer can see that the audit did not run, where `unverified` says that
the receipt was not fit for one.

A peer that makes a drawn turn end without a verdict is not paid for that
turn. It can still be paid for the turns that the VRF does not draw. Three
rules bound that:

- **Sampling is earned.** Until a peer is proven for a model, the VRF draws
  every turn of it that a plugin waits on. A peer is proven when as many of
  its drawn turns in a row passed as one draw stands for: 9 at 1 in 10,
  and at most 16. A drawn turn that does not pass, for any reason, makes
  the peer unproven again. An honest peer is audited more often for a
  while, and its turns pass. Standing is in memory, per peer and model, so
  every peer is unproven after a restart.
- **Unsettled turns are counted.** A drawn turn is unsettled when it ended
  without a verdict for a reason that a peer can cause: the turn was gone
  at a late challenge, the commit was past the budget, or every worker was
  busy while audits of this peer's own turns held some of them. Three
  unsettled turns of a peer in 24 hours take the peer out of the `receipts`
  tier on this node for 24 hours, and for 7 days when that happens again
  within a week. Only turns that a plugin waits on are counted. A cause on
  the side of the node alone is not: Hugging Face out of reach, a slow
  first open of a model, or workers that are busy with other peers.
- **A commit without the file is a strike.** When the request pins no
  commit, the peer chooses the one that its receipt names. A commit of the
  right branch that does not hold the file opens nothing, so the audit of
  a drawn turn that names one is a strike, not a turn without an audit.

These rules bound the gain. They do not remove it. A proven peer that then
serves another model is paid for its undrawn turns until the VRF draws one,
which is 9 turns on average at 1 in 10. After three drawn turns in a day
that ended unsettled the peer is out of the tier, so it is paid for about
27 turns a day that it did not serve as claimed, per identity, buyer and
model. The number grows with `audit_one_in`. An honest peer can be counted
too: when the node challenges late because its own queue is long, and the
peer needed the room that the turn took. A buyer that wants no turn paid
without a pass sets `audit_one_in = 1`.

Outside of payments, a turn without an audit has no consequence for the
peer. An upgrade on one side thus does not block an honest node.

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
