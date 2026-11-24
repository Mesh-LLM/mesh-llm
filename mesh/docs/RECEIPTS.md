# Receipts on a Mesh node

A Mesh node serves inference receipts through its Skippy runtime and can
audit the turns that its peers serve. This guide shows how to turn both on,
how to read the verdicts of the node and how to lift a block.

How a receipt works, what a pass proves, the requirements on a stage, the
`skippy audit-receipt` command and the troubleshooting tables are in the
Skippy guide, [RECEIPTS.md](https://github.com/Mesh-LLM/skippy/blob/main/docs/RECEIPTS.md). The policy of a Mesh node, with
the reason for each rule, is in
[Audit policy in Mesh](design/RECEIPTS.md#audit-policy-in-mesh) and
[TRUST_TIERS.md](design/TRUST_TIERS.md#receipts).

## Configure the serving node

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
`x-mesh-trust: receipts` does not reach it. The same is true for a file
of a Hugging Face cache that you pass by its path: the receipt names the
Hugging Face file, but the node serves it as `local-gguf/sha256-...`,
and the auditor ties the name of the request. Use one of these instead:

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

## Keep the defaults of a serving node neutral

An auditing peer compares the sampler statement of a receipt with the
request, and the response with the sampled tokens. Defaults on the
serving node can thus give an honest node strikes.

- Do not set defaults that add a penalty or a logit bias, or that drop a
  sampler filter. The sampler statement then contradicts a request that
  set none of them.
- Do not set default stop strings. They cut the text, and the response
  check of the peer fails.

## Make sure that the serving node advertises receipts

A node advertises a model with receipts only when all of these are true:

- the loaded runtime issues receipts, because its recorder started. The
  recorder does not start on a stage with a quantized KV cache, or on a
  backend other than the CPU and CUDA.
- the backend of the runtime has rounding profiles: the CPU or CUDA.
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

## Configure the client node

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
[Require verifiable receipts](PROVIDER_SELECTION.md#require-verifiable-receipts).

An audit runs on a worker thread, off the request path. The node runs two
by default, and `[receipts] audit_threads` sets from 1 to 16. It opens the
challenge on the peer over the mesh, reads the claimed Hugging Face GGUF
by range, and checks the opening on the CPU.

## Name the model so that the node can audit it

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

When the name has a branch or no revision, the commit that the receipt
names must be in the history of that branch, or of `main`. The node reads
the commit list of the branch from Hugging Face before it fetches the
file, and keeps each commit that it found until it stops. A commit that is
not on the branch, for example the commit of a pull request, gives a
strike. A commit older than the newest 4,000 of the branch gives
`no audit`. A name with a commit needs no check. With `[receipts] offline`
the node makes no request to Hugging Face.

## Read the verdicts

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
| `every audit worker is busy` | Every worker audits a turn and has eight more in its queue. Set `[receipts] audit_threads` higher, or `audit_one_in` higher |
| `this node's audit workers are off` | The workers ended too many times, or their threads did not start. Start the node again |
| `this node is shutting down` | The node stops |
| `past the 600 s challenge window` | The turn waited more than 10 minutes for a worker or for the claimed model to open |
| `the node no longer held the turn` | The node sent the challenge more than 8 s after the receipt, and the peer had needed the room the turn took. A peer keeps a turn for sure for 12 s from its commit, and then while it has room |
| `the attention rows were not checked` | The node was ready to ask for the rows later than 28 s after the challenge |
| `this node could not open the claimed model` | Hugging Face is out of reach, or the file is over a limit |
| `checks could not run here` | A check was not able to run on this machine |
| `the response did not end whole`, `the peer answered with status`, `the peer's answer reported an error` | The response was not a whole turn |
| `no rounding profiles for backend`, `the node ran` | See [NO AUDIT causes](https://github.com/Mesh-LLM/skippy/blob/main/docs/RECEIPTS.md#no-audit-causes) |

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

**The workers.** If an audit worker ends unexpectedly, the node logs an
error and starts it again, at most three times. Then the log has `receipt
audits are off until this node restarts`.

## Read and lift a block

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
