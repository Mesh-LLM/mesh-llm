# Inference audit conformance (draft)

What an audit sender and an adjudicator must pin and publish so that anyone can
re-check a behavioural verdict on a node. Context and measurements:
[#1233](https://github.com/Mesh-LLM/mesh-llm/issues/1233),
[#2548](https://github.com/Mesh-LLM/mesh-llm/issues/2548), and the
[live check on the public mesh](https://github.com/Mesh-LLM/mesh-llm/issues/1233#issuecomment-6364726480).

## Roles

- **Sender:** sends an audit request to one peer, pinned with `x-mesh-target`,
  and keeps the answer as audit evidence only if the response succeeded (HTTP
  200 with a completion in the body) and `x-mesh-served-by` names that peer.
  Senders should be many and not recognisable (hosts of the same model, at a
  low random rate).
- **Adjudicator:** scores the answer against pinned reference weights. Scoring is
  one prefill over the prompt and the answer; the adjudicator never talks to the
  audited node, so it can be a known, stable operator.
- **Node:** decides locally what to do with a score. Nothing in this document
  asks a node to block a peer.

## Reference identity

- A model is identified by its content, not its name. A GGUF model pulled from
  Hugging Face is announced to peers under the internal name
  `local-gguf/sha256-<digest>` (in `serving_models`); its public name, derived
  from the repository and selector, is linked to it by `source_ref` in
  `/api/models`.
- The digest is the lower-case hex SHA-256 of this JSON, serialised compactly
  with the keys in this order: `schema_version` (1), `source`
  (`"hugging-face-gguf"`), `repository`, `revision` (the resolved 40-character
  commit, lower case) and `files`, one `{path, bytes, sha256}` entry per GGUF
  part in part order (`huggingface_identity_sha256` in `skippy-api`). Anyone
  can recompute it from Hugging Face's published digests without downloading
  the weights.
- The adjudicator loads exactly those files and publishes the id and the sha256
  of each file. An audit of a node advertising another id is inconclusive.

## Scoring setup

The adjudicator publishes, per model:

- the runtime and its version (llama.cpp commit or Skippy ABI), the backend and
  device class;
- how the prompt is rendered: chat template source and whether reasoning is on,
  since a Qwen3-style template adds an empty thinking block when it is off;
- the check that the rendered prompt has the token count the node reported in
  `usage.prompt_tokens`. A mismatch makes the audit inconclusive, never
  contradicted.

## Audit requests

- Non-streamed, greedy (`temperature: 0`), bounded `max_tokens`.
- `logprobs: true` with `top_logprobs: 20` when the node serves them.
- Prompts drawn from the current corpus version (see below), never a user's
  prompt or answer.

## Statistics and thresholds

- Text only (greedy): the share of answer tokens that are the reference's top
  choice. With logprobs: the divergence between the node's top-20 and the
  reference's.
- Thresholds are per model id and per request shape, calibrated on honest runs
  on at least two backends (for example GPU and CPU), with answers held out from
  calibration.
- Each threshold is published with its calibration counts as k of n (false
  alarms out of honest answers), never as a rate, and with the number of
  answers a verdict pools. For scale: on the 22 Oct live check, honest greedy
  agreement was 98.4 % and 98.8 % on two models, another model served under the
  name scored 75 % to 79 %, and 1,152 honest Qwen answers in #2548 gave 0 false
  alarms (links above).

## Verdicts

- `corroborated`, `inconclusive` or `contradicted`, as in the AAC adjudication
  record.
- Inconclusive whenever the audit cannot be scored cleanly: identity or token
  count mismatch, `x-mesh-served-by` naming another peer, an answer too short
  for its threshold.
- A contradiction counts only when adjudicators run by at least two different
  operators agree. An operator is an owner identity
  ([NODE_OWNER_IDENTITY.md](../design/NODE_OWNER_IDENTITY.md)), and each
  adjudication record is signed with that owner key. Two records score the same
  audit when they cite the same sender record (its request digest, response
  digest and nonce together) and the same corpus version; they agree when they
  give the same verdict and, for `contradicted`, name the same
  `contradicted_party`. Operators differ when their owner identities differ.

## Audit corpus

- Versioned. A SHA-256 commitment to each version is published before it is
  used; it covers the version number, the prompts and the version's digest key
  (below). Its exact byte encoding is settled with the record mapping (open
  points below).
- Prompts stay private while the version is in use. At retirement the prompts
  and the key are published next to the commitment, so anyone can check them
  against it, and every record from that period becomes re-checkable.
- Records exchanged while a version is live carry a keyed prompt digest
  (HMAC-SHA-256 with a per-version key revealed at retirement), so a node cannot
  match its incoming requests against published digests. Provisional until the
  record mapping below settles how that digest is carried.

## Records

- Sender side: the fields `openai.exchange.v1` already carries (served-by, model
  identity hash, weights digest, request and response digests, nonce and its
  source).
- Adjudicator side: the AAC judge record family (adjudication/v1,
  sample-manifest/v1, calibration-summary/v1), as of
  [agent-action-capsule@062e89bd145d](https://github.com/action-state-group/agent-action-capsule/tree/062e89bd145d/schemas/judge).
- Both sides name the corpus version the prompt came from, which also tells a
  re-checker which retired key applies.
- Open points for the record mapping:
  - AAC's `prompt_digest` allows only plain SHA-256 today, so the keyed digest
    above needs either an allowed `HMAC-SHA-256` algorithm with a key id or a
    sibling field.
  - adjudication/v1 has no citation slot for the calibration-summary/v1 and
    sample-manifest/v1 records its threshold comes from, so the mapping needs a
    digest citation or a wrapper.
  - The corpus commitment needs a fixed encoding.

## Re-checking a verdict

With the retired corpus and its key, the reference files and the records,
anyone can match each record to its prompt by recomputing the keyed digest,
render the same prompt, run the same prefill and recompute the score.
