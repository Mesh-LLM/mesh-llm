# Inference audit conformance (draft)

What an audit sender and an adjudicator must pin and publish so that anyone can
re-check a behavioural verdict on a node. Context and measurements:
[#1233](https://github.com/Mesh-LLM/mesh-llm/issues/1233),
[#2548](https://github.com/Mesh-LLM/mesh-llm/issues/2548), and the
[live check on the public mesh](https://github.com/Mesh-LLM/mesh-llm/issues/1233#issuecomment-6364726480).

## Roles

- **Sender:** sends an audit request to one peer, pinned with `x-mesh-target`,
  and checks that `x-mesh-served-by` names that peer. Senders should be many and
  not recognisable (hosts of the same model, at a low random rate).
- **Adjudicator:** scores the answer against pinned reference weights. Scoring is
  one prefill over the prompt and the answer; the adjudicator never talks to the
  audited node, so it can be a known, stable operator.
- **Node:** decides locally what to do with a score. Nothing in this document
  asks a node to block a peer.

## Reference identity

- A model is identified by its content, not its name. For a model pulled from
  Hugging Face this is the `local-gguf/sha256-…` id the host advertises: the
  sha256 of the canonical JSON of repository, resolved commit, and the path,
  size and sha256 of each file. Anyone can recompute it from Hugging Face's
  published digests without downloading the weights.
- The adjudicator loads exactly those files and publishes the id and the sha256
  of each file. An audit of a node advertising another id is void.

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
  operators agree.

## Audit corpus

- Versioned. A hash commitment to each version is published before it is used.
- Prompts stay private while the version is in use and are published when it is
  retired, so every record from that period becomes re-checkable.
- Records exchanged while a version is live carry a keyed prompt digest
  (HMAC-SHA-256 with a per-version key revealed at retirement), so a node cannot
  match its incoming requests against published digests.

## Records

- Sender side: the fields `openai.exchange.v1` already carries (served-by, model
  identity hash, weights digest, request and response digests, nonce and its
  source).
- Adjudicator side: the AAC judge record family (adjudication/v1,
  sample-manifest/v1, calibration-summary/v1). Open point: its `prompt_digest`
  allows only plain SHA-256 today, so the keyed digest above needs either an
  allowed `HMAC-SHA-256` algorithm with a key id or a sibling field.

## Re-checking a verdict

With the retired corpus, the reference files and the records, anyone can render
the same prompt, run the same prefill and recompute the score.
