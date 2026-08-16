---
title: Automatic Routing
---

# Automatic Routing

Send `model: "mesh"` and let the mesh decide how to serve the request. You do
not have to know what models are online, how big they are, or whether several
machines are available.

```sh
curl -s http://localhost:9337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mesh","messages":[{"role":"user","content":"Explain QUIC in two sentences."}]}'
```

`mesh` is the canonical automatic directive — `auto` still works as a
[deprecated alias](#auto-is-deprecated). Naming a model explicitly
(`"model": "unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL"`) always routes to that
model and is unaffected by anything on this page.

## What you get

Automatic routing serves a request in one of two ways, and it chooses for you.

**A committee.** Several models answer, see each other's drafts, improve them,
and one model synthesises the result. This is Mixture-of-Agents (MoA). It is
what `mesh` does whenever the mesh can field a committee and the request allows
it.

**A single model.** One model, selected for the request, answers directly.

Both are normal outcomes. A single model is not a failure or a downgrade — for
some requests it is the only correct answer, and for some meshes it is the best
available one.

## When you get a single model

Four cases, all decided from your request or the mesh's shape:

| Situation | Why |
|---|---|
| Your request includes an image, audio, or a file | A committee compares and merges answers as text, so it has no defined way to combine answers about an image. Your request is routed to a model whose runtime actually supports that input. |
| You asked for `"stream": true` | Committee members must finish before their answers can be compared, so a committee cannot emit real tokens as they are produced. A single model streams for real. |
| You sent no `model` field at all | You did not ask for automatic serving, so Mesh does not add committee latency and cost on your behalf. Send `"model": "mesh"` to opt in. |
| You posted to `/v1/completions` | Committee members are asked a chat question each, and a completions request carries a `prompt` rather than a conversation. `/v1/chat/completions` and `/v1/responses` both convene committees. |

If you send an image and **no** online model supports image input, the request
fails with `422` rather than quietly answering the text part of your question.

## What the mesh's shape changes

| Your mesh | What `mesh` does |
|---|---|
| One model | Serves that model. `mesh` works on a single node. |
| Several copies of one model on different machines | Committee. Repeated sampling of one model ensembles well. |
| A mix of larger and smaller models | Committee of the stronger models, unless dropping the smaller ones would leave no committee at all — in which case the mix is kept, because a mixed committee beats one strong model alone. |
| Only small models (under ~10B) | The strongest single model answers. Measured on this configuration, a committee of small models did not beat the best member alone, so Mesh does not spend the extra calls. |

The last row is a deliberate, measured choice rather than a limitation of the
idea. See `evals/moa-openrouter/RESULTS.md` in the repository for the data.

## Streaming and resilience are a trade

Worth knowing if you care about robustness:

- A **committee** collects whole answers before replying, so if a machine drops
  out mid-request the remaining members still produce an answer. It cannot send
  you a first token early.
- A **single model** streams tokens as they are generated. But once the first
  bytes of a streamed response have been sent, there is no way to retry
  elsewhere — if that machine disappears mid-answer, the response is cut short.

Because `"stream": true` selects a single model, asking to stream is also
asking for the less recoverable path. For long unattended work on a mesh whose
machines come and go, leaving streaming off is the more robust choice.

## Discovering what is available

`mesh` appears in `/v1/models` whenever the node can serve anything, so you can
send it without checking first.

```sh
curl -s http://localhost:9337/v1/models | jq '.data[].id'
```

The `mesh` entry advertises the combined capabilities of the mesh: if any online
model accepts images, `mesh` reports vision support, because an image request
will be routed to that model.

## Choosing a model yourself

Automatic routing deliberately does not accept "give me a big one" or "give me a
cheap one". Those are preferences only you can weigh, and `/v1/models` gives you
what you need to act on them — each entry carries `parameter_count_b`,
`parameter_size`, `quant`, `context_length`, `provider`, `replicas`, and
`active_requests`.

So a client that wants the largest model sorts the listing and sends that
model's name. That keeps preference-picking in your hands, where the intent is,
and keeps `mesh` focused on serving whatever you send as well as it can.

## `auto` is deprecated

`model: "auto"` still works and behaves identically to `model: "mesh"`. It is a
deprecated alias and will be removed in a future release; the daemon logs a
warning when a request uses it.

Historically `auto` picked one "good" model while `mesh` convened a committee.
That distinction leaked an implementation detail into the API — callers had to
know which mechanism they wanted, when what they actually wanted was a good
answer. There is now one name for that intent.

Migration is a string change:

```diff
- {"model": "auto",  "messages": [...]}
+ {"model": "mesh", "messages": [...]}
```
