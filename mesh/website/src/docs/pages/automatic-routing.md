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

By default, `mesh` picks **one model** that can handle the request (tools,
images, context length) and answers from it, streaming or not. If no online
model advertises tool support, a tool request still goes to one of the
available models, which may not handle tools. The machine is
chosen from every machine serving that model: measured speed weighs the
choice, along with current load and session affinity, so faster machines
usually take more of the traffic.

A machine that is busy (`429`) or unavailable (`502`–`504`) is skipped before
the first byte is sent; a `500` is returned as is. If every machine for the
model is busy or unavailable, Mesh tries another compatible model, up to three
more. A timeout may move the request to another machine of the same model,
but never to another model.

If you send an image and **no** online model supports image input, the request
fails with `422` rather than quietly answering the text part of your question.

Once a streamed response has started, it cannot be retried elsewhere — if that
machine disappears mid-answer, the response is cut short.

## Mixture-of-Agents plugin (opt-in)

Mesh ships a built-in Mixture-of-Agents (MoA) plugin: several placements
(machines serving a model, possibly the same model) answer,
and one synthesises the result. It is **off by default**. Enable it in
`~/.mesh-llm/config.toml`:

```toml
[[plugin]]
name = "mesh-moa"
```

With it enabled, non-streaming text chat sent to `mesh` is answered by the
committee (responses carry `x-moa-*` headers) when at least two placements are
eligible and at least one has a known size of 10B parameters or more. When
only one placement is eligible, or every eligible one is small (under 10B) or
of unknown size, the
plugin answers from one model and sends no `x-moa-*` headers. Streaming,
image/audio/file requests and named models still use a single model. See
`mesh/evals/moa-openrouter/RESULTS.md` for when a committee helps.

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

Historically `auto` and `mesh` behaved differently. That distinction leaked an implementation detail into the API — callers had to
know which mechanism they wanted, when what they actually wanted was a good
answer. There is now one name for that intent.

Migration is a string change:

```diff
- {"model": "auto",  "messages": [...]}
+ {"model": "mesh", "messages": [...]}
```
