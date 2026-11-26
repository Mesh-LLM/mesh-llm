---
title: Private Routing
---

# Private Routing

Run your nodes in a shared mesh, such as the [public mesh](/docs/pages/public-mesh/),
and still send one request only to nodes you choose. Add `@private` (or
`@incognito`) to the model name and the request is computed only on your own
node and the nodes it lists. It never falls back to anyone else's.

Private routing controls request destinations, not mesh membership. For a
**private mesh** with restricted membership, configure
[admission control](/docs/pages/private-meshes/#restrict-membership).

## Who can answer a private request

A private request is answered only by the node the client talks to (the
*entry node*) and the nodes that node lists as its **private nodes**. With no
list, a private request never leaves the entry node.

A node is listed by its **node id**: its QUIC endpoint id, which every
connection to it proves. Model names, node labels, hostnames and gossip are
never used for this decision.

The list belongs to the entry node and is not transitive: if your laptop lists
your studio and your studio lists a friend's node, your laptop's private
requests do not reach the friend's node.

## Set up your nodes

On each node you send private requests to, list your other nodes:

```sh
mesh-llm private add studio      # a connected node's label or hostname
mesh-llm private add mini
mesh-llm private add 8a88e3dd…   # or its node id (64 hex characters)
mesh-llm private list            # the list, and which nodes are connected
mesh-llm private remove mini
```

`add` looks a label or hostname up among the nodes the running node is
connected to (console port `3131`, or `--port`) and saves that node's id. Only
ids are saved. `mesh-llm auth status` on a node prints its id; the console and
`/api/status` show it too.

The list is `[private] nodes` in `~/.mesh-llm/config.toml`, so you can also
edit it by hand; see [Private nodes](/docs/pages/config-toml/#private-nodes).
A running node picks up the change within a few seconds, without a restart.
If an entry is not a valid node id, `mesh-llm config validate` says which, and
the running node keeps private requests on itself until the file is fixed.

Listing is one-way. If you also send private requests to your studio, list
your other nodes on the studio too.

Listed nodes also count as yours for admission: a request from one is admitted
as local, so a node that pauses remote inference while you use it still serves
your own nodes.

## Send a private request

Send the request to one of your own nodes, in any of these ways:

```sh
# Suffix on the model name
curl -s http://localhost:9337/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-32b@private","messages":[{"role":"user","content":"hi"}]}'

# Header
curl -s http://localhost:9337/v1/chat/completions \
  -H 'Content-Type: application/json' -H 'x-mesh-scope: private' \
  -d '{"model":"qwen3-32b","messages":[{"role":"user","content":"hi"}]}'

# Base URL, for clients that only let you set one
curl -s http://localhost:9337/private/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-32b","messages":[{"role":"user","content":"hi"}]}'
```

- `incognito` works everywhere `private` does: `qwen3-32b@incognito`,
  `x-mesh-scope: incognito`.
- [Automatic routing](/docs/pages/automatic-routing/) takes the suffix too:
  `mesh@private` picks a model, or convenes a committee when the MoA plugin is
  on, from your listed nodes only.
- `GET /v1/models` with `x-mesh-scope: private`, or under `/private/v1`, lists
  only what this node and its listed nodes serve.

## Guarantees

- **No public fallback.** If neither the entry node nor a listed node serves the
  model, the request fails with 404. It is not retried on other nodes.
- **The entry node decides.** Peers never re-route a forwarded private request,
  and the QUIC tunnel refuses any unlisted peer as a backstop. `x-mesh-target`
  and failover choose only among listed nodes.
- **No stranger's split stages.** A split whose stages include an unlisted node
  is not a private target, because its activations and KV would leave your
  nodes.
- **Plugins stay in scope.** While a plugin handles a private request (such as
  an MoA committee turn), its inference calls back to the host are kept within
  the same nodes.
- **The suffix is reserved.** A peer cannot attract private requests by naming a
  model `something@private`, and a model name that is only the suffix is
  rejected with 400.

## Limits

- Private routing controls **where** a request is computed. It does not hide
  the request from the nodes you list, and it adds no encryption beyond the
  mesh's QUIC transport.
- A plugin inference endpoint on the entry node can still serve a private
  request, and may send it to an outside service. Install only plugins you
  trust.
- Only OpenAI-compatible HTTP requests (`/v1`, `/private/v1`) can be private.
  The native SDK has no private option yet.
- Nodes on releases without private routing never receive private requests;
  upgrade them to include them.
- Until a node is ready, its startup proxy refuses private requests with 503.
- No console or launcher (`--private`) support yet.

The full design is in the
[private routing spec](https://github.com/Mesh-LLM/mesh-llm/blob/main/mesh/docs/specs/private-routing.md).
