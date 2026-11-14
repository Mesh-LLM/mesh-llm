# Private routing

Run your nodes in a shared mesh, such as the public one, and still send one
request only to nodes you choose.

## Who can answer a private request

A private request is answered only by the node the client talks to (the
*entry node*) and the nodes that node lists as its **private nodes**. With no
list, a private request never leaves the entry node.

A node is listed by its **node id**: its QUIC endpoint id, which every
connection to it proves. Nothing a peer merely says counts: model names,
node labels, hostnames and gossip are never used for this decision. No owner
key or certificate is involved.

The list is the entry node's own. It is not transitive: if your laptop lists
your studio and your studio lists a friend's node, your laptop's private
requests do not reach the friend's node.

## Set up your nodes

On each node you send private requests to, list your other nodes:

```bash
mesh-llm private add studio      # a connected node's label or hostname
mesh-llm private add mini
mesh-llm private add 8a88e3dd…   # or its node id (64 hex characters)
mesh-llm private list            # the list, and which nodes are connected
mesh-llm private remove mini
```

The list is `[private] nodes` in `~/.mesh-llm/config.toml` (or the file
`--config` / `MESH_LLM_CONFIG` names), so you can also edit it by hand:

```toml
[private]
nodes = [
  "8a88e3dd7409f195fd52db2d3cba5d72ca6709bf1d94121bf3748801b40f6f5c",
  "ed4928c628d1c2c6eae90338905995612959273a5c63f93636c14614ac8737d1",
]
```

Only node ids are saved. `add` looks a label or hostname up among the nodes
the running node (console port `3131`, or `--port`) is connected to and
writes that node's id; a name is never trusted when routing. `mesh-llm auth
status` on a node prints its id, and the console and `/api/status` show it too.

A running node notices the change within a few seconds; no restart is
needed. An entry that is not a node id makes the file invalid: `mesh-llm
config validate` says which, and a running node keeps private requests on
itself until the file is fixed.

Listing is one-way. If you also send private requests to your studio, list
your other nodes on it too.

Listed nodes also count as yours for admission: a request from one is
admitted as local, so a node that pauses remote inference while you use it
still serves your own nodes.

## Ask for private routing

Send the request to one of your own nodes, in any of these ways:

```bash
curl localhost:9337/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-32b@private","messages":[{"role":"user","content":"hi"}]}'
curl localhost:9337/v1/chat/completions -H 'Content-Type: application/json' -H 'x-mesh-scope: private' \
  -d '{"model":"qwen3-32b","messages":[{"role":"user","content":"hi"}]}'
curl localhost:9337/private/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-32b","messages":[{"role":"user","content":"hi"}]}'
```

`/private/v1` is for clients that only let you set a base URL.

`incognito` is an alias everywhere: `qwen3-32b@incognito`,
`x-mesh-scope: incognito`. The suffix works on the automatic directive too:
`mesh@private` or `auto@private` picks a model, or convenes a committee when
the MoA plugin is on, from the listed nodes only.

`GET /v1/models` with `x-mesh-scope: private` (or under `/private/v1`) lists
only what this node and its listed nodes serve.

## Guarantees

- Never falls back to public nodes. If neither the entry node nor a listed
  node serves the model, the request fails (404); it is not retried publicly.
- The entry node decides once; peers never re-route a forwarded request, so
  the entry node is the whole boundary. `x-mesh-target` and failover only
  choose among listed nodes, and the QUIC tunnel refuses any other peer as a
  backstop.
- The selector is consumed at the entry node: `@private` is stripped before
  model lookup and a client's `x-mesh-scope` is not forwarded. A private
  request forwarded to a listed node carries `x-mesh-scope: private` and
  `x-mesh-scope-nodes` naming the entry node's scope. That node keeps the
  request to itself and the entry node, within the nodes named; it does not
  use its own list for it. Both headers only ever narrow what the peer does.
- A split whose stages include an unlisted node is not a private target: its
  activations and KV would leave the scope.
- Plugins: while a plugin handles a private request (a virtual-model turn
  such as the MoA committee, an OpenAI exchange hook, or a request
  transform), every inference request that plugin makes back to the host is
  sent with `x-mesh-scope: private`, and with `x-mesh-scope-nodes` naming the
  request's scope when it has one yet. The host scopes such a call within its
  own scope, so a plugin on a listed node cannot reach nodes the entry node
  did not list. The nodes header can only narrow, never widen. This is per
  plugin, not per request: a public request the same plugin handles at the
  same moment is narrowed too. The plugin stays private until it replies,
  even if the client leaves first, and for five more minutes after a missed
  deadline, since it may still be working. A virtual-model turn or request
  transform that times out also restarts the plugin. An exchange hook stays
  private until it replies or its connection ends, and is restarted if it has
  done neither five minutes after its deadline.
- A plugin inference endpoint on the entry node can still serve a private
  request for a model no listed node serves. Such an endpoint may send
  the request to an outside service; install only plugins you trust.
  What a plugin does on its own, outside these calls, is up to the plugin:
  install only plugins you trust.
- Private routing is for OpenAI-compatible HTTP requests (`/v1`, `/private/v1`).
  Other ways into a node, such as the native SDK, have no private option yet.
- A trailing `@private` or `@incognito` is reserved. A model name that is
  only the suffix (`"@private"`) is rejected with 400. Peer-advertised names
  ending in either are not listed, and a peer naming a model `x@private`
  cannot attract a private request. A Hugging Face revision with either name
  must be referenced by commit.

Private routing controls where a request is computed. It does not hide the
request from the nodes you list, and it does not encrypt anything
beyond what the mesh's QUIC transport already does.

## Mixed versions

A node advertises private routing in its gossip (the `private-routing`
subprotocol). A private request goes only to listed nodes that advertise it: a node on an older release would ignore the private marker and
could serve the request from a split with a stranger's stage. Until you
upgrade them, your older nodes simply don't serve private requests.

On the node itself, requests marked private turn off peer consultation
(Mesh hooks for images, uncertainty and drift), which would pick peers
outside the scope, and run its exchange hooks in the request's scope.

## Not yet

- The startup (bootstrap) proxy refuses private requests (503) until the node
  is ready.
- A separate prompt-cache namespace, console and launcher (`--private`)
  support.
- Per-turn private scope for MoA nested requests.
- Peer consultation for private requests.
