# Private routing

Run your nodes in a shared mesh, such as the public one, and still send one
request only to nodes you trust.

## Who can answer a private request

A private request is answered only by the **trust circle** of the node the
client talks to (the *entry node*):

1. nodes signed by the entry node's own owner key, and
2. nodes signed by an owner key in the entry node's trust store
   (`mesh-llm auth trust add`, or `--trust-owner` on `serve`).

With an empty trust store, the circle is just your own nodes.

The circle is checked against each peer's **ownership certificate**: the
owner key's signature over the peer's QUIC endpoint id, which the peer
presents when it connects. A peer is in the circle only while its certificate
verifies, is bound to that peer's endpoint, has not expired and is not
revoked in the entry node's trust store. Nothing a peer merely says counts:
model names, node labels, hostnames and gossip are never used for this
decision.

The circle is the entry node's view. It is not transitive: if you trust
Alice and Alice trusts Bob, your private requests do not reach Bob.

## Set up your nodes

The owner key is the identity. Create it once, then give every node you own
that key. A node only needs the key to sign (and renew) its own certificate.

1. On your main machine, create the owner key and note its owner id:

   ```bash
   mesh-llm auth init            # writes ~/.mesh-llm/owner-keystore.json
   mesh-llm auth status          # prints "Owner ID: …"
   ```

   On a headless node, use `auth init --no-passphrase`, or keep the passphrase
   and set `MESH_LLM_OWNER_PASSPHRASE` for `serve`.

2. Copy `~/.mesh-llm/owner-keystore.json` to each of your other machines,
   over a channel you trust. It is the owner's secret key: anyone holding it
   can sign nodes into your circle.

3. Start every node with it. With no `--owner-key`, a node uses
   `~/.mesh-llm/owner-keystore.json` when it exists:

   ```bash
   mesh-llm serve --auto --owner-key ~/.mesh-llm/owner-keystore.json --node-label studio
   ```

   A node with an owner key signs a certificate for its endpoint at startup
   and renews it while it runs (certificates last 7 days). Add
   `--owner-required` to refuse to start without one.

4. Check from any of your nodes: the console and `/api/status` show each
   peer's owner and whether it is `verified`. Your nodes show your owner id.

### Add someone you trust

To let a friend's nodes answer your private requests, ask them for their
owner id (`mesh-llm auth status` on their side) and add it on each of your
entry nodes:

```bash
mesh-llm auth trust add <their-owner-id> --label alice
mesh-llm auth trust list
mesh-llm auth trust remove <their-owner-id>
```

or pass `--trust-owner <their-owner-id>` to `serve`. A node reads its trust
store at startup, so restart it after a change. Adding an owner here only
widens private routing; it does not change who may join your mesh unless you
also set `--trust-policy allowlist` (see
[Ownership and admission control](../MESHES.md#private-ownership-and-trust)).
Their nodes keep their own key; you never share yours.

Trust is one-way. Their private requests reach your nodes only if they add
your owner id too.

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
the MoA plugin is on, from the circle only.

`GET /v1/models` with `x-mesh-scope: private` (or under `/private/v1`) lists
only what the circle serves.

## Guarantees

- Never falls back to public nodes. If no node in the circle serves the model,
  the request fails; it is not retried publicly.
- An entry node with no verified owner certificate answers 403.
- The entry node decides once; peers never re-route a forwarded request, so
  the entry node is the whole boundary. `x-mesh-target` and failover only
  choose among nodes in the circle, and the QUIC tunnel refuses any other peer
  as a backstop.
- The selector is consumed at the entry node: `@private` is stripped before
  model lookup and a client's `x-mesh-scope` is not forwarded. A private
  request forwarded to a peer carries `x-mesh-scope: private`. The peer then
  keeps that request to its own owner's and the entry node's owner's nodes,
  the only owners it knows are in the entry node's circle. It does not use
  its own trust store for it. The marker only ever narrows what the peer does.
- A split whose stages include a node outside the circle is not a private
  target: its activations and KV would leave the circle.
- Your own nodes come first: a request from a node of the same owner is
  admitted as local, so a node that pauses remote inference still serves you.
  This priority is for your own owner key only, not for trusted owners.
- Plugins: while a plugin handles a private request (a virtual-model turn
  such as the MoA committee, an OpenAI exchange hook, or a request
  transform), every inference request that plugin makes back to the host is
  sent with `x-mesh-scope: private`, so it only reaches the circle. This is
  per plugin, not per request: a public request the same plugin handles at
  the same moment is narrowed to the circle too. If the host stops waiting
  before the plugin replies (the client left, or the deadline passed), the
  plugin stays private for five more minutes, since it may still be working.
  What a plugin does on its own, outside these calls, is up to the plugin:
  install only plugins you trust.
- A trailing `@private` or `@incognito` is reserved. A model name that is
  only the suffix (`"@private"`) is rejected with 400. Peer-advertised names
  ending in either are not listed, and a peer naming a model `x@private`
  cannot attract a private request. A Hugging Face revision with either name
  must be referenced by commit.

Private routing controls where a request is computed. It does not hide the
request from the nodes in your circle, and it does not encrypt anything
beyond what the mesh's QUIC transport already does.

## Mixed versions

Only the entry node needs this release for the guarantees above, with one
exception. A node of yours running an older release ignores the
`x-mesh-scope: private` marker on a request forwarded to it. If that node
serves the model through a split whose stages include nodes outside your
circle, the request's activations reach those nodes. Upgrade every node in
your circle before relying on private routing with splits.

## Not yet

- The startup (bootstrap) proxy refuses private requests (503) until the node
  is ready.
- A separate prompt-cache namespace, console and launcher (`--private`)
  support.
- Per-turn private scope for MoA nested requests.
