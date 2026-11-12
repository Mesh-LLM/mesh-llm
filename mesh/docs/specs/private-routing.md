# Private routing

Run your nodes in a shared mesh, and still send one request only to nodes you
own.

## Use

Sign every node you own with the same owner key:

```bash
mesh-llm auth init                     # once, on your main machine
mesh-llm serve --auto --owner-key ~/.mesh-llm/owner-keystore.json --node-label studio
```

With no `--owner-key`, a node uses `~/.mesh-llm/owner-keystore.json` when it
exists. A headless node unlocks an encrypted keystore with
`MESH_LLM_OWNER_PASSPHRASE`.

Then ask any of your nodes for private routing, either way:

```bash
curl localhost:9337/v1/chat/completions -d '{"model":"qwen3-32b@private", ...}'
curl localhost:9337/v1/chat/completions -H 'x-mesh-scope: private' -d '{"model":"qwen3-32b", ...}'
```

`GET /v1/models` with `x-mesh-scope: private` lists only what your nodes serve.

## Meaning

"Private" means: only nodes whose ownership certificate verifies against the
same owner as the node the client is talking to, bound to that node's QUIC
endpoint id and not expired. The entry node decides once; peers never re-route
a forwarded request, so the entry node is the whole boundary.

- Never falls back to public nodes. No owned node serving the model is an
  error, not a public retry.
- A node with no verified owner answers 403.
- `x-mesh-target` and failover only choose among owned nodes; the QUIC tunnel
  refuses any other peer as a backstop.
- The selector is consumed at the entry node: `@private` is stripped before
  model lookup and `x-mesh-scope` is not forwarded.
- A trailing `@private` is reserved. Peer-advertised names ending in it are
  not listed, and a peer naming a model `x@private` cannot attract a private
  request: trust comes only from the certificate, never from names, labels or
  hostnames. A Hugging Face revision named `private` must be referenced by
  commit.

## Not yet

- Virtual (MoA) models refuse private requests (409): their nested requests
  would not carry the scope.
- The startup (bootstrap) proxy refuses private requests (503) until the node
  is ready.
- Owned-node priority under load, a separate prompt-cache namespace, a
  `/private/v1` base path, console and launcher (`--private`) support, and
  multi-owner domains.
