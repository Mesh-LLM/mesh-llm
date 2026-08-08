---
name: mesh-join
description: Use this skill when creating, joining, publishing, or connecting mesh-llm nodes into a mesh — private meshes with invite tokens, the public mesh via --auto, named/published meshes, client-only nodes, NAT/firewall/bind issues, or verifying multi-node setups.
metadata:
  short-description: Create and join mesh-llm meshes
---

# mesh-join

Use this when wiring two or more mesh-llm nodes together, or attaching a
client-only node to an existing mesh. Per-platform install/serve steps live in
`deploy-macos` and `deploy-linux-gpu`; this skill covers the mesh topology
itself. Full reference: `docs/MESHES.md`.

## Mental model

- A node can **serve** models (`serve`), be an **API-only client** (`client`),
  or both at once.
- Starting `serve` with no `--join`/`--discover`/`--auto` **creates a private
  mesh** and emits an invite token.
- `--auto` discovers published meshes (Nostr by default) and joins the best
  one — the public community mesh in practice.
- `--publish` makes your mesh discoverable; without it the mesh is private and
  joinable only via the invite token.
- Every node exposes the same OpenAI API on `:9337`; `/v1/models` returns the
  union of local + peer models and requests route by the `model` field.

## Public mesh (the easy path)

```bash
mesh-llm serve --auto                  # serve hardware + join the public mesh
mesh-llm serve --model <ref> --auto    # serve a specific model + join
mesh-llm client --auto                 # API-only client, no GPU needed
```

Confirm joining via `discovery_joined` in the log (use `--log-format json` for
machine-readable events) or `peers` in `/api/status`.

## Private mesh: create + join

```bash
# Node A — creates the mesh, prints an invite token
mesh-llm serve --model Qwen3-8B-Q4_K_M
```

Grab the token: with `--log-format json` it is the `invite_token` event
(`token` field). In pretty mode it is printed to the terminal at startup.

```bash
# Node B — another serving node
mesh-llm serve --join <token>

# Or an API-only client
mesh-llm client --join <token>
```

`--join` is repeatable. Requirement-aware meshes (version/attestation policy)
use signed bootstrap tokens; legacy/private meshes use the older unsigned
token. Either way, the flow above is the same.

## Published / named meshes

```bash
# Publish for discovery, with a friendly name
mesh-llm serve --model Qwen3-8B-Q4_K_M --publish --mesh-name "lab-a"

# Join by name from anywhere
mesh-llm serve --discover "lab-a"
mesh-llm client --discover "lab-a"

# Browse what's out there
mesh-llm discover
mesh-llm discover --name "lab-a"
mesh-llm discover --model qwen --min-vram 24
mesh-llm discover --auto        # prints the best invite token (script-friendly)
```

`--mesh-name` without `--publish` is only a local label — the mesh stays
private.

## LAN-only discovery

`--mesh-discovery-mode mdns` keeps discovery and transport startup LAN-only:
no Nostr relays, no public iroh relays, no public STUN. Joins still require a
supplied matching invite token (mDNS advertisements only carry fingerprints).

## NAT, firewalls, multi-interface hosts

- Default Nostr mode uses managed iroh relays when direct UDP paths fail —
  usually no port forwarding is needed.
- For direct connectivity, pin QUIC with `--bind-port <UDP_PORT>` on the
  mesh-creating node and forward that UDP port. **Only the creator side needs
  forwarding**; joiners don't.
- On multi-interface Linux/Docker hosts (`--network host`), iroh may advertise
  bridge addresses like `172.17.0.1` that collide across machines. Pin the real
  interface: `--bind-ip <host-ip> --bind-port <port>`.
- `--listen-all` only affects the local HTTP API/console listener, not mesh
  QUIC.

## Verify a multi-node mesh

```bash
# Peers on each node (expect N-1)
curl -s http://localhost:3131/api/status | python3 -m json.tool

# Union of models across the mesh
curl -s http://localhost:9337/v1/models | python3 -m json.tool

# Route to a specific peer's model — the response "model" field
# confirms which node answered
curl -s http://localhost:9337/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<peer-model-id>","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```

`/api/status` also reports the publication state (`private`, `public`,
`publish_failed`).

## Splitting big models across nodes

When one node cannot fit the model, use Skippy layer splits — same mesh
mechanics plus `--split` and a layer-package model on every serving node. See
`docs/SKIPPY_SPLITS.md`; diagnose readiness with `mesh-llm doctor split`.

## Ownership / trust (private deployments)

For owner-attested meshes: `mesh-llm auth init`, then start nodes with
`--owner-key`, `--node-label`, `--trust-policy`, `--trust-owner`. Details in
`docs/MESHES.md` ("Private ownership and trust").

## Gotchas

- Two instances on one machine need distinct ports: `--port` (API, default
  9337) and `--console` (management, default 3131).
- Model load after join takes time — poll `/v1/models`, don't assume failure.
- Clients are zero-state on the host side: a `client` node doesn't appear in
  the host's peer list. That's expected, not a bug.
- `--headless` only hides the web UI; the management API stays on `--console`.
