---
title: Config File
description: mesh-llm configuration via ~/.mesh-llm/config.toml and environment variables
---

# Config File

mesh-llm reads configuration from `~/.mesh-llm/config.toml` by default. You can override the path with the `MESH_LLM_CONFIG` environment variable.

The file uses standard [TOML](https://toml.io/en/) syntax. Config is validated when loaded and on save; invalid files produce a clear error at startup.

To serve one local GGUF across independent meshes, use a separate local
`memberships.toml` with `serve --memberships-config`. Membership identities,
trust inputs, and shared host limits belong in that manifest, outside the
ordinary configuration and web config editor. See
[One model across independent meshes](/docs/pages/shared-serving/).

## Version

```toml
version = 1
```

Must be `1`. This is the only supported schema version. Future schema changes will bump this field for backward compatibility.

## GPU

Controls GPU assignment and parallelism.

```toml
[gpu]
assignment = "auto"       # "auto" (default) or "pinned"
parallel  = 2              # Optional total parallel inference slots
```

- `assignment` — `"auto"` lets mesh-llm assign GPUs dynamically; `"pinned"` locks models to specific GPUs.
- `parallel` — optional total parallel inference-slot count. Omit it to let the runtime choose.

## Owner Control

Network identity and binding.

```toml
[owner_control]
bind           = "[::]:7447"   # Mesh owner-control listen address
# advertise_addr = "203.0.113.10:7447"  # Override the address advertised to peers
```

- `bind` — address and port for the owner-control endpoint. The inference API remains on `9337` unless changed by the runtime surface.
- `advertise_addr` — address and port announced to peers. Auto-detected when unset; set it explicitly for NAT or Docker setups, with the same port as `bind`.

## Payments

Paid inference is not part of the node and has no `config.toml` section. Payment
rails are ordinary plugins that use the generic seams described above (operation
bindings, the OpenAI-exchange hook, model `requires`). The Lightning rail lives in
[Mesh-LLM/ln-payments-plugin](https://github.com/Mesh-LLM/ln-payments-plugin);
its README covers installation, pricing and spending policy.

Install it together with one wallet plugin in a single command:

```sh
mesh-llm plugins install Mesh-LLM/lexe-wallet-plugin Mesh-LLM/ln-payments-plugin
# or, for a Nostr Wallet Connect wallet:
mesh-llm plugins install Mesh-LLM/nwc-wallet-plugin Mesh-LLM/ln-payments-plugin
```

A wallet plugin alone does not enable payments, and `pay-lightning` without a
wallet refuses every wallet call.

## Receipts

Inference receipts for the models this node serves, alone or as the
coordinator of a split. A receipt lets a client check which model computed a
turn.

Receipts are experimental and off by default. An audit checks a sample of
each turn, and some parts of the design are not built. Read the
[open limitations](https://github.com/Mesh-LLM/mesh-llm/blob/main/mesh/docs/design/RECEIPTS.md#open-limitations)
before you rely on them. The
[receipts guide](https://github.com/Mesh-LLM/mesh-llm/blob/main/skippy/docs/RECEIPTS.md)
has the procedures.

```toml
[receipts]
enabled = true   # issue receipts for the models this node serves
audit = false    # audit the receipted turns peers serve this node's clients
require = false  # route this node's clients only to runtimes with receipts
offline = false  # never audit, and never contact Hugging Face for an audit
```

All four are off by default. A key the node does not know under `[receipts]`,
such as a misspelt `require`, has no effect. `mesh-llm config validate` reports
it as an unknown key, so run it after you change this table.

- `enabled` — when a request names a client key in `x-receipt-client-key`, the
  node signs a receipt for the turn with its node key and returns it in the
  response body (the `receipt` field, or the last stream chunk). The client can
  then challenge it once on `POST /v1/receipts/open` and check the opening
  against the claimed model. Read when a model's runtime starts.
- The node tells its peers that it serves a model with receipts only when the
  loaded runtime does issue them, and all of the model runs on this node. It
  does not for a backend or a model the receipt recorder cannot run on, for a
  backend no auditor can check (only the CPU and CUDA have rounding
  profiles; a runtime on Vulkan, Metal or ROCm issues no receipts), for a
  model that does not serve chat completions, for a model served from a local
  path (its receipts name `local-gguf/sha256-...`, which no client can fetch),
  or for a split with a stage on another node. The log says why once, when
  the model loads.
- Keep the node's sampling defaults neutral when `enabled` is on. A default
  penalty, logit bias or stop string makes an honest node's receipts
  contradict the requests of auditing peers, which gives the node strikes.
- On a split, every stage records its part of the receipt, and the peers that
  serve the later stages record theirs when the coordinator asks for it,
  whatever their own setting. A peer that predates split receipts records
  nothing, and the turns of that split get no receipt.
- `audit` — the node names a fresh receipt key on every chat completion it
  forwards to a peer for a local client, and audits about one in ten of the
  receipted turns. An audit reads the claimed model from Hugging Face by
  range and keeps the parts it read in the platform cache directory
  (`~/.cache/mesh-llm/receipts/` on Linux). When an audit fails in something
  the peer signed, the node blocks the peer for seven days and keeps the
  evidence under `receipt-evidence/` in its state directory (at most 64 files
  and 1 GiB, readable by the owner only). The node also checks the receipt of
  every turn a peer serves: a peer that advertises receipts and returns no
  usable receipt gets a strike. So does a peer that does not answer a
  challenge, or that fails a check only this node saw. Three strikes in 24
  hours block the peer for seven days. Strikes are kept in memory. An audit
  that cannot run on this node has no effect on the peer. The node audits
  only when it can tie the request's model name to a Hugging Face file: name
  the model as `owner/repo:selector`, as `owner/repo@revision/file.gguf`, or
  by a name in this node's loaded catalog. Every watched turn writes one log
  line with its verdict. Read for every request.
- `require` — the node's clients get chat completions only from a runtime
  that issues receipts: this node's own, or a peer that advertises them, whose
  turns the node then audits even with `audit` off. A request nothing can
  serve that way fails with HTTP 409 and the error code
  `trust_policy_unsatisfied`. So does every other inference endpoint
  (embeddings, realtime sessions, plugin and virtual models), since receipts
  cover chat completions only. `GET /v1/models` and `POST /v1/receipts/open`
  still work. Read for every request.
- `offline` — the node never audits a receipt, whatever `audit`, `require` or a
  request ask, so it never contacts Hugging Face and keeps no model data for an
  audit. A peer's receipts then go unchecked and no peer counts as serving
  with receipts: a request that requires them is served by this node's own
  runtime, or fails with `trust_policy_unsatisfied` and a message that names
  this setting. The node then watches no turn, so its log has no line per
  turn. Read for every request.

A single request asks for the same policy as `require` with the header
`x-mesh-trust: receipts`. The header can narrow the node's policy, not widen
it: on a node with `require = true`, `x-mesh-trust: any` is an error.

A failed audit blocks the peer. It does not withhold a payment.

## Telemetry

OpenTelemetry (OTLP) metrics export.

```toml
[telemetry]
enabled            = false                  # Enable OTLP metrics export
service_name       = "mesh-llm"             # OTLP service.name attribute
endpoint           = "http://localhost:4318" # OTLP HTTP endpoint
headers            = {}                     # Extra headers sent with OTLP requests
export_interval_secs = 10                   # Metrics export interval
queue_size         = 2048                   # Metrics queue capacity

[telemetry.metrics]
endpoint = "http://localhost:4318"           # Override endpoint for metrics (optional)
```

The `[telemetry.metrics]` sub-section is optional — it overrides the top-level endpoint specifically for metric signals.

## Mesh Requirements

Peer compatibility constraints.

```toml
[mesh_requirements]
# Optional node/release bounds. Omit these unless the mesh needs an admission gate.
# min_node_version         = "0.72.0"
# max_node_version         = "0.72.1"
min_protocol_version      = 0    # Minimum acceptable peer protocol version
max_protocol_version      = 0    # Maximum acceptable peer protocol version
require_release_attestation = false
release_signer_keys       = []   # Allowed release signer public keys
```

- `min_node_version` / `max_node_version` — optional semantic-version bounds for peers.
- `min_protocol_version` / `max_protocol_version` — constrain peer protocol versions. `0` means no constraint.
- `release_signer_keys` — if set, only accept peers whose release binary is signed by one of these keys.

## Runtime

Runtime mode, startup behavior, draining, activity-aware admission, and model
reconciliation.

```toml
[runtime]
mode = "serve"
startup_failure_policy = "best_effort"
drain_timeout_secs = 30
drain_timeout_max_secs = 300
reconcile_model_targets                = false
reconcile_model_target_demand_upgrades = false
model_target_demand_upgrade_min_requests = 2
model_target_demand_upgrade_max_age_secs = 3600

[runtime.activity]
enabled = false
idle_after_secs = 300
poll_interval_secs = 5
resume_debounce_secs = 30
response = "pause_remote"
advertisement = "coarse_state"
```

| Mode | Configured models | Explicit CLI models | Local loads |
|---|---|---|---|
| `serve` (default) | Eager | Eager | Allowed |
| `on_demand` | Candidate metadata; not eager | Eager | Allowed |
| `client` | Never loaded locally | Conflicts | Disabled |

`startup_failure_policy` accepts `best_effort` (default, keep the daemon
running) or `fail_fast` (abort when an eager model fails).
`drain_timeout_secs` and `drain_timeout_max_secs` must each be from 1 to 3600,
and the normal timeout cannot exceed the maximum.

Activity adaptation is opt-in. `idle_after_secs` is 30–86400,
`poll_interval_secs` is 1–60, and `resume_debounce_secs` is 0–300. `response`
accepts `pause_remote`, `pause_all`, or `reduce_priority`. `advertisement`
accepts `none`, `availability_only`, `coarse_state`, or
`private_coarse_state`. Manual activity overrides are session-only and are not
written to this file.

Model target reconciliation keeps running models aligned with configured
targets. Demand upgrades use the minimum request count and maximum request age
above when deciding whether a higher-quality target is warranted. See
[Runtime Lifecycle](/docs/pages/runtime-lifecycle/) for the complete behavior,
security, and status model.

## Deep Dives

| Page | What it covers |
|---|---|
| [Config Defaults](/docs/pages/config-defaults/) | Model defaults: memory sizing, hardware, throughput, skippy, speculative, sampling |
| [Config Models & Plugins](/docs/pages/config-models/) | Model entries and plugin configuration |
| [Config Reference](/docs/pages/config-reference/) | Environment variables, CLI commands, rejected fields |
| [Runtime Lifecycle](/docs/pages/runtime-lifecycle/) | Modes, startup policy, lifecycle control, draining, and activity admission |
