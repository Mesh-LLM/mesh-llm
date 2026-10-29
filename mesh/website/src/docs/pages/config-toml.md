---
title: Config File
description: mesh-llm configuration via ~/.mesh-llm/config.toml and environment variables
---

# Config File

mesh-llm reads configuration from `~/.mesh-llm/config.toml` by default. You can override the path with the `MESH_LLM_CONFIG` environment variable.

The file uses standard [TOML](https://toml.io/en/) syntax. Config is validated when loaded and on save; invalid files produce a clear error at startup.

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

Paid inference is not built into the node. It is provided by the `pay-lightning`
plugin together with a separately installed `wallet.v1` plugin (for example the
external `lexe-wallet`). The node only supplies the generic pieces: an
operation binding that lets the payments plugin reach the wallet, and an
exchange grant that lets it admit or refuse requests. `mesh-llm plugins install`
grants both from the package manifest, so installing the two plugins is the
whole setup; prices and spending policy are then set with `mesh-llm wallet
pricing …` / `mesh-llm wallet policy …` and live in the payments plugin's own
JSON file under `~/.mesh-llm/payments/`, not in `config.toml`.

The `[[plugin]]` entry below is the optional operator override: it replaces the
installed grants with explicit ones (a named wallet provider and open payload, a
separate payment JSON file).

```toml
[[plugin]]
name = "pay-lightning"
settings = { config_path = "/absolute/payment/config.json", wallet_binding = "wallet", allow_wallet_provisioning = false }
allow_peer_blocks = true

[plugin.openai_exchange_grant]
endpoints = ["chat_completions", "completions", "responses"]
phases = ["request_received", "backend_selected", "exchange_finished"]
headers = ["x-plugin-7061792d6c696768746e696e67-pay-terms"]
admission = true
metadata = true
deadline_ms = 30000
failure_policy = "required"

[[plugin.operation_bindings]]
name = "wallet"
provider = "lexe-wallet"
profile = "default"
setup_operation = "wallet_open"
setup_input_json = '{"directory":"/absolute/existing/wallet-state"}'
operations = ["wallet_balance", "wallet_transactions", "wallet_create_invoice", "wallet_pay", "wallet_lookup", "wallet_wait_for_payment", "wallet_wait_for_arrival"]

[[plugin]]
name = "lexe-wallet"
```

- `settings.config_path` — the payments plugin's JSON file (ledger directory,
  prices, buy ceilings, policy). Defaults never authorize spending.
- `settings.wallet_binding` — the `operation_bindings` entry the plugin may use.
  The binding fixes the wallet provider, its open payload and the allowed
  operations; the plugin cannot choose a wallet or open directory itself.
- `settings.allow_wallet_provisioning` — required once, to adopt a wallet into a
  fresh ledger. Afterwards the ledger pins the wallet identity, and a changed
  identity is refused.
- `openai_exchange_grant` — when present, replaces the hook grant made at
  install; it must still grant the phases, header and admission access the
  plugin's manifest requires, or the plugin cannot gate requests.

A node without the payments plugin serves free inference only; with it
installed and nothing configured, the defaults are non-spending (no prices,
`free_only` policy). See the
[plugin README](https://github.com/Mesh-LLM/ln-payments-plugin/blob/main/crates/mesh-llm-pay-lightning/README.md)
for the payment JSON and the [exchange contract](https://github.com/Mesh-LLM/mesh-llm/blob/main/mesh/docs/plugins/openai-exchange-lifecycle.md)
for the grant fields.

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
