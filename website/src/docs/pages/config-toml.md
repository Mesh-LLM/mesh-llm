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
advertise_addr = ""             # Override the address advertised to peers
```

- `bind` — address and port for the owner-control endpoint. The inference API remains on `9337` unless changed by the runtime surface.
- `advertise_addr` — address and port announced to peers. Auto-detected when empty; set it explicitly for NAT or Docker setups.

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

Runtime behavior and model reconciliation.

```toml
[runtime]
reconcile_model_targets                = false
reconcile_model_target_demand_upgrades = false
model_target_demand_upgrade_min_requests = 2
model_target_demand_upgrade_max_age_secs = 3600
```

Model target reconciliation keeps running models aligned with configured targets. Demand upgrades use the minimum request count and maximum request age above when deciding whether a higher-quality target is warranted.

## Deep Dives

| Page | What it covers |
|---|---|
| [Config Defaults](/docs/pages/config-defaults/) | Model defaults: memory sizing, hardware, throughput, skippy, speculative, sampling |
| [Config Models & Plugins](/docs/pages/config-models/) | Model entries and plugin configuration |
| [Config Reference](/docs/pages/config-reference/) | Environment variables, CLI commands, rejected fields |
