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
parallel  = {}            # Optional GPU parallel config map
```

- `assignment` — `"auto"` lets mesh-llm assign GPUs dynamically; `"pinned"` locks models to specific GPUs.
- `parallel` — map of device type to parallel config (rarely needed; `auto` handles most cases).

## Owner Control

Network identity and binding.

```toml
[owner_control]
bind           = "[::]:9337"   # Listen address for inference API
advertise_addr = ""             # Override advertised address (auto-detected if empty)
```

- `bind` — address and port for the inference HTTP server. Default `[::]:9337`.
- `advertise_addr` — advertised address for mesh discovery. Auto-detected when empty. Set this explicitly for NAT or Docker setups.

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
min_node_protocol         = 0    # Minimum acceptable peer protocol version
max_node_protocol         = 0    # Maximum acceptable peer protocol version
min_release_version       = ""   # Minimum release version string
require_release_attestation = false
release_signer_keys       = []   # List of allowed release signer public keys
```

- `min_node_protocol` / `max_node_protocol` — constrain which protocol versions to accept. `0` means no constraint.
- `min_release_version` — reject peers running older release versions. Empty means no constraint.
- `release_signer_keys` — if set, only accept peers whose release binary is signed by one of these keys.

## Runtime

Runtime behavior and model reconciliation.

```toml
[runtime]
reconcile_model_targets              = true
reconcile_model_target_demand_upgrades = false
model_target_demand_upgrade_interval_secs = 300
model_target_demand_upgrade_threshold = 0
model_target_demand_upgrade_min_wait_secs = 0
```

Model target reconciliation automatically adjusts running models to match configured targets. Demand upgrades allow the runtime to upgrade model quality (e.g., larger quantizations) when demand metrics cross the threshold.

## Deep Dives

| Page | What it covers |
|---|---|
| [Config Defaults](/docs/pages/config-defaults/) | Model defaults: memory sizing, hardware, throughput, skippy, speculative, sampling |
| [Config Models & Plugins](/docs/pages/config-models/) | Model entries and plugin configuration |
| [Config Reference](/docs/pages/config-reference/) | Environment variables, CLI commands, rejected fields |
