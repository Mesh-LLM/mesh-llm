---
title: Runtime Lifecycle
description: Run Mesh as a durable daemon, manage models on demand, and control admission safely
---

# Runtime Lifecycle

Mesh is a daemon first and a model runner second. The API, console, mesh
membership, plugins, and private owner-control listener can be ready even when
no local inference process is running. Models are managed resources that can be
loaded, drained, replaced, or stopped during the daemon session.

This distinction matters operationally: a healthy zero-model node can still
route to a peer or plugin, accept lifecycle commands, expose status, and later
load a local model without restarting.

## Healthy zero-model states

`GET /api/status` reports a derived `daemon_state`:

| State | Meaning |
|---|---|
| `starting` | The durable listeners are not ready yet. |
| `ready_idle` | The listeners are ready, but no local, plugin, or remote model route is available. |
| `ready_proxying` | No local model is serving, but at least one healthy plugin or remote route is available. |
| `ready_serving` | At least one local model is serving. |
| `degraded` | A policy-relevant terminal model failure or priority-restoration failure exists. |
| `stopping` | Shutdown has begun. |

The exact precedence is `stopping` → `degraded` → `ready_serving` →
`ready_proxying` → `ready_idle` → `starting`. Capability flags alongside the
state report whether the node is `worker_capable`, `local_serving`, `proxying`,
has `plugin_ingress`, and is `accepting_local` or `accepting_remote`.

## Runtime modes

Configure the mode in `~/.mesh-llm/config.toml`:

```toml
[runtime]
mode = "serve"
```

| Mode | Configured `[[models]]` | Explicit `--model` or `--gguf` | Local lifecycle commands |
|---|---|---|---|
| `serve` | Loaded eagerly at startup | Loaded eagerly | Enabled |
| `on_demand` | Kept as candidate and preference metadata; not loaded eagerly | Loaded eagerly | Enabled |
| `client` | Never loaded locally | Invalid | Disabled |

`serve` is the compatibility default. Bare `mesh-llm serve` is valid even when
there are no configured models.

`on_demand` starts a worker-capable daemon without eagerly loading configured
models. Explicit CLI model arguments remain an operator instruction and
therefore remain eager.

`client` is routing-only. A persisted `runtime.mode = "client"` conflicts with
explicit serving and model flags such as `--model`, `--gguf`, `--split`, and
other local-serving options. Use `on_demand` when the node should remain able to
load a model later.

## Startup failure policy

```toml
[runtime]
startup_failure_policy = "best_effort"
```

- `best_effort` is the default. The daemon records an eager model failure,
  continues bringing up durable surfaces, and may enter `degraded`.
- `fail_fast` aborts startup when an eager startup model fails.

The policy applies to eager startup work, not to later accepted lifecycle
intents.

## Local lifecycle control

Use local commands against the management API on the same machine:

```bash
mesh-llm load Qwen3-8B-Q4_K_M --port 3131
mesh-llm unload Qwen3-8B-Q4_K_M --port 3131
mesh-llm status --port 3131
```

The REST equivalents are:

```bash
curl -s -X POST localhost:3131/api/runtime/models \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-8B-Q4_K_M"}'

curl -s -X DELETE \
  localhost:3131/api/runtime/models/Qwen3-8B-Q4_K_M

curl -s -X DELETE \
  localhost:3131/api/runtime/instances/<instance-id>
```

Model-targeted unload applies to the resolved model. Instance-targeted unload
selects one exact running instance.

Lifecycle mutation is asynchronous. An accepted response means the reconciler
accepted the desired intent; it does not mean the model has finished loading or
unloading. Observe completion with:

```bash
mesh-llm status --port 3131
curl -s localhost:3131/api/runtime/intents | jq .
curl -s localhost:3131/api/status | jq '.runtime'
curl -s localhost:9337/v1/models | jq '.data[].id'
```

Lifecycle states include accepted intent, loading, serving, draining, failed,
and stopped outcomes. `/api/runtime/intents` returns at most 256 filtered
entries plus `total_count` and `truncated`. Each entry includes its source,
desired state, session or durable persistence, timestamps, optional instance
target, and a bounded recent error.

## Owner-control lifecycle

Use owner-control when the target is another node belonging to the same owner:

```bash
mesh-llm runtime load-model \
  --endpoint '<control-endpoint>' --model Qwen3-8B-Q4_K_M

mesh-llm runtime ensure-model \
  --endpoint '<control-endpoint>' --model Qwen3-8B-Q4_K_M

mesh-llm runtime unload-model \
  --endpoint '<control-endpoint>' --model Qwen3-8B-Q4_K_M

mesh-llm runtime drain-model \
  --endpoint '<control-endpoint>' --instance-id '<instance-id>'
```

- `load-model` creates a one-shot present intent.
- `ensure-model` maintains a present intent with bounded retry after transient
  failures.
- `unload-model` creates an absent intent for one canonical model reference or
  one exact instance ID.
- `drain-model` rejects new work, lets admitted work finish, then unloads at
  zero in-flight. At the configured deadline it force-cancels remaining work
  and unloads.

Obtain the target's explicit endpoint token locally with
`mesh-llm runtime bootstrap --json` or
`GET /api/runtime/control-bootstrap`, then transfer it out of band. The token
cryptographically pins one canonical target; it is never inferred from a peer
ID, gossip, Nostr, routing, or public status. The requester must authenticate
as the same owner.

Owner lifecycle intents last only for the target daemon session. They never
edit `config.toml`. Use owner-control config apply when a change must persist.
Responses acknowledge queueing within the command deadline, not completion;
observe the target's local intent and runtime status surfaces.

Owner commands travel over the private `mesh-llm-control/1` ALPN. They do not
use public mesh gossip or request streams, and never silently fall back to
those surfaces.

## Draining

Draining is safer than immediate unload when requests may be in flight:

```toml
[runtime]
drain_timeout_secs = 30
drain_timeout_max_secs = 300
```

`drain_timeout_secs` is the normal deadline and must be from 1 to 3600 seconds.
`drain_timeout_max_secs` is the owner-command cap, also from 1 to 3600 seconds.
The normal timeout cannot exceed the maximum. During drain, new requests for
the instance are temporarily unavailable; already-admitted work may complete.

## Activity-aware admission

Activity adaptation is opt-in and changes admission, not model residency:

```toml
[runtime.activity]
enabled = true
idle_after_secs = 300
poll_interval_secs = 5
resume_debounce_secs = 30
response = "pause_remote"
advertisement = "coarse_state"
```

The detector applies the configured response while the host is active:

| Response | Behavior |
|---|---|
| `pause_remote` | Reject remote mesh and stage work; keep local, plugin, and management access available. |
| `pause_all` | Reject all new inference; keep management and owner-control available. |
| `reduce_priority` | Continue accepting inference with reduced process priority. |

No response unloads a model. When the policy resumes, the already-loaded model
can accept work again.

Inspect and temporarily override the detector:

```bash
curl -s localhost:3131/api/runtime/activity | jq .

curl -s -X PUT localhost:3131/api/runtime/activity/override \
  -H 'Content-Type: application/json' \
  -d '"active"' | jq .

curl -s -X DELETE \
  localhost:3131/api/runtime/activity/override | jq .
```

Override modes are `auto`, `active`, and `idle`. `active` forces the configured
capacity-yielding response; `idle` forces normal admission; `auto` returns to
detector control. Overrides are in-memory and disappear on restart.

The activity API intentionally exposes only `effective_state`,
`override_mode`, and the coarse `detector_category` (`active`, `idle`, or
`unavailable`). It does not reveal applications, window titles, keystrokes,
process names, or raw detector timestamps.

Mesh advertisement is similarly coarse. `advertisement` accepts `none`,
`availability_only`, `coarse_state` (default), or `private_coarse_state`.
Peers may see only accepting, deprioritized, remote-paused, or all-paused
admission. Raw owner intent payloads, model/instance targets, endpoint tokens,
and private command results are not copied into public gossip, status, or
telemetry.

## Model discovery and request errors

`GET /v1/models` lists routes that are usable or advertised through local
serving, plugins, or the mesh. A healthy `ready_idle` daemon can correctly
return an empty `data` array. A `ready_proxying` daemon can list plugin or
remote models without any local model process.

A request for a model with no known route is an expected model-not-found
response (`404`, `model_not_found`). A known route that is draining, paused, or
otherwise temporarily unable to admit work returns a service-unavailable
response (`503`, `service_unavailable`). Clients may retry temporary
unavailability after policy or lifecycle state changes.

## Mixed versions and security boundary

Public mesh compatibility remains additive: older and newer nodes can join,
gossip, and route together, and older peers ignore the optional coarse
admission field. An older peer may temporarily route to a paused newer node
because it cannot understand that hint.

Owner lifecycle commands have a separate compatibility boundary. Older targets
that do not implement them return a typed unsupported result. Current clients
do not downgrade the command onto the public mesh.

The management lifecycle and activity facades are loopback-only. Owner-control
adds explicit target pinning and same-owner authentication for a single remote
node. Keep the inference/data plane, public mesh plane, and private owner
control plane distinct when exposing ports or debugging policy.

## Foreground process versus OS service

`mesh-llm serve` runs the daemon in the foreground and keeps the terminal
attached. It is the clearest mode for development and direct observation.

`mesh-llm setup` can install a launchd service on macOS or a systemd user
service on Linux. Those service managers start the same logical runtime in the
background and own restart/logging behavior. Starting a second foreground
daemon against the same ports does not control the service; stop or disable
the setup-managed service first when switching execution styles.

## Related guides

- [Config File](/docs/pages/config-toml/#runtime)
- [Config Models & Plugins](/docs/pages/config-models/)
- [API Reference](/docs/pages/api-reference/#runtime-lifecycle)
- [CLI Reference](/docs/pages/CLI/#runtime-lifecycle-and-modes)
- [OpenAI-Compatible API](/docs/pages/openai-compatible-api/)
- [Troubleshooting](/docs/pages/troubleshooting/#runtime-lifecycle)
