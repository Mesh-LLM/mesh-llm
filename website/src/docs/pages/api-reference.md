---
title: API Reference
description: Local management, lifecycle, owner-control, activity, and status APIs
---

# API Reference

Mesh exposes an OpenAI-compatible inference endpoint at
`http://localhost:9337/v1` and a management API at
`http://localhost:3131`.

The lifecycle, activity, configuration, and owner-control management routes are
loopback-only. A non-loopback caller is rejected even if it can reach the
management listener.

## OpenAI-compatible endpoint

Use the inference endpoint for clients, SDKs, and agents:

```text
http://localhost:9337/v1
```

See [OpenAI-Compatible API](/docs/pages/openai-compatible-api/) for model
listing and inference behavior.

For privacy-safe local request history, live replay, export, cleanup, and
terminal webhook delivery, see the
[Local Logging API](/docs/pages/logging-api/).

## Liveness and local readiness

Use `GET /health` for a lightweight management-process probe:

```http
GET /health
```

This endpoint is served on the management port (default `3131`). The
OpenAI-compatible serving port has its own `/health`, `/healthz`, and `/readyz`
probes; those do not include the management endpoint's mesh and local-model
summary.

An answering management process always returns HTTP `200` with
`Content-Type: application/json` and `"status":"ok"`. This is deliberately a
liveness contract: joining a mesh, having peers, loading a model, and
successful inference are not required. The response also reports the
node's current `mode` (`worker`, `client`, or `serving`), compact mesh
connection state, and locally healthy serving models for callers that need
readiness context without fetching `/api/status`.

Representative response:

```json
{
  "status": "ok",
  "mode": "serving",
  "mesh": {
    "status": "connected",
    "admitted_peer_count": 2,
    "connected_peer_count": 2
  },
  "serving": {
    "status": "healthy",
    "models": ["Qwen3-8B-Q4_K_M"]
  }
}
```

The exact `mode` vocabulary is `worker`, `client`, or `serving`. Mesh
`status` is `standalone` when there are no admitted peers, `connected` when at
least one admitted peer has a current control connection, and `disconnected`
when admitted membership exists without a current connection. Serving `status`
is `healthy` when local models or worker stages are ready, `degraded` when
ready work coexists with a terminal local failure, `unhealthy` when local work
has failed and none is ready, `starting` when local work exists but has not
reached readiness, `idle` when a worker/serving node has no local work, and
`not_applicable` for clients. Worker `models` contains ready local split-stage
model IDs; serving `models` contains healthy local or cached plugin-inference
models.

For full mesh topology, routing, model inventory, and runtime diagnostics, use
`GET /api/status`. Do not treat `/health`'s HTTP status as an inference
readiness result.

## Runtime lifecycle

### Load a local model

```http
POST /api/runtime/models
Content-Type: application/json

{"model":"Qwen3-8B-Q4_K_M"}
```

A profile may be appended with `#`, for example
`{"model":"Qwen3-8B-Q4_K_M#low-ctx"}`.

Representative response:

```json
{
  "loaded": "Qwen3-8B-Q4_K_M",
  "instance_id": "model-instance-id"
}
```

### Unload a local model or instance

```http
DELETE /api/runtime/models/Qwen3-8B-Q4_K_M
```

```http
DELETE /api/runtime/instances/model-instance-id
```

Model references and instance IDs in paths must be percent-encoded when they
contain reserved URL characters.

Representative response:

```json
{
  "dropped": "Qwen3-8B-Q4_K_M",
  "instance_id": "model-instance-id"
}
```

Model targeting resolves the selected model. Instance targeting selects one
exact runtime instance.

### Observe intents

```http
GET /api/runtime/intents
```

Representative response:

```json
{
  "intents": [
    {
      "intent_id": "intent-id",
      "model_ref": "Qwen3-8B-Q4_K_M",
      "source": "api_request",
      "desired_state": "load",
      "persistence": "session",
      "created_at_secs": 1770000000,
      "updated_at_secs": 1770000001
    }
  ],
  "total_count": 1,
  "truncated": false
}
```

The list is filtered and capped at 256 entries. Entries can also contain a
profile, instance target, and bounded `last_error`.

Lifecycle requests are intent-driven. An accepted owner lifecycle response
means the target reconciler queued the desired state; it does not mean a model
has finished loading, draining, or stopping. Observe `/api/runtime/intents`,
`/api/status`, `/api/runtime`, and `/v1/models` for completion.

## Owner-control lifecycle

These loopback REST facades ask the local node to contact exactly one remote,
owner-attested target over the private `mesh-llm-control/1` ALPN:

```http
POST /api/runtime/control/load-model
POST /api/runtime/control/ensure-model
POST /api/runtime/control/unload-model
POST /api/runtime/control/drain-model
Content-Type: application/json
```

Load and ensure require a canonical model reference:

```json
{
  "endpoint": "<control-endpoint>",
  "model": "org/model:file.gguf",
  "profile": "low-ctx"
}
```

Unload and drain require exactly one model or instance target and do not accept
a profile:

```json
{
  "endpoint": "<control-endpoint>",
  "instance_id": "model-instance-id"
}
```

Representative accepted response:

```json
{
  "accepted": true,
  "intent_id": "intent-id",
  "accepted_state": "draining",
  "model": "org/model:file.gguf",
  "instance_id": "model-instance-id"
}
```

`load-model` creates a one-shot present intent. `ensure-model` creates a
maintained present intent with bounded retry. `unload-model` creates an absent
intent. `drain-model` rejects new work, permits admitted work to finish, and
force-unloads at the configured deadline.

All four intents are session-only. They do not persist configuration and
disappear when the target daemon restarts.

The `endpoint` token must be read locally from the target's
`GET /api/runtime/control-bootstrap` response and transferred out of band. It
pins one canonical target. The requester must authenticate as the same owner.
Endpoint tokens are not discovered through peer IDs, Nostr, gossip, routes, or
status.

Owner commands never use or silently fall back to the public mesh plane. An
older target returns a typed unsupported result.

## Activity policy

### Read status

```http
GET /api/runtime/activity
```

```json
{
  "effective_state": "remote_paused",
  "override_mode": "active",
  "detector_category": "active"
}
```

Effective states are `accepting`, `accepting_deprioritized`, `remote_paused`,
and `all_paused`. Detector categories are deliberately coarse: `active`,
`idle`, or `unavailable`.

### Set or clear an override

The PUT body is a JSON string, not an object:

```http
PUT /api/runtime/activity/override
Content-Type: application/json

"active"
```

Accepted values are `"auto"`, `"active"`, and `"idle"`. The response has the
same shape as the GET route.

Clear the override and return to automatic detector control:

```http
DELETE /api/runtime/activity/override
```

Overrides are in-memory only and are never persisted to `config.toml`.
Activity policy changes admission or priority; it does not unload models.

## Runtime status

`GET /api/status` retains its existing fields and may add these optional
runtime fields for mixed-version compatibility:

```json
{
  "runtime": {
    "daemon_state": "ready_proxying",
    "capabilities": {
      "worker_capable": true,
      "local_serving": false,
      "proxying": true,
      "plugin_ingress": true,
      "accepting_local": true,
      "accepting_remote": true
    },
    "lifecycle_instances": [
      {
        "instance_id": "model-instance-id",
        "model_ref": "Qwen3-8B-Q4_K_M",
        "state": "draining"
      }
    ],
    "intent_summary": {
      "durable_count": 0,
      "session_count": 1,
      "recent_errors": []
    }
  }
}
```

`daemon_state` precedence is `stopping`, `degraded`, `ready_serving`,
`ready_proxying`, `ready_idle`, then `starting`. Older nodes may omit every new
field; clients must tolerate absence.

## Owner-control inventory scan

Scan one explicitly targeted owner-attested node:

```http
POST /api/runtime/control/scan-refresh
Content-Type: application/json

{"endpoint":"<control-endpoint>"}
```

Example response:

```json
{
  "target_node_id": "<hex-node-id>",
  "disposition": "executed",
  "inventory": [
    {
      "canonical_model_ref": "owner/model:Q4_K_M",
      "display_name": "model",
      "total_size_bytes": 4294967296
    }
  ]
}
```

Inventory entries are sorted by `canonical_model_ref`. `disposition` is
`executed` when this request ran the scan and `coalesced` when it joined an
in-progress scan. When an older owner-control server returns only the legacy
snapshot, the request still succeeds with `disposition` and `inventory` set to
`null`.

The retained `POST /api/runtime/control/refresh-inventory` route is a
compatibility facade with the legacy snapshot-only response shape.

## Privacy boundary

The activity API exposes no window titles, applications, keystrokes, process
names, or raw detector timestamps. Public mesh gossip may contain only the
configured coarse admission projection. Raw owner payloads, model and instance
targets, endpoint tokens, intent state, and rich inventory results remain on
the private/local control surfaces and are not copied wholesale into public
gossip, `/api/status`, or telemetry.

For the complete state model and mixed-version behavior, see
[Runtime Lifecycle](/docs/pages/runtime-lifecycle/).

## CLI

For command-line equivalents, see the [CLI reference](/docs/pages/CLI/).
