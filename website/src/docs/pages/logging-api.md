---
title: Local Logging API
description: Query, stream, export, and maintain privacy-safe request logs on one Mesh node
---

# Local Logging API

Mesh keeps a bounded operational history for the node on which it is running.
The logging API exposes privacy-safe request summaries, lifecycle events,
artifact metadata, proxy attempts, and audit entries through the management
listener, normally `http://localhost:3131`.

This is an operator API, not an inference API. Applications should continue to
use the [OpenAI-compatible API](/docs/pages/openai-compatible-api/) on port
`9337` for inference.

In the embedded Logs console, ledger search accepts either a full endpoint ID
or the abbreviated form shown in the interface, such as `9f0c…bb04`, for
request caller identities and mesh-peer audit subjects.

## Scope and trust boundary

Every `/api/logs` and `/api/logs/**` route requires a trusted local caller. A
request passes that boundary only when all applicable checks succeed:

- The TCP peer is an IPv4 or IPv6 loopback address, including an IPv4-mapped
  loopback address.
- A present `Host` header names `localhost` or a loopback address. The SSE
  endpoint requires one unambiguous `Host` header.
- A present `Origin` header uses `http` or `https` and names `localhost` or a
  loopback address. Requests without `Origin`, such as `curl`, are allowed.

A bearer token, a trusted DNS name, or network reachability does not replace
these checks. If a local reverse proxy is used, it must connect from loopback
and preserve a trusted local `Host` and `Origin`. Rejected requests receive the
normal typed `403 forbidden` response before the logging store is touched.

The ledger and its bounded export API remain trusted-local. Audit and request
records are not read by OTLP exporters or telemetry surveys, advertised in
gossip, replicated, or otherwise sent to peer nodes.

Examples below assume:

```bash
LOG_API=http://localhost:3131
```

## Status and health

`GET /api/status` includes an optional `logging` object after local logging
state has initialized:

```json
{
  "logging": {
    "metadata_available": true,
    "artifact_capture_available": false,
    "artifact_capture_degradation": "artifact_capture_disabled_privacy_unavailable",
    "persistence": {
      "state": "running",
      "queue_drops": 0,
      "failures": 0,
      "shutdown_losses": 0,
      "outstanding": 0
    },
    "cleanup": {
      "state": "running",
      "shutdown_timeouts": 0,
      "last_outcome": "completed",
      "last_deleted_count": 12
    }
  }
}
```

This projection describes only the current node. It contains fixed state
labels and counters, never storage paths, backend errors, request identifiers,
or endpoint URLs. The field is additive and omitted when logging has not been
initialized. `GET /api/status` retains its existing access policy; the
trusted-local rule applies specifically to `/api/logs/**`.

`metadata_available` and `artifact_capture_available` are independent. Mesh
can continue recording metadata if privacy-safe artifact capture has tripped
its circuit breaker. Persistence states are `not_started`, `running`,
`stopping`, `stopped`, or `unavailable`; cleanup states are `not_started`,
`running`, `stopping`, `timed_out`, or `stopped`.

## Read API

List endpoints return a keyset page:

```json
{
  "items": [],
  "nextCursor": null
}
```

Pass a non-null `nextCursor` back as `cursor`. Cursors are opaque and bound to
their query scope. A malformed cursor returns `invalid_cursor`; a forged,
expired, or filter-mismatched cursor returns `cursor_expired`. Do not decode,
modify, or reuse a cursor with different filters. Unless noted otherwise,
`limit` defaults to 50 and accepts 1 through 100, while `sort` is `asc` or
`desc`.

### Requests

`GET /api/logs/requests` merges the bounded active snapshot with durable
history. An active request takes precedence over a durable row with the same
ID. Its response contains `requestId`, `outcome`, `createdAt`, optional
`terminalAt`, `route`, `model`, `provider`, `engine`, `statusCode`,
`callerEndpointId`, `callerAddr`, and `callerPathType`, plus a `source` of
`active` or `durable`.

Caller fields are observational and optional. The vocabulary is `"local_http"`,
`"remote_quic_http"`, and `"relay"`. A direct remote HTTP path uses
`callerPathType: "remote_quic_http"` and includes the observed peer address
when available. A relay path uses `"relay"` and omits `callerAddr`; a relay
server address is never substituted. Local management requests use
`"local_http"` and have no mesh endpoint ID. Caller fields are absent when
attribution was not observed, rather than being fabricated. Skippy stage streams have no top-level
request ID and therefore never create request-summary rows. Staged QUIC is
represented as authenticated QUIC audit evidence, not as a synthetic request
row.

Authentication can establish `callerEndpointId` when the selected path was not
observed or its value was not recognized. Such a request omits `callerAddr` and
`callerPathType`; the API does not infer `"remote_quic_http"` or `"relay"`.
Endpoint-only identity is not a fourth caller path type.

Supported query parameters:

| Parameter                              | Values                                                                  |
| -------------------------------------- | ----------------------------------------------------------------------- |
| `limit`, `cursor`, `sort`              | Keyset pagination controls; request listing defaults to `sort=desc`.    |
| `from`, `to`                           | Inclusive RFC 3339 creation-time bounds.                                |
| `route`, `model`, `provider`, `engine` | Exact metadata matches, each at most 128 bytes.                         |
| `status`                               | HTTP status from 100 through 599.                                       |
| `outcome`                              | `active`, `completed`, `failed`, `rejected`, `cancelled`, or `dropped`. |
| `source`                               | `active` or `durable`; omit it to merge both sources.                   |

Unknown or duplicate parameters are rejected rather than ignored.

`GET /api/logs/requests/{requestId}` returns one active or durable summary.
`requestId` must be a UUID. Detail reads are audited with fixed metadata and do
not put the request ID or response body into the audit record.

### Request events and artifacts

`GET /api/logs/requests/{requestId}/events` returns the request's durable
lifecycle projections. It accepts `limit`, `cursor`, and `sort` and defaults to
ascending order. Event objects contain bounded identifiers, timestamps, kind,
and applicable model/provider/engine, attempt, status, duration, or token
fields. They never return the stored canonical event JSON or raw error text.

`GET /api/logs/requests/{requestId}/artifacts` accepts the same page controls
and returns artifact metadata only. `contentState` is `available`,
`unavailable`, `missing`, or `corrupt`; the response also reports whether the
artifact was redacted or truncated.

`GET /api/logs/artifacts/{artifactId}` reads one artifact. It includes
`contentBase64` only when the bytes were captured through the
`redacted_artifacts` policy and pass the confined integrity checks. Missing or
corrupt files become typed metadata states instead of exposing paths or I/O
errors. Artifact reads are audited without recording bytes or identifiers.

### Proxy attempts and audit entries

`GET /api/logs/proxy` returns durable routing attempts. It accepts `limit`,
`cursor`, `sort`, `request_id`, `provider`, `engine`, and `status`. A target is
reduced to scheme, host, and port; credentials, paths, query strings, and
fragments are not returned.

`GET /api/logs/audit` returns sparse operational audit entries with `entryId`,
`occurredAt`, `source`, `code`, and optional `severity`, `contextVersion`,
`subjectKind`, `subjectId`, `remoteAddr`, `pathType`, `operationId`, `requestId`,
`reasonCode`, `outcome`, `durationMs`, and `numericSummaries`. It accepts
`limit`, `cursor`, `source`, and `severity`. Sources are `logging_service`,
`runtime`, `mesh`, `cli`, or `logs_api`; severities are `info`, `warning`, or
`error`. CLI audit entries may also contain the optional `commandSummary`.
It is a bounded parsed-command projection, never raw argv: positional values,
paths, URLs, credentials, model and plugin names, identifiers, and reasons are
redacted or omitted. It accepts at most 32 tokens and 256 characters, rejects
control text, and retains only fixed command vocabulary plus explicitly supplied
numeric or enum values in their approved option contexts. Entries may omit it
when attribution was not observed. Audit responses contain no
arbitrary detail JSON or free-form message; malformed summaries are omitted at
every REST and SSE projection boundary.

For mesh-peer audits, `subjectId` is the authenticated endpoint ID. Direct
paths can include the observed peer `remoteAddr`; relay paths omit it and never
expose the relay server as the peer. `pathType` is absent when no live path was
available, and entries can omit typed context fields when that context was not
observed. Authenticated
Skippy stage connections use `mesh_quic_inbound_accepted` for this peer/path
evidence before stage stream dispatch.

### Exact mesh audit contract

| Code                                   | Outcome                    | Reason                                                                                                                                                                                                                                                        | Numeric summaries                                                            | Duration owner                       |
| -------------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------ |
| `gossip_direct_peer_promoted`          | `promoted`                 | none                                                                                                                                                                                                                                                          | `direct_peers`                                                               | none                                 |
| `gossip_peer_removed`                  | `removed`                  | `stale_direct_and_transitive`, `heartbeat_unreachable`, `peer_down_probe_failed`, `closed_connection_no_address`, `reconnect_failed`, `recovered_gossip_failed`, `clean_shutdown`, or `tunnel_open_failed`                                                    | `direct_peers`                                                               | none                                 |
| `gossip_policy_rejected`               | `rejected`                 | `owner_attestation_required`, `owner_attestation_expired`, `owner_attestation_invalid`, `owner_attestation_node_mismatch`, `owner_revoked`, `owner_attestation_revoked`, `owner_node_revoked`, `owner_attestation_protocol_unsupported`, or `owner_untrusted` | none                                                                         | none                                 |
| `gossip_incompatible_version_rejected` | `rejected`                 | `protocol_version_unsupported`                                                                                                                                                                                                                                | `peer_gen`, `local_gen`                                                      | none                                 |
| `mesh_quic_inbound_accepted`           | `accepted` or `readmitted` | none                                                                                                                                                                                                                                                          | negotiated family `protocol_gen` (`4` for the current Skippy stage protocol) | none                                 |
| `mesh_control_connection_accepted`     | `accepted`                 | none                                                                                                                                                                                                                                                          | `protocol_gen`                                                               | none                                 |
| `mesh_control_alpn_rejected`           | `rejected`                 | `alpn_unsupported`                                                                                                                                                                                                                                            | none                                                                         | none                                 |
| `mesh_quic_handler_failed`             | `failed`                   | `capacity` or `internal`                                                                                                                                                                                                                                      | none                                                                         | QUIC handler boundary, when timed    |
| `mesh_control_handler_failed`          | `failed`                   | `capacity` or `internal`                                                                                                                                                                                                                                      | none                                                                         | control handler boundary, when timed |

`gossip_incompatible_version_rejected` is emitted when a direct peer is
rejected during admission for an incompatible version. A transitive
announcement below the local version floor is dropped without emitting this
event.

`mesh_auto_join_succeeded` and `mesh_auto_join_failed` remain generic,
context-free entries. Pre-authentication failures may be sparse because peer
identity is unavailable before authentication completes. Direct paths may
include the observed address. Relay paths use `pathType: "relay"` and omit
`remoteAddr`, never substituting the relay address. Entries may omit typed
context when it was not observed. Audit and request records remain trusted-local only, with no
OTLP export, gossip advertisement, or peer replication.

## Live SSE

`GET /api/logs/events` is a bounded Server-Sent Events stream. Send
`Accept: text/event-stream` and at least one repeated `channel` parameter:

```bash
curl -N -H 'Accept: text/event-stream' \
  "$LOG_API/api/logs/events?channel=requests&channel=operations"
```

Lifecycle channels are `requests`, `operations`, and `system`. Filters are
`request_id`, `route`, `model`, `provider`, `engine`, `outcome`, `from`, and
`to`. A field may be repeated to select any value for that field; distinct
fields combine with AND. The equivalent `filter=field:value` form is also
supported. A stream accepts at most 32 query pairs and 16 filter values.

Audit streaming is a separate mode:

```text
GET /api/logs/events?audit=true&source=runtime&severity=warning
```

Audit mode accepts `source`, `severity`, and its cursor, but not lifecycle
channels or filters.

`audit_entry` frames use the same sparse projection as `GET /api/logs/audit`,
including optional `commandSummary`. Live and replay frames enforce the same
redaction, token, control-character, and length contract as durable rows; a
malformed persisted or internal summary is omitted instead of being streamed.

Each accepted connection first receives matching entries retained after its
cursor, then live updates. Lifecycle event IDs use the v1 vector form
`v1:<requests>.<operations>.<system>`, for example `v1:42.7.3`. Audit event IDs
use `a1:<sequence>`. Reconnect with either the `cursor` query parameter or the
standard `Last-Event-ID` header. When both are present, Mesh resumes from the
newer position so reconnects do not move backwards.

The stream emits:

| SSE event      | Meaning                                                                        |
| -------------- | ------------------------------------------------------------------------------ |
| `log_event`    | A privacy-safe lifecycle projection.                                           |
| `audit_entry`  | A sparse audit projection in audit mode.                                       |
| `replay_gap`   | Requested entries have left the bounded replay window.                         |
| `stream_error` | A retained entry could not be projected safely (`invalid_event`), or durable audit reconciliation failed (`audit_reconcile_failed`). |

If a client lags but the relevant entries remain in replay, the session catches
up from that buffer. If entries have been evicted, `replay_gap` identifies the
missing sequence interval and supplies a REST recovery endpoint and a
best-effort durable cursor. Recover lifecycle history from
`GET /api/logs/requests` and then the per-request event routes; recover audit
history from `GET /api/logs/audit`. Resume SSE from the gap frame's event ID
after the REST reconciliation.

For `stream_error`, `invalid_event` means one retained entry could not be
projected safely. `audit_reconcile_failed` means durable audit reconciliation
failed. Clients preserve the `a1:` cursor, mark audit data stale, hydrate it
authoritatively through `GET /api/logs/audit`, then resume or reconnect the
audit stream. Audit recovery is fail-open: it does not fail inference or the
original request.

Each frame is capped at 16 KiB. A connection has a 64-frame handoff queue,
receives a keepalive every 15 seconds, and is disconnected when writes remain
blocked. Treat disconnect as a reconnect signal and use `Last-Event-ID`; do not
assume one long-lived socket is lossless storage.

## Export and artifact controls

`POST /api/logs/requests/export` exports only durable request summaries and
their bounded event/artifact children. It accepts the request-list filters,
except `source`; `limit` may not exceed 50. The strict body is:

```json
{
  "reason": "incident review",
  "includeArtifacts": false
}
```

The operation is capped at 50 total selected rows, two seconds, and
`logging.export_limit_bytes`. A response reports `truncated`, `retryRequired`,
`nextCursor`, and `artifactContentIncluded`. Follow `nextCursor` when present.
If `retryRequired` is true, narrow the request because a child collection could
not be represented safely by a request-level cursor.

Artifact bytes are excluded by default. `includeArtifacts: true` succeeds only
when `logging.artifact.capture_mode = "redacted_artifacts"`; otherwise the API
returns `artifact_export_forbidden`. Even when enabled, only already-redacted
bytes that fit the response budget are included. Exports use the same safe DTOs
as ordinary reads and are recorded in the audit ledger.

## Cleanup and deletion

Cleanup is an explicit preview-then-run protocol. Both stages have a two-second
cooperative bound and require a non-empty operator `reason` of at most 256
bytes.

First create a UUID operation ID and freeze a bounded durable selection with
`POST /api/logs/cleanup/preview`:

```json
{
  "operationId": "c76db69a-c7a3-4c4e-80a1-17c6f2407adb",
  "cutoffBefore": "2026-08-01T00:00:00Z",
  "requestLimit": 100,
  "source": "durable",
  "outcome": "completed",
  "reason": "retention cleanup"
}
```

`requestLimit` accepts 1 through 100. Optional selection fields are `from`,
`to`, `route`, `model`, `provider`, `engine`, and a terminal `outcome`. The
receipt records the exact scope, selection fingerprint, planned counts,
`hasMore`, artifact-deletion progress, and a fixed audit ID.

Execute exactly that preview with `POST /api/logs/cleanup/run`:

```json
{
  "operationId": "c76db69a-c7a3-4c4e-80a1-17c6f2407adb",
  "reason": "retention cleanup"
}
```

The operation ID, action, and reason are immutable. Repeating the same preview
or completed run returns its durable receipt without selecting or deleting a
second time. Reusing the ID with different intent returns
`maintenance_conflict`. When `hasMore` is true, perform another preview with a
new operation ID for the next bounded batch.

Delete one terminal request and its owned events, artifacts, and proxy rows
with `POST /api/logs/requests/{requestId}/delete`:

```json
{
  "operationId": "2d069a51-a465-4faf-b7ad-aaedce062ed2",
  "reason": "operator-requested deletion"
}
```

The per-request receipt has the same planned/executed counts, fingerprint,
audit ID, and artifact-deletion progress. The operation ID makes retries
idempotent even after the request row has been deleted. Active requests return
`request_active`; wait for a terminal outcome before deletion.

## Terminal webhooks

When webhooks are enabled, Mesh durably queues one small notification with the
terminal request transition and delivers it asynchronously. Request serving is
fail-open: webhook latency or receiver failure never delays inference.

```json
{
  "request_id": "2a3b6e32-e417-4bd6-9423-c73cda0802fd",
  "outcome": "completed",
  "status_code": 200
}
```

`outcome` is the real bounded terminal result: `completed`, `failed`,
`rejected`, `cancelled`, or `dropped`. `status_code` is omitted when no HTTP
status belongs to the terminal result. The payload never contains prompts,
completions, artifacts, paths, endpoint data, transport errors, or raw
lifecycle JSON.

The request ID, outcome, and optional status code are persisted as immutable
delivery intent, so every automatic retry, restart recovery, and manual retry
sends the same public payload. The outbox row does not retain an endpoint URL,
response body, raw error, or lifecycle-event JSON.

Any 2xx response completes delivery. Timeouts, transport errors, and non-2xx
responses retry with bounded exponential backoff and jitter until
`logging.webhook.max_attempts` is exhausted. Delivery state and in-flight
leases are durable, so a restart can reclaim an interrupted attempt without a
stale worker completing over a newer claim. Exhausted deliveries enter
dead-letter state and remain subject to
`logging.webhook.dead_letter_retention_secs`.

An operator who has a dead-letter delivery ID can request a new bounded attempt
cycle with `POST /api/logs/webhooks/{deliveryId}/retry`:

```json
{ "reason": "receiver recovered" }
```

The response outcome is `scheduled` or, for an idempotent repeat,
`already_scheduled`. A delivery outside dead-letter/manual-retry state returns
`webhook_not_retryable`. The reason is redacted and written to the local audit
ledger; endpoint and delivery details are not projected into that audit entry.

Webhook URLs must be absolute `http` or `https` URLs with a host. Credentials,
query strings, and fragments are rejected, and redirects are disabled so only
the configured endpoint receives the payload.

## Privacy and errors

Metadata logging is enabled by default, but artifact contents are not. The
default `metadata_only` policy records bounded lifecycle and routing fields.
The opt-in `redacted_artifacts` policy applies mandatory redaction before bytes
enter the confined artifact store and enforces per-artifact and aggregate
limits. A privacy failure disables artifact capture without disabling request
serving or metadata logging.

Across REST, SSE, status, export, audit, and webhook surfaces:

- Prompt and completion text is not part of lifecycle or webhook DTOs.
- Error causes become bounded event kinds or error codes, not raw messages.
- Filesystem paths and application-state roots are never returned.
- Metadata that looks like a path, URL, credential, or configured secret is
  redacted; proxy targets expose only scheme, host, and port.
- Operator read and mutation audits use fixed actions/outcomes and redacted
  reasons, not request bodies, artifact bytes, or target identifiers.

REST errors have one stable envelope:

```json
{
  "error": {
    "code": "invalid_query",
    "message": "limit must be between 1 and 100"
  }
}
```

Clients should branch on `error.code`; messages are explanatory, not a parsing
contract.

| HTTP status | Stable codes                                                                                                        |
| ----------- | ------------------------------------------------------------------------------------------------------------------- |
| 400         | `invalid_request`, `invalid_query`, `invalid_cursor`, `cursor_expired`, `invalid_id`, `invalid_webhook_delivery_id` |
| 403         | `forbidden`, `artifact_export_forbidden`                                                                            |
| 404         | `not_found`                                                                                                         |
| 405         | `method_not_allowed`                                                                                                |
| 406         | `not_acceptable`                                                                                                    |
| 409         | `maintenance_conflict`, `request_active`, `webhook_not_retryable`                                                   |
| 503         | `artifact_deletion_unavailable`, `export_timed_out`, `maintenance_cancelled`, `logging_schema_incompatible`, `logging_unavailable`, `store_unavailable` |

Methods, request bodies, query keys, duplicate parameters, identifiers, and
enum values are validated strictly. A store or audit failure does not cause
raw SQLite or filesystem detail to cross the API boundary.

## Configuration and limits

Logging is an additive config-v1 section in `~/.mesh-llm/config.toml`. Existing
files remain valid. This example shows the defaults:

```toml
[logging]
enabled = true
summary_line_limit = 2048
event_buffer_size = 10000
retention_ttl_secs = 129600
retention_max_rows = 100000
replay_capacity = 128
queue_capacity = 4096
export_limit_bytes = 5242880
cleanup_cadence_secs = 3600

[logging.artifact]
capture_mode = "metadata_only"
byte_limit_bytes = 262144
aggregate_limit_bytes = 8388608

[logging.webhook]
enabled = false
# url = "https://operator.example/log-terminal"
max_attempts = 3
timeout_secs = 15
dead_letter_retention_secs = 259200
```

`logging.application_state_root` may optionally override the runtime's private
application-state root. It must not be empty, traverse out of its scope, name a
protected system/root directory, or be world-writable when it already exists.

| Field                                        |                           Allowed range | Apply behavior |
| -------------------------------------------- | --------------------------------------: | -------------- |
| `logging.enabled`                            |                 Boolean, default `true` | Restart        |
| `logging.application_state_root`             |                       Safe private path | Restart        |
| `logging.summary_line_limit`                 |                                1–65,536 | Restart        |
| `logging.event_buffer_size`                  |                              50–100,000 | Restart        |
| `logging.retention_ttl_secs`                 |                 3,600–7,776,000 seconds | Live           |
| `logging.retention_max_rows`                 |          1–1,000,000 terminal summaries | Restart        |
| `logging.replay_capacity`                    |                         1–10,000 events | Live           |
| `logging.queue_capacity`                     |                      64–131,072 entries | Restart        |
| `logging.artifact.capture_mode`              | `metadata_only` or `redacted_artifacts` | Restart        |
| `logging.artifact.byte_limit_bytes`          |                            1 KiB–16 MiB | Restart        |
| `logging.artifact.aggregate_limit_bytes`     |                         512 KiB–500 MiB | Restart        |
| `logging.export_limit_bytes`                 |                          64 KiB–100 MiB | Restart        |
| `logging.cleanup_cadence_secs`               |                      300–86,400 seconds | Restart        |
| `logging.webhook.enabled`                    |                Boolean, default `false` | Restart        |
| `logging.webhook.url`                        |           Valid constrained HTTP(S) URL | Restart        |
| `logging.webhook.max_attempts`               |                                    1–20 | Restart        |
| `logging.webhook.timeout_secs`               |                            1–60 seconds | Restart        |
| `logging.webhook.dead_letter_retention_secs` |                 3,600–1,555,200 seconds | Restart        |

`logging.retention_ttl_secs` and `logging.replay_capacity` are the only logging
settings currently applied to a running service. Other changes are validated
immediately but take effect after restart.

## Compatibility

The logging API and config are local, additive node capabilities. They do not
change the mesh QUIC ALPN, gossip schema, routing protocol, plugin protocol, or
Skippy ABI, and log history is not replicated to peers. A newer node can join
older nodes without requiring them to understand `/api/logs/**`.

The optional `logging` object on `GET /api/status` is additive, so consumers
must continue to tolerate absent and unknown fields. SSE IDs beginning with
`v1:` and `a1:` version only the local replay cursor syntax; they are not Mesh
protocol versions. Persist cursors only for reconnect/recovery and handle
`replay_gap` instead of assuming replay retention is permanent.

### Log-store schema and recovery

The durable log store uses one complete schema version, `1`. The forward
migration registry is intentionally empty. The API reports an unknown or
incompatible store with `logging_schema_incompatible` and typed details in
`schema_version` and `supported_schema_version`.

Mesh does not automatically migrate, reset, or delete an unknown schema. The
store is left unchanged, logging metadata becomes unavailable, and inference
remains available. The database is not modified while the schema is unknown.
Recover by updating or restoring a compatible build. To
start new history instead, stop Mesh and move or back up `log_store.db`,
`log_store.db-wal`, and `log_store.db-shm` together before restarting. Do not
edit `PRAGMA user_version`.
