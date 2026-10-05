# Operator request logging

The **Logs** console is the local operator surface for request lifecycle
records. It is separate from the public mesh, normal runtime status events,
and OTLP telemetry. Open the Logs tab in the embedded console to inspect the
request ledger, then select a row to open its detail view.

Ledger search accepts either a full endpoint ID or the abbreviated form shown
in the console, such as `9f0c…bb04`, for request caller identities and mesh-peer
audit subjects.

## What is retained

The logging service keeps compact request metadata in the node-local
application-state root and keeps optional artifacts in a separate local
artifact store. The application-state root can be selected with
`logging.application_state_root`; if it is not set, mesh-llm uses its normal
local state location. The current working directory is not a log-store lookup
location.

Request rows progress from `active` to exactly one terminal outcome. The
canonical lifecycle includes completed, failed, rejected, cancelled, and
dropped outcomes; the ledger's normal success/failure workflow visibly covers
completed and failed rows. A terminal row and its detail record are available
as soon as the terminal outcome is recorded. The details view loads events,
routing attempts, and artifact metadata only when their tabs are opened.
Opening **Payloads** does not download retained bodies; each available request
or response body requires its own explicit **View payload** or **Load payload**
operator action.

Artifact content is deliberately conservative:

- `metadata_only` is the default capture mode. It records no request or
  response body.
- `redacted_artifacts` is an explicit opt-in. Captured content is subject to
  redaction and per-artifact and aggregate byte limits.
- An artifact can be unavailable, missing, or corrupt. Those are operator
  states, not invitations to reconstruct content from another log source.

The console renders diagnostic text as text. It does not render log content as
HTML, and it never automatically downloads optional artifact content.

### Caller and mesh-peer context

Request summaries can include `callerEndpointId`, `callerAddr`, and
`callerPathType`. Mesh operational audits can include `subjectKind`,
`subjectId`, `remoteAddr`, and `pathType`, together with bounded outcome,
reason, duration, and numeric-summary fields appropriate to the audit code.
These fields are optional because local HTTP callers have no mesh endpoint ID,
not every boundary has a live connection, and attribution is only recorded when
the relevant caller or peer context was observed.

The request caller vocabulary is `local_http`, `remote_quic_http`, and
`relay`. Skippy stage connections and streams are transport work without a
top-level request ID, so they never fabricate request-summary rows. Staged QUIC
is represented as authenticated QUIC audit evidence, not as a synthetic request
row. Its authenticated peer and selected direct/relay path appear on the local
`mesh_quic_inbound_accepted` audit entry.

An authenticated QUIC endpoint identity can be known even when the selected
path was not observed or its value was not recognized. In that case the request
summary keeps `callerEndpointId` and omits both `callerAddr` and
`callerPathType`. It does not infer a direct or relay path. Endpoint-only caller
identity is not a fourth caller path type.

For a selected direct path, the ledger records `pathType: "direct"` and the
observed socket address when it is available. For a relay path, it records
`pathType: "relay"` and omits the address; it never substitutes a relay server
address for the peer. When the relevant context was not observed, these fields
remain absent rather than fabricating attribution.

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
include the observed address. Relay paths use `pathType: "relay"` and omit the
address, never substituting the relay address. Records may omit typed context
when it was not observed. Audit and request records are trusted-local only, with no OTLP export,
gossip advertisement, or peer replication.

## Local access model

The log API is a trusted-local management surface. Every `/api/logs/**` route
requires a loopback caller and a trusted local Host and Origin when those
headers are present. It is not advertised through mesh gossip and must not be
put behind a public reverse proxy. Use the local embedded console rather than
sharing its log routes with other users or nodes.

Audit and request records remain node-local. No OTLP exporter or telemetry
survey reads them, and no gossip or replication path sends them to peers. The
bounded export API is also trusted-local and requires the same access checks.

### Invite tokens are public connection information

Mesh invite tokens are intentionally shareable connection descriptors, not
private credentials or secrets. They may appear in normal CLI/TUI/JSON output,
diagnostic logs, and test evidence and do not require redaction merely because
they are invite tokens. Do not classify or report their presence as a logging
privacy defect.

This exception applies only to mesh invite tokens. Owner keys, API keys,
authorization headers, owner-control endpoint tokens, signed private URLs, and
other authentication material remain secrets and must follow the normal
redaction and evidence-handling rules.

An older host can lack this API. The console treats a missing or explicitly
unsupported ledger endpoint as **unsupported** and asks the operator to
upgrade the host; it does not substitute the older status or runtime event
streams. Unsupported capability is inert: the page does not open the log
stream, hydrate the ledger, or schedule polling timers.

## Live updates and recovery

The ledger first hydrates from the request listing, then opens the dedicated
`/api/logs/events` Server-Sent Events stream. This is independent from the
existing status and runtime event streams.

The stream subscribes to the request and operation channels and uses standard
SSE event IDs for browser reconnects. Each subscription repeats the
backend-supported ledger filters: `from`, `to`, `model`, `provider`, `engine`,
`route`, and `outcome`. `source` remains REST-only: changing it reopens the
stream and performs an authoritative ledger hydration, but it is never sent as
an SSE filter. When a view reopens, the last received replay cursor is reused.
Duplicate event IDs, request IDs, and channel sequence values do not create
duplicate ledger rows.

If the host reports a replay gap, including a gap whose recovery cursor is
omitted or explicitly `null`, the console performs an authoritative ledger
refresh. If the dedicated stream cannot stay connected, the console visibly
enters reconnecting and then bounded polling mode until a stream connection is
available again. A stale or polling indication means the REST ledger remains
the authority; it does not mean a request has failed.

SSE `stream_error` uses two stable codes. `invalid_event` means one retained
entry could not be projected safely. `audit_reconcile_failed` means durable
audit reconciliation failed. In the audit stream, clients preserve the `a1:`
cursor, mark audit data stale, hydrate authoritatively through
`GET /api/logs/audit`, then resume or reconnect. This recovery is fail-open,
so it does not fail inference or the original request.

## Retention and maintenance

Logging settings are available in the production Configuration page. The two settings with a
dynamic application contract are `logging.retention_ttl_secs` and
`logging.replay_capacity`; other logging settings show their restart
requirement in the schema-driven configuration UI. Defaults include a 36-hour
retention TTL, a terminal-summary cap, and bounded persistence, replay, export,
and artifact budgets. Check the configuration UI for the constraints accepted
by the running host instead of copying values between hosts.

The durable ledger shown by the Logs page is distinct from the rotating
`[logging.audit]` file sink and the CLI `--audit-log-path`, `--audit-log-format`,
and `--audit-log-level` flags. File-sink controls do not turn ledger capture on
or off, and ledger retention/maintenance does not edit audit files.

Use the console's operations deliberately:

- **Export view** creates a bounded metadata-only snapshot from the selected
  durable ledger scope. The current console control never loads or includes
  retained artifact bodies. An available artifact can be downloaded only
  through its explicit **Download redacted artifact** control, and only when
  its metadata says it is redacted; unavailable, missing, or corrupt content
  remains unavailable.
- **Scoped cleanup** always starts with a server preview. Review the cutoff and
  bounded request scope, supply a meaningful audit reason, then confirm the
  same operation. A `completed` or `partial` receipt automatically refreshes
  the active ledger. When a partial receipt retains failed artifact-file
  deletion work, **Retry cleanup** reuses the frozen operation ID and audit
  reason. Previews and failed runs do not refresh the ledger.
- **Delete terminal request** applies only to the selected durable terminal
  row and also requires an audit reason. A `completed` or `partial` receipt
  likewise refreshes the active ledger; a partial receipt with failed
  artifact-file deletion work offers **Retry deletion** with the frozen
  operation ID and audit reason.
- **Retry dead-letter delivery** accepts only a manually entered, validated
  delivery ID plus a meaningful audit reason. It does not derive a delivery
  context from request details or reveal a webhook destination.

For investigations, export the smallest metadata-only scope that answers the
question. Do not add prompts, completions, credentials, artifact data, or
operator identifiers to incident tickets by default.

## CLI and terminal output

`mesh-llm --help-advanced` documents the local logging configuration keys,
capture modes, retention, and local-store precedence. Canonical lifecycle
events are emitted through the production `OutputEvent` presentation path:
`--log-format json` writes one JSON object per stdout line, while pretty and
TUI presentation stays on stderr. The projection retains only bounded local
`request_id`, `event_id`, replay `channel` and `sequence`, terminal `outcome`,
HTTP `status`, `duration`, and numeric `tokens` when present. It excludes
prompts, completions, artifact bodies, credentials, URLs, and free-form
payload/error detail. Fatal startup projection redacts credentials and private
paths selectively, retaining the surrounding diagnostic category. These local correlation IDs do not cross mesh/network or
OTLP telemetry boundaries, and the CLI stream remains a process-observation
projection rather than a replacement for the trusted-local ledger.

Use the console for investigation and the CLI stream for process observation.
They are bounded projections of the same lifecycle, but the console remains
the authority for details, artifacts, replay, retention, and audited
operations.

## Operational event coverage

The operational ledger records events only where an existing Rust owner has an
authoritative transition. The following matrix is the current support contract;
it is intentionally narrower than every line that may appear in terminal or
native debug output.

| Owner                              | Captured now                                                                                                                                  | Displayed in Logs                                                              |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| CLI dispatch                       | Parsed command start and terminal result, bounded parse failure, and normalized command family                                                | Yes, with command family and outcome                                           |
| Host runtime                       | Startup, ready, shutdown start, and authoritative shutdown completion                                                                         | Yes, with runtime subject context where available                              |
| Model lifecycle                    | Resolve/load start, ready, load failure, unload start, unload completion/failure, and unexpected model exit at existing Rust-owned boundaries | Yes, with model and runtime-instance correlation                               |
| Configuration and diagnostics      | Existing apply and diagnostics outcomes                                                                                                       | Yes                                                                            |
| Discovery, mesh, and local serving | Existing join/discovery, connection, readiness, and availability outcomes already owned by Rust                                               | Yes                                                                            |
| OpenAI-compatible inference        | Admission, route and attempts, stream boundaries, usage, terminal outcome, and optional request/response artifacts                            | Yes; payload bodies require `redacted_artifacts` and an explicit operator read |
| Logging operations                 | Queue/persistence failures, cleanup/delete/export/retry operations, audited reads, and local authorization rejection                          | Yes; routine management records remain filterable                              |

The request ledger is canonical for inference lifecycle records. Operational
logging does not emit duplicate audit rows for those same request events.

CLI dispatch audits may include the optional `commandSummary` field. It is a
bounded projection of the parsed command, not raw argv: positional values,
paths, URLs, credentials, model and plugin names, identifiers, and reasons are
represented by `[REDACTED]` or omitted. The summary is limited to 32 tokens and
256 characters, rejects control text, and preserves only the fixed command
vocabulary plus explicitly supplied numeric or enum values in their approved
option contexts. Rows without this field remain valid when attribution was not
observed. The same contract is enforced when records
enter host context, durable storage, REST, live SSE, and replay SSE; malformed
internal or persisted summaries are omitted rather than exposed.
Streaming response artifacts are assembled only in `redacted_artifacts` mode,
bounded by the configured artifact limit while frames arrive, and redacted
before persistence. An incomplete stream or an over-limit stream is recorded
as explicitly unavailable rather than as a partial payload.

Lower-level native runtime/model phases, stage and topology lifecycle, session
and capacity events, tokenization/prefill/decode progress, KV/cache pressure,
device degradation, and reducer-owned availability remain deferred to the
future event pipeline.
Do not change the mesh protocol, Skippy ABI, native callbacks, or low-level
runtime ownership solely to add logging hooks before that system is implemented.

## Troubleshooting and rollback

- **The Logs tab says unsupported.** The connected host predates the local log
  API or has it disabled. Upgrade or use a host that exposes the service; do
  not point the console at status/runtime SSE as a workaround.
- **The Logs tab reports `logging_schema_incompatible`.** The warning includes
  typed `schema_version` and `supported_schema_version` details. Unknown or
  incompatible schemas are left unchanged. Logging metadata is unavailable,
  but inference remains available. Update or restore a compatible build. If
  you intentionally start new history, stop the node and move or back up
  `log_store.db`, `log_store.db-wal`, and `log_store.db-shm` together, then
  restart. Do not edit `PRAGMA user_version`, retry until it changes, or
  automatically migrate, reset, or delete an unknown schema.
- **The ledger says reconnecting, polling, gap, or stale.** Check local host
  availability first. The page will hydrate from the request listing after a
  replay gap and uses bounded polling only while the dedicated stream is
  unavailable.
- **An artifact is redacted, missing, corrupt, or unavailable.** Treat that
  state as final for the displayed record. Capture settings apply to future
  records; they do not retroactively recover content.
- **A maintenance operation is rejected.** Reopen the operation, request a
  fresh preview when required, review the scoped count, and provide a valid
  audit reason. Never bypass the preview by editing local log-store files.
- **Rolling back a host.** Restore or update to a compatible build. If the
  whole application state is moved, stop the node first and move or back up
  `log_store.db`, `log_store.db-wal`, and `log_store.db-shm` together. Do not
  edit individual database files or `PRAGMA user_version`.

For the wire and UI contracts behind this guide, see the logging section of
[the architecture notes](design/DESIGN.md#operator-request-logging) and the
[logging test playbook](design/TESTING.md#logging-workflow-certification).
