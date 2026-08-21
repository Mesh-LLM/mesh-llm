# Operator request logging

The **Logs** console is the local operator surface for request lifecycle
records. It is separate from the public mesh, normal runtime status events,
and OTLP telemetry. Open the Logs tab in the embedded console to inspect the
request ledger, then select a row to open its detail view.

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

## Local access model

The log API is a trusted-local management surface. Every `/api/logs/**` route
requires a loopback caller and a trusted local Host and Origin when those
headers are present. It is not advertised through mesh gossip and must not be
put behind a public reverse proxy. Use the local embedded console rather than
sharing its log routes with other users or nodes.

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

| Owner | Captured now | Displayed in Logs |
|---|---|---|
| CLI dispatch | Parsed command start and terminal result, bounded parse failure, and normalized command family | Yes, with command family and outcome |
| Host runtime | Startup, ready, shutdown start, and authoritative shutdown completion | Yes, with runtime subject context where available |
| Model lifecycle | Resolve/load start, ready, load failure, unload start, unload completion/failure, and unexpected model exit at existing Rust-owned boundaries | Yes, with model and runtime-instance correlation |
| Configuration and diagnostics | Existing apply and diagnostics outcomes | Yes |
| Discovery, mesh, and local serving | Existing join/discovery, connection, readiness, and availability outcomes already owned by Rust | Yes |
| OpenAI-compatible inference | Admission, route and attempts, stream boundaries, usage, terminal outcome, and optional request/response artifacts | Yes; payload bodies require `redacted_artifacts` and an explicit operator read |
| Logging operations | Queue/persistence failures, cleanup/delete/export/retry operations, audited reads, and local authorization rejection | Yes; routine management records remain filterable |

The request ledger is canonical for inference lifecycle records. Operational
logging does not emit duplicate audit rows for those same request events.
Streaming response artifacts are assembled only in `redacted_artifacts` mode,
bounded by the configured artifact limit while frames arrive, and redacted
before persistence. An incomplete stream or an over-limit stream is recorded
as explicitly unavailable rather than as a partial payload.

Lower-level native runtime/model phases, stage and topology lifecycle, session
and capacity events, tokenization/prefill/decode progress, KV/cache pressure,
device degradation, and reducer-owned availability remain deferred to the
future event pipeline in [`.omo/specs/event-system.md`](../.omo/specs/event-system.md).
Do not change the mesh protocol, Skippy ABI, native callbacks, or low-level
runtime ownership solely to add logging hooks before that system is implemented.

## Troubleshooting and rollback

- **The Logs tab says unsupported.** The connected host predates the local log
  API or has it disabled. Upgrade or use a host that exposes the service; do
  not point the console at status/runtime SSE as a workaround.
- **The Logs tab reports a database version mismatch.** Compare the database
  and runtime schema versions shown in the warning. Update MeshLLM when the
  database is newer, or restore/update to a build that recognizes an older
  schema, then restart the node. The incompatible store is left unchanged and
  inference remains available; repeated retries and manual database edits are
  not recovery steps.
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
- **Rolling back a host.** Export the permitted, bounded operator view before
  changing versions. An older binary may show the unsupported state and must
  not be assumed to understand a newer local log store. Follow the release
  rollback procedure for the binary and its application state; do not manually
  copy or edit log database or artifact files.

For the wire and UI contracts behind this guide, see the logging section of
[the architecture notes](design/DESIGN.md#operator-request-logging) and the
[logging test playbook](design/TESTING.md#logging-workflow-certification).
