# OpenAI exchange lifecycle contract v1

An installed plugin can observe OpenAI exchanges and deny admission through the
authenticated generation-3 plugin connection. The feature capability is
`openai_exchange.v1`; the lifecycle contract version is `1`. No separate HTTP
server, MCP tool, owner private key, or unrestricted signing service is involved.
The [acceptance matrix](openai-exchange-acceptance.md) records which checks have
run. The [earlier design note](openai-exchange-lifecycle-design-note.md) is a
historical prototype, not this contract's normative definition.

External lifecycle connections must belong to the process launched by the host.
The host checks the OS peer PID on Unix sockets and Windows named pipes before
enabling this capability. An unavailable or mismatched PID fails closed.
Wrappers must `exec` the connecting plugin so it retains the launched PID.
An unverified connection cannot acquire lifecycle access through later grants
or call identity services. Host-owned in-process runners use a private duplex
connection. Ordinary plugins without lifecycle declarations keep their existing
protocol behavior.

Lifecycle callbacks and body-stream negotiation require an authenticated,
successfully initialized declaration on the connection generation receiving
them. Cached declarations from a disconnected generation cannot authorize a
replacement connection, including when grants were added after startup.

## Manifest requests and operator grants

The optional manifest `openai_exchange_hook` names a handler and requests
endpoints, phases, permissions, and resource limits. The private
`ServiceKind::OpenaiExchange` invokes that handler with JSON. A manifest cannot
grant itself access. The operator supplies `[plugin.openai_exchange_grant]` for
that configured plugin. An ordinary install with no grant receives no events,
bodies, identity bundle, or delegation authority.

Supported endpoint names are `chat_completions`, `completions`, and `responses`.
Phase names are `request_received`, `backend_selected`, and `exchange_finished`.
The first phase precedes downstream dispatch. The selected phase observes the
effective request after core defaults and transformations, immediately before
backend dispatch. The terminal phase reports the outcome, including denial,
validation failure, cancellation, timeout, backend failure, and successful end.
A request denied before selection has no selected phase.

Both sides explicitly list allowed endpoints, phases, and headers. Independent
booleans control `request_body`, `effective_request_body`, `response_body`,
`admission`, `metadata`, `read_identity_bundle`, and `delegate_signing_key`.
Effective permissions are the intersection; numeric limits use the lower bound.
The resource fields are `deadline_ms` (1–30000), `max_body_bytes` (1–16 MiB),
`max_queue_bytes` (1–64 MiB), and `max_in_flight` (1–1024). Delegation uses
`signing_scopes` and `max_delegation_ttl_secs` (1–86400 when enabled).

A successful persisted owner apply refreshes grants immediately, disconnects
copies whose grants changed, and invalidates affected delegations. Committed
applies reconcile grants even after command cancellation. Startup seeds current
persisted grants and installs the manager under the same apply serialization;
an apply that began before installation resolves the current manager after
commit. Configuration
schema leaves advertise dynamic apply. Editing a file alone requires the normal
configuration reload or restart. Grants never originate from plugin config
schema or a peer manifest.

Removing a grant or a subscription deactivates it. If a still-subscribed required
declaration loses body or metadata permissions, admission fails closed without
sending a callback or side-stream payload. Operator status reports
`permissions_unavailable` until its declaration and grant agree again.
An explicit required operator subscription also fails closed when its process
failed to start, its manifest is unavailable, or it lacks a lifecycle handler.
The host retains that grant independently of the live plugin map. Adding a
required plugin through owner apply therefore blocks subscribed endpoints until
the plugin is available after restart, or the operator removes its grant.
Best-effort unavailable plugins mark evidence unavailable and preserve inference.

A required unsupported lifecycle declaration fails initialization clearly. An
optional unsupported declaration remains inactive. New plugins require the
capability when their manifest marks the contract required. Older generation-3
manifests omit the additive field and retain their existing behavior. Invalid
names, credentials in header lists, inconsistent signing permissions, and out of
range limits are rejected during validation.

## Events, decisions, and observation points

Events include `exchange_id`, `phase`, `endpoint`, `observation_point`, method,
path, optional model, sanitized headers, and permissioned parsed body data.
They also carry original/effective request byte commitments and, at termination,
a response byte commitment. A local observation ID distinguishes observations
of the same exchange. IDs asserted by a peer provide correlation; they are not
proof of a global execution claim.

Rust authors can call `OpenAiExchangeEvent::parse(&event)` inside the existing
raw JSON callback. The typed view preserves `observation_id` and the optional
boolean fields `evidence_complete` and `observer_evidence_complete`. An absent
field means unknown, rather than false. The old `evidence_completeness` string
field is retained for SDK source compatibility but is not emitted by v1 hosts.
The raw payload remains available for additive fields outside the typed view.

Final client cancellation, timeout, or transport failure takes precedence over
the admission outcome. The optional `admission_denied` and
`required_admission_failure` booleans retain the admission result separately,
including when delivery of a 403 or 503 response fails. Observer-copy loss
changes evidence completeness without changing the execution outcome.

Terminal events preserve a bounded history of up to 16 prepared dispatches,
their commitments, reported usage when available, and elapsed lifecycle time.
Truncating the history marks evidence incomplete. Usage is reported data, never
an estimate or proof of execution.

Observation points describe the host's actual seam: gateway ingress, serving
host ingress, backend dispatch, or client egress. A gateway can attest to what
it received, selected, forwarded, and emitted. It cannot infer that a remote
host performed inference merely because it forwarded a request. Typed request
serialization must be labelled as such; it is not the original HTTP entity.

The strict decision schema accepts `abstain`, `allow`, or `deny`, an optional
reason of at most 1024 bytes, bounded annotations, and bounded response headers.
There is no request rewrite field. Allow and abstain do not override core
validation or another policy's denial. Admission permission is required for a
denial before dispatch. A terminal response cannot deny completed inference.

Host fan-out shares one absolute deadline across plugins. Deny wins over allow;
a required unavailable policy produces a stable OpenAI 503 hook error, while a
policy denial produces 403. Best-effort observer faults mark evidence incomplete
without converting successful inference into a backend failure. Circuit and
in-flight limits bound repeated failures. Backend deadlines and hook deadlines
are different causes and must remain distinguishable. Lifecycle RPCs send once;
deadline expiry or cancellation removes that call's pending response without
retrying or restarting the plugin.

Metadata permission is required for annotations or response headers. Annotation
keys are host-namespaced by plugin, at most 128 bytes, values at most 1024 bytes,
and the aggregate at most 4096 bytes. At most 16 response headers are accepted;
names are at most 128 bytes, values at most 1024 bytes, ASCII without CR/LF, and
must match the plugin's `x-plugin-<hex-encoded-plugin-id>-` namespace plus the explicit
header grant. Names containing auth, cookie, session, jwt, token, secret, or key
are excluded even if requested. Metadata cannot replace protocol
or transport headers.
Each recipient receives only its own annotation namespace; one plugin cannot
use annotations to disclose a granted body to another plugin.
The header namespace encodes the exact plugin ID's UTF-8 bytes as lowercase
hexadecimal, so punctuation or case differences cannot collide. Metadata merges
deduplicate headers and retain at most 16 names in lexical order across plugins
and dispatches. Each author's accumulated annotations remain within 4096 bytes;
discarding excess metadata marks evidence incomplete without breaking HTTP
emission.

## Exact bytes and bounded side streams

[Wire byte commitments](openai-wire-byte-transcripts.md) define exact SHA-256
coverage and published test vectors. Original request bytes precede parsing;
effective request bytes are the entity sent to the chosen backend. Response
bytes are observed at final emission. JSON whitespace and SSE fields, newlines,
event delimiters, and `[DONE]` are included. HTTP transfer chunk framing,
headers, and trailers are excluded. Transport fragmentation does not alter the
commitment. A digest does not prove delivery to the peer.

`emission_boundary` is `socket_write_accept` for raw ingress and
`http_body_poll` for the typed frontend. The latter commits bytes handed to the
HTTP transport; it does not establish a socket-accepted prefix or client
receipt. Gateway egress remains the final boundary for routed client traffic.

Small granted parsed bodies may accompany the control event up to 64 KiB.
Exact bodies travel through the existing authenticated `OpenStream` side
channel, with a UUID stream ID, exchange correlation metadata, raw-byte mode,
and `bidirectional=true`. The purpose metadata identifies original request,
effective request, or response. The receiver authenticates the supplied token
before consuming payload bytes. `receipt_protocol="sha256-v1"` requires one
bounded newline JSON receipt `{ "sha256": "<lowercase hex>", "byte_count": N }`
after EOF. The receiver stores its independent digest before acknowledging.
The host verifies the receipt, which prevents terminal control delivery from
racing ahead of receipt storage.

Request admission may await a bounded exact body copy. Response observation is
an ordered, bounded asynchronous copy; queue pressure never stalls inference
streaming. Overflow, disconnect, revocation, timeout, and cancellation abandon
the copy and explicitly mark it incomplete. A response observer hashes
incrementally; it need not buffer the response. Prefix commitments preserve the
observed digest and length when termination is abnormal.

Execution outcome and evidence completeness are separate facts. Successful
inference can have incomplete observer evidence. Missing optional identity is
expected absence, not an internal hook fault. Client cancellation stops local
work and attempts downstream cancellation; it does not establish that a remote
backend acknowledged an abort. Terminal evidence reports the available local
observation rather than inventing that assurance.

## Public software identity and scoped delegation

Authenticated private RPC methods `ReadIdentityBundle` and
`DelegatePluginSigningKey` require their distinct host grants. The public bundle
contains node endpoint identity, signed ownership evidence and status, owner
public signing key, host version, optional release attestation and verification
summary, installed plugin artifact metadata/hash, and current delegation state.
It contains no private keys, unlock material, or arbitrary signing endpoint.
Artifact status is separate from node ownership status. A changed or missing
executable disables signing and revokes existing delegations; an authorized
bundle reader can still inspect its actual digest/status and revoked IDs.
Owner unlocking and delegation renewal occur during setup, outside inference
callbacks; callbacks cannot trigger interactive key access or renewal.
While a plugin has an active lifecycle callback, its authenticated
`DelegatePluginSigningKey` RPC is rejected before artifact hashing and checked
again before issuance. Another plugin's active callback does not block its
setup RPCs. `ReadIdentityBundle` remains independently permissioned and uses
the bounded asynchronous service path.

A delegation request contains only `lifetime_ms`, `signing_public_key`, and
`scope`; unknown fields are rejected. The only registered scope is
`mesh.openai.exchange.evidence.sign.v1`. The host binds claims to its installed
plugin ID, executable SHA-256 captured at launch and checked against the current
file, registered Ed25519 key, current verified owner/node certificate, and expiry.
The host chooses a UUID delegation ID. Requests require at least one second of
lifetime. Matching active certificates are reused until their remaining lifetime
enters the smaller of 30 seconds or half the requested lifetime. Reuse preserves
the requested lifetime bound and all current identity/artifact/key/scope bindings.
Fresh owner signatures, including key rotation, are limited to one per second
and five per rolling minute per plugin; registry capacity is checked before
signing. This limit is independent of certificate expiry and pruning.
All identity RPCs are admitted before filesystem work, with one active call
and thirty calls per rolling minute per plugin. Artifact inspection accepts
regular files up to 256 MiB. Hashing has a size-scaled deadline capped at
17 seconds, following a one-second metadata deadline. The whole service has
a 20-second execution deadline; registry admission waits at most 250 ms.
A transient inspection timeout requires retry and preserves existing
delegations. Identity work runs separately from the
serial plugin mesh-event forwarder.
Maximum lifetime is the smaller of the
operator grant, 24 hours, and the ownership certificate's remaining lifetime.
Plugins cannot submit an arbitrary owner ID, certificate, artifact claim, or
message to sign.

Version-1 delegation signatures cover the ASCII domain prefix
`mesh-llm-plugin-signing-delegation-v1:` followed by UTF-8 JSON in the declared
Rust DTO field order. JSON string escaping follows `serde_json`; this is a fixed
schema encoding, not general JSON canonicalization. Keys, digests, and
signatures use lowercase hexadecimal. Owner IDs are lowercase hexadecimal
SHA-256 of the 32 raw Ed25519 owner verification-key bytes. Verifiers must check the domain, version,
registered scope, owner signature, ownership certificate, expected node,
expected installed artifact/key, validity window, and applicable revocations.
Trust expectations come from the verifier's policy, not the supplied claim.

`missing`, `unverified`, `verified`, `expired`, `revoked`, and `invalid` remain
distinct. Local revocation covers changed grants, keys/artifacts, owner/node or
certificate bindings. Revocation tombstones are scoped to the current plugin
manager; restart continuity requires durable inputs from the verifier policy.
Offline verification requires current revocation inputs from its trust policy.
The contract does not promise a universal revocation
feed. Release provenance, software ownership, and execution evidence are
independent assurance axes. Owner delegation does not establish hardware
attestation, remote execution, or confidential inference; hardware delegation
is not a supported scope in v1.

## Exemplar and privacy

The [Rust exemplar](../../crates/mesh-llm-plugin/examples/README.md) uses the real
plugin server, authenticated side streams, and host-private lifecycle service.
`just package-openai-exchange-exemplar` creates an installable archive. Observe
mode requests metadata only; admission and body modes are explicit options.
The exemplar's `--identity-probe` flag separately exposes an MCP setup diagnostic;
the lifecycle callbacks themselves remain private and are never MCP tools.
Run `just test-openai-exchange-conformance` to package the exemplar and execute
the installed HTTP, QUIC, prepared-dispatch, typed-router, and identity tests.
These tests use `--ignored` in ordinary unit runs and fail if their package is
missing.

Body access exposes prompts and generated content to a separately installed
process. Grant only the phases and body variants that its policy requires.
An event-log option in the exemplar writes the permissioned event content to a
local file; operators control that path and retention. Digests are stable
content identifiers and can permit guessing low-entropy content; they provide
integrity, not confidentiality or anonymization. Public identity is software
provenance evidence, not a privacy guarantee.
