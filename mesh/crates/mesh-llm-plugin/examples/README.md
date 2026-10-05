# OpenAI lifecycle observer

`openai-exchange-observer` is an installable generation-3 plugin using the host's
private lifecycle service. It declares no inference endpoint, HTTP binding, or
MCP tool. Its default mode observes metadata and abstains. `--admission` enables
allow/deny responses and requires an explicit admission grant. Set
`MESH_LLM_EXEMPLAR_DENY_MODEL` in the host environment to reject that exact model.
`MESH_LLM_EXEMPLAR_DENY_SELECTED_MODEL` rejects an exact model at
`backend_selected`, before its backend dispatch. `--identity` requests the public
identity service; `--identity --delegate` also requests fixed-scope signing.
`--delegate` alone is rejected with a flag-specific error. Operator grants must
independently authorize these services. `--identity-probe` explicitly exposes
the `identity_probe` setup operation as an MCP tool, using the process's own
signing key with no HTTP binding. The identity permission flags alone do not
expose this diagnostic tool.
`MESH_LLM_EXEMPLAR_EVENT_LOG` optionally appends granted event JSON to a local
JSONL file. New event, stream-metadata, and virtual-invocation log files use
Unix mode `0600` (owner read/write, subject to the process umask). Existing
files keep their current permissions; operators must keep those files and
their parent directories private when body permissions are enabled. Other
platforms use their filesystem's normal access controls.

The manifest requires `openai_exchange.v1`. An older host fails initialization
clearly. The same host continues to accept plugins without this declaration.
Conformance fixtures use `--plugin-id <id>` for separately installed instances.
`--optional` makes the declaration optional; operator failure policy still
controls fail-closed behavior. `MESH_LLM_EXEMPLAR_FAULT=disconnect_on_response_stream`
exits the observer process when its response copy is opened. These options test host isolation
without changing the request or response entity bytes.

Build with `just build-openai-exchange-exemplar`. Package for local installation with `just package-openai-exchange-exemplar`, producing `dist/openai-exchange-observer.tar.gz`. A configured external plugin
can run the built example directly:

```toml
[[plugin]]
name = "openai-exchange-observer"
command = "/absolute/path/to/target/debug/examples/openai-exchange-observer"
args = [] # use ["--admission"] for admission mode

[plugin.openai_exchange_grant]
endpoints = ["chat_completions", "completions", "responses"]
phases = ["request_received", "backend_selected", "exchange_finished"]
metadata = true
admission = false # requires args = ["--admission"] when true
failure_policy = "best_effort" # use "required" to fail closed on admission failure
deadline_ms = 250
max_body_bytes = 1048576
max_queue_bytes = 4194304
max_in_flight = 32
```

An installed plugin does not receive these grants automatically. The exemplar `--body` mode requests all three body permissions; grant
`request_body`, `effective_request_body`, and `response_body` only after reviewing
its code and retention policy. The contract supports independent permissions,
while this required exemplar mode needs all three. Headers require a separate allowlist; credentials,
cookies, tokens, and key headers remain forbidden. A persisted owner apply refreshes grants immediately and disconnects affected
body copies. The host intersects declaration and grant permissions.

Lifecycle decisions cannot rewrite request or response bytes. Read-only observers
cannot deny a request. Required plugins must receive every declared permission at
startup, while resource ceilings use the lower declaration/operator value.
Identity services also require separate permissions and a registered signing
scope. Software identity and delegation prove key relationships, not hardware
execution attestation.

Identity diagnostics require an installed artifact, rather than the direct
`command` example above. Install the package with
`mesh-llm plugins install --archive dist/openai-exchange-observer.tar.gz --name openai-exchange-observer --version 1.0.0`.
For the configured installed plugin, use
`args = ["--identity", "--delegate", "--identity-probe"]` and add these fields to
its existing lifecycle grant:

```toml
read_identity_bundle = true
delegate_signing_key = true
signing_scopes = ["mesh.openai.exchange.evidence.sign.v1"]
max_delegation_ttl_secs = 60
```

The exposed `identity_probe` MCP operation accepts
`{"method":"ReadIdentityBundle"}` or `{"method":"DelegatePluginSigningKey"}`.
Delegation uses the process-owned key and defaults to the registered scope and
a 60-second lifetime. These calls are setup diagnostics; they do not sign a
lifecycle evidence record or prove inference execution.

Side streams negotiate a random UUID token over the authenticated control
connection. The host sends the 36 ASCII token bytes before the entity bytes.
The exemplar verifies it before hashing payload, restricts Unix socket access to
the user, and caps receipt at 16 MiB and 30 seconds. Metadata carries `kind` and
`exchange_id`; streams never rewrite bytes. Terminal SHA-256 and byte counts are
compared with independently hashed received response bytes.

Terminal diagnostics distinguish `independently verified`, `not requested`
when body mode is disabled, `unavailable` when a requested receipt is missing
or incomplete, and `mismatch` when a complete receipt differs from the host
commitment. Metadata-only observation does not independently verify response
bytes. Verification covers the local emitted entity, not remote execution.

After EOF the exemplar stores the independent receipt, then returns a bounded
newline JSON `{ "sha256": "...", "byte_count": N }` under the `sha256-v1`
receipt protocol. The host verifies it before publishing completed evidence.
Original/effective request streams retain at most 16 MiB in total for admission;
responses are hashed incrementally without retaining generated text. Set
`MESH_LLM_EXEMPLAR_DENY_TEXT` with `--body --admission` to deny a substring of the
original parsed request, including requests too large for the control envelope.
See the [normative contract](../../../docs/plugins/openai-exchange-lifecycle.md).

`--virtual-echo` optionally registers the deterministic `exchange-echo` virtual
model for installed dispatch conformance. It returns a fixed response and does
no inference. `MESH_LLM_EXEMPLAR_VIRTUAL_LOG` records its received typed invocation
only when that backend actually runs; denial tests assert the file is absent.
The default exemplar does not advertise this virtual model. Prepared pipeline
and virtual-model tests check selected-request admission after transformations
and ensure a denied subdispatch cannot fall back to an unapproved call.
