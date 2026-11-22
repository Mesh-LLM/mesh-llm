# mesh-llm-plugin

`mesh-llm-plugin` owns the plugin author API and shared plugin wire protocol
types used by host-side plugin runtimes and external plugins.

This crate includes:

- the plugin protobuf schema at `proto/plugin.proto`
- generated plugin protocol types exposed through `mesh_llm_plugin::proto`
- typed helpers for plugin manifests, operations, resources, prompts, tasks,
  mesh events, HTTP bindings, MCP projections, host-projected web UI bundles,
  and side-stream I/O

Plugins that declare web UI use the `web_ui`, `web_ui_bundle`, `web_ui_page`,
and `web_ui_config_section` manifest builders. Declarative runtimes can place
config schemas under `config: [...]` and UI declarations under `web_ui: [...]`
in `plugin!`, alongside their real handlers. Macro fields remain in declaration
order: `metadata`, optional `startup_policy`, `provides`, `config`, `web_ui`,
`mesh`, `events`, `mcp`, `http`, `inference`, then lifecycle hooks. The v1
contract is local, same-origin, trusted, and projection-only; see
[`mesh/docs/plugins/README.md`](../../docs/plugins/README.md) and its maintained
[`web UI exemplar`](../../docs/plugins/exemplars/web-ui/README.md).

Host-only orchestration, process lifecycle, plugin config loading, MCP bridge
hosting, and built-in plugin wiring should remain in the host crate for now and
move later to a dedicated plugin-host crate. Keep this crate focused on the
stable API and protocol surface that plugin authors can depend on.

## Invocation callbacks

Hosts and SDK runtimes negotiate `invocation_callbacks.v1` during initialization.
The host assigns an optional `invocation_id` to forwarded RPC and service calls.
`PluginContext::request_host` and `notify_host` echo it automatically, including
from a context retained with `owned()`. Keep that invocation context when spawning
work; a fresh context has no callback owner.

Capable plugins can serve concurrent MCP calls. The host routes each callback to
its exact originating session and rejects missing, unknown, or finished IDs.
Callbacks already running may finish, but cannot keep a completed invocation open
for new callbacks. Legacy plugins retain per-plugin serialization. The new proto3
fields are optional, and the protocol version and existing field numbers stay
unchanged.

## Peer evidence verification

A plugin declares `peer_evidence::peer_evidence_verifier(namespace, handler)` alongside its operation handler. Contract version 1 uses `PeerEvidenceVerificationRequest` and `PeerEvidenceVerificationResult` over the existing operation transport. Hosts reject unsupported versions, empty or oversized namespaces/handlers and undeclared handlers, and run at most one verifier. A host treats a provider that published no evidence as ineligible without calling the verifier. Plugins that require this contract must check the `peer_evidence.v1` host capability during initialization.

Publish with `context.request_host("PublishPeerEvidence", ...)`: `{"evidence": ...}` publishes a JSON reference of at most 1,024 encoded bytes under the declared namespace, and `{"evidence": null}` stops publishing it. A request can fail before the node is ready for it, for example with "plugin manager unavailable" during startup, and the host does not queue it, so send it again until it succeeds. Withdrawing does not revoke evidence: a peer that has not yet received the record without it may rely on the old evidence until its verdict expires, so a deadline should not outlast what backs the evidence. A peer that receives that record treats the provider as ineligible at once. Publishing uses the identity services, and a verifier decides which providers the host routes to in its own mesh (additional mesh memberships do not apply it), so a plugin that publishes or verifies must run as an OS-authenticated process, and a restarted verifier must declare the same namespace and handler. The verifier receives the provider's authenticated node ID, the local mesh ID (empty until known) and the provider's evidence from its signed record; acceptance is plugin policy. Return `valid_until` in Unix seconds; a verdict that depends on no external state may use a far deadline, since a verifier restart discards all verdicts. A rejection may give a `reason`, which the host logs. It applies at once, and one without a future `valid_until` is checked again after a few seconds. Handler errors, malformed results and timeouts leave a provider that is not yet accepted ineligible; when renewing an acceptance, they keep it until it expires. Either way the host asks again after a few seconds.
