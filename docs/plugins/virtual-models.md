# Virtual Model Plugins

Status: design contract for the first implementation slice.

A virtual model is an OpenAI-compatible model advertised by `mesh-llm` whose
response is orchestrated by a plugin instead of proxied to one inference
endpoint. Examples include a mixture-of-agents committee, an agent loop, or a
router that selects a concrete model per request.

This contract extends the process plugin protocol. It does not extend the
native serving plugin ABI: native serving plugins participate inside one model
decode, while virtual-model plugins own request-level orchestration.

## Ownership Boundary

The host owns:

- OpenAI HTTP ingress and `/v1/models` projection;
- model-id collision checks and virtual-model registry state;
- admission, deadlines, cancellation, authentication, and usage accounting;
- concrete model discovery and local, mesh, or provider dispatch;
- response framing and client stream lifetime.

The plugin owns:

- orchestration policy and any plugin-local session state;
- choosing concrete models from the host-provided candidate snapshot;
- constructing nested inference requests;
- combining nested results into the virtual model response.

The plugin must not open its own public OpenAI endpoint. It uses the existing
control connection for small calls and negotiated side streams for streaming
payloads.

## Request Flow

```text
client
  -> OpenAI ingress
  -> virtual-model registry
  -> host invokes plugin handler
       -> plugin asks host to infer with a concrete model
            -> local model, mesh peer, or external provider
       <- normalized inference result
  <- plugin result
  <- host-framed OpenAI response
```

The first implementation is buffered chat completions. Streaming is additive:
the host and plugin negotiate an event stream correlated to the same request.

## Manifest

Add a distinct virtual-model declaration rather than overloading
`EndpointManifest`. An endpoint says where an external server lives; a virtual
model says which plugin handler owns an advertised model id.

```proto
message PluginManifest {
  // Existing fields remain unchanged.
  repeated VirtualModelManifest virtual_models = 13;
}

message VirtualModelManifest {
  string model_id = 1;
  string handler = 2;
  repeated string input_modalities = 3;
  repeated string output_modalities = 4;
  bool supports_tools = 5;
  bool supports_streaming = 6;
  bool requires_candidates = 7;
}
```

V1 validation rules:

- `model_id` and `handler` are non-empty;
- a plugin may declare each model id once;
- model ids may not collide with a concrete, built-in, or another plugin's
  virtual model;
- a model is advertised only while its plugin is healthy;
- a model with `requires_candidates` is advertised only while at least one
  concrete model is reachable;
- undeclared handlers cannot be invoked.

## Host-to-Plugin Invocation

The host invokes a new typed service kind on the existing request/response
control channel:

```proto
message InvokeVirtualModelRequest {
  string model_id = 1;
  string handler = 2;
  string request_json = 3;
  string candidate_snapshot_json = 4;
  optional uint64 deadline_unix_ms = 5;
}

message InvokeVirtualModelResponse {
  string response_json = 1;
  bool is_error = 2;
}
```

`request_json` is a normalized OpenAI request with the requested virtual model
preserved. The host assigns a correlation id in the envelope. Propagating an
explicit cancellation notification for that id, including cancellation of all
nested inference when the client disconnects, remains a requirement for the
later cancellation slice. The current buffered bridge bounds nested calls with
request timeouts but does not yet propagate client-disconnect cancellation.

## Plugin-to-Host Inference

Plugin-originated calls use the reserved request-id range already used by
`PluginContext::open_mesh_stream`. The implemented Rust request carries a
concrete model id, an OpenAI-compatible JSON body, and an optional timeout:

```rust
HostInferenceRequest {
    model_id: String,
    request: serde_json::Value,
    timeout_ms: Option<u64>,
}
```

Register the virtual model in the manifest, bind the declared handler on a
`VirtualModelRouter`, and pass one `HostInferenceRequest` to
`PluginContext::infer`:

```rust
fn plugin(metadata: PluginMetadata) -> SimplePlugin {
    let manifest = plugin_manifest![
        virtual_model("mesh-agent", "chat").supports_tools(true)
    ];
    let mut router = VirtualModelRouter::new();
    router.add_raw(
        operation_with_schema("chat", "Run an agent turn", serde_json::Map::new()),
        |request, context| {
            let context = context.owned();
            Box::pin(async move {
                let invocation: VirtualModelInvocation = request.arguments()?;
                structured_tool_result(handle_mesh_agent(invocation, context).await)
            })
        },
    );
    SimplePlugin::new(metadata)
        .with_manifest(manifest)
        .with_virtual_model_router(router)
}

async fn handle_mesh_agent(
    invocation: VirtualModelInvocation,
    context: PluginContext<'static>,
) -> anyhow::Result<VirtualModelResponse> {
    let candidate = choose_route(&invocation.candidates)?;
    let response = context
        .infer(HostInferenceRequest {
            model_id: candidate.model_id.clone(),
            request: invocation.request,
            timeout_ms: Some(60_000),
        })
        .await?;
    Ok(VirtualModelResponse {
        status_code: response.status_code,
        body: response.body,
        headers: response
            .served_by
            .map(|host| vec![("x-mesh-served-by".into(), host)])
            .unwrap_or_default(),
        event_stream: false,
    })
}
```

The built-in implementation in `crates/mesh-llm-moa-plugin/src/lib.rs` is the
complete conformance example.

V1 host inference accepts only concrete model ids. It rejects built-in aliases
and all virtual model ids, including the caller, so a plugin cannot recurse.
Virtual-to-virtual composition can be added later with an explicit hop budget.

## Routing And Affinity Hooks

Virtual-model orchestration chooses *which model* should handle a nested call.
Affinity chooses *where* an eligible instance of that model should run. Keep
those policies separate so every nested call still receives mesh admission,
health, cache-locality, and sticky-routing behavior.

An optional affinity hook runs only after the host applies hard eligibility:

```text
host trust/admission/capability filter
  -> optional plugin candidate ranking
  -> host validates and clamps the result
  -> host dispatches and records the outcome
```

The rank request contains opaque target ids and a bounded snapshot of locality,
health, RTT, capacity/queue signals, and cache-affinity evidence. Request
context contains host-derived session, explicit cache-key, and stable scaffold
hashes by default; it does not disclose raw prompt text to a generic routing
plugin.

The plugin may reorder or remove eligible candidates. It may not invent a
target, restore a host-rejected target, bypass admission or trust, or weaken a
required model capability. The host uses its current deterministic affinity
logic when the plugin is absent, unhealthy, times out, or returns an invalid
ranking.

`PluginContext::infer` should make this policy explicit:

```rust
context
    .infer(
        route.model,
        request.with_reasoning_effort(route.effort),
        RoutingPolicy::PluginHook("affinity"),
    )
    .await
```

`RoutingPolicy::HostDefault` remains the safe default. A named hook must be
declared by the calling plugin; the host rejects ambiguous or undeclared hooks.
A later protocol slice may add `RequireTarget`, but only for a target already
present in the host's eligible snapshot.

## Gossip Extensions

Plugins already subscribe to `PEER_UP`, `PEER_DOWN`, and `PEER_UPDATED`, can
use declared mesh channels, and contribute healthy external inference models
to the host's normal model advertisement. They do not currently have a typed
initial peer snapshot or a bounded way to add plugin-specific advisory data to
peer announcements.

Add host-governed gossip extensions rather than exposing the gossip encoder:

```proto
message GossipExtensionManifest {
  string namespace = 1;
  uint32 schema_version = 2;
  GossipVisibility visibility = 3;
  uint32 max_bytes = 4;
}

message PluginGossipAdvertisement {
  string namespace = 1;
  uint32 schema_version = 2;
  bytes payload = 3;
  uint64 expires_at_unix_ms = 4;
}
```

A plugin submits a namespaced value and TTL to the host. The host validates
the declared namespace and schema, enforces fixed size/rate/TTL and public vs
private visibility limits, and publishes the value as a self-reported advisory
claim. Compatible values are exposed through an initial peer snapshot and
subsequent peer events.

Plugin gossip may not overwrite endpoint identity, ownership/trust,
admission, role, hardware, serving-model truth, or the host's core
cache-affinity evidence. Direct-peer-only propagation should be the default;
transitive propagation requires an explicitly reviewed schema and bound.

## MoA Migration

Move the existing `model: "mesh"` MoA implementation behind this interface as
the first in-process conformance plugin. MoA exercises parallel nested
inference, candidate snapshots, partial failure, cancellation, streaming
fan-in, and aggregate usage before Goose adds session and tool semantics.

Migrate without changing the public model id:

1. add the registry and buffered nested-inference path;
2. register an in-process `mesh-moa` plugin through the existing in-process
   plugin loader;
3. adapt the current orchestrator behind its virtual-model handler and run
   response/error parity tests;
4. switch `model: "mesh"` dispatch to the registry;
5. remove the special ingress intercept only after streaming, cancellation,
   and usage accounting have parity.

The host advertises `mesh` from healthy plugin registration. The MoA plugin
does not directly mutate gossip. Its nested calls choose concrete models, then
host or plugin-ranked affinity chooses an eligible placement for each model.

## Delivery Order

1. Add manifest types, builders, validation, and registry projection.
2. Add buffered host-to-plugin invocation and plugin-to-host inference.
3. Add bounded affinity-rank and namespaced gossip-extension contracts.
4. Move the built-in MoA path behind the contract as a conformance test.
5. Add negotiated streaming, cancellation, and aggregate usage.
6. Implement the Goose/OpenJEV virtual model against the stable contract.

Goose is deliberately downstream of this boundary. Its unrolled state machine
can yield on each host inference or tool operation without adding Goose types
to the host runtime.
