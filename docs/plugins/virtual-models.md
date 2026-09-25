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
}
```

V1 validation rules:

- `model_id` and `handler` are non-empty;
- a plugin may declare each model id once;
- model ids may not collide with a concrete, built-in, or another plugin's
  virtual model;
- a model is advertised only while its plugin is healthy;
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
preserved. The host assigns a correlation id in the envelope. Cancellation is
an explicit host notification for that id; dropping the client connection also
cancels all nested inference owned by the call.

## Plugin-to-Host Inference

Plugin-originated calls use the reserved request-id range already used by
`PluginContext::open_mesh_stream`:

```proto
message HostInferenceRequest {
  uint64 parent_request_id = 1;
  string model_id = 2;
  string request_json = 3;
  optional uint64 deadline_unix_ms = 4;
}

message HostInferenceResponse {
  string response_json = 1;
  string served_by = 2;
  string usage_json = 3;
}
```

The Rust authoring API should hide this wire shape:

```rust
plugin! {
    metadata: metadata,
    virtual_models: [
        virtual_model::new("mesh-agent")
            .supports_tools(true)
            .handle(handle_mesh_agent),
    ],
}

async fn handle_mesh_agent(
    request: ChatRequest,
    context: &mut PluginContext<'_>,
) -> Result<ChatResponse> {
    let route = choose_route(&request).await?;
    context
        .infer(route.model, request.with_reasoning_effort(route.effort))
        .await
}
```

V1 host inference accepts only concrete model ids. It rejects built-in aliases
and all virtual model ids, including the caller, so a plugin cannot recurse.
Virtual-to-virtual composition can be added later with an explicit hop budget.

## Delivery Order

1. Add manifest types, builders, validation, and registry projection.
2. Add buffered host-to-plugin invocation and plugin-to-host inference.
3. Move the built-in MoA path behind the contract as a conformance test.
4. Add negotiated streaming, cancellation, and aggregate usage.
5. Implement the Goose/OpenJEV virtual model against the stable contract.

Goose is deliberately downstream of this boundary. Its unrolled state machine
can yield on each host inference or tool operation without adding Goose types
to the host runtime.
