# ADR: Native Mesh Node SDK for mesh-llm

## Status

Accepted direction, superseding the previous client-only SDK contract.

The earlier SDK work treated `MeshClient` as the primary product surface and
left host/server APIs, local model management, and model downloading out of
scope. That shape was useful for extraction, but it is not the right long-term
SDK contract. The SDK should embed a mesh node into an application. That node
may consume inference from the mesh, manage local model artifacts, serve local
models into the mesh, or combine those roles.

---

## Context

Applications embedding mesh-llm need more than an OpenAI-compatible client. A
useful SDK must let the embedding application participate in the mesh as a
first-class node:

- discover and call models already available in the mesh
- download, inspect, install, delete, and clean up local model artifacts
- load and unload local models for serving
- advertise serving capacity and model capabilities to peers
- observe model, serving, request, and connection lifecycle events

The current tree already has pieces of this split across the CLI, host runtime,
client SDK crates, and management API:

- `mesh-client` contains low-level client transport and routing behavior
- `mesh-llm-api-server` exposes a client-oriented Rust SDK
- `mesh-llm-ffi` wraps the SDK for Swift and Kotlin
- `mesh-llm-node` exists as the intended embeddable node boundary
- `mesh-llm-host-runtime` owns CLI, management API, model acquisition, model
  cache inspection, local runtime control, and serving orchestration

Because the existing client SDK has no known external consumers, preserving the
`MeshClient`-first surface is not a compatibility requirement. We should
replace it with the node-oriented SDK now, before language bindings and app
integrations depend on the narrower shape.

---

## Decision

The public SDK concept is `MeshNode`.

`MeshNode` is the object an application embeds. It owns node identity,
connection lifecycle, model cache configuration, local runtime configuration,
event emission, and role configuration. Client-only operation is just a node
with serving disabled. Serving operation is the same node with local model
serving enabled through an in-process serving controller. SDK serving APIs must
not use the local REST management API as their primary implementation; REST is
only an adapter for controlling an external daemon.

The reference in-process serving controller is the host runtime management
state, `mesh-llm-host-runtime::api::MeshApi`. Host runtime construction wires
that controller into `MeshNodeBuilder` with
`MeshApi::configure_sdk_node_builder(...)`, so `MeshNode::serving().load()` and
`MeshNode::serving().unload()` enter the existing runtime-control loop directly.

The public API is organized into namespaces rather than a large flat method
list:

```rust
node.inference()
node.models()
node.serving()
node.status()
node.events()
```

Public SDK names should use Mesh/product language. The current internal serving
runtime name is not part of the public SDK contract. APIs should say
`ModelRuntime`, `ServingConfig`, `StageTopology`, `DevicePolicy`, and
`ModelConfig`, not implementation-specific runtime names.

### Example

```rust
let node = MeshNode::builder()
    .identity(identity)
    .join(invite_token)
    .cache_dir(cache_dir)
    .runtime_dir(runtime_dir)
    .serving_enabled(true)
    .build()?;

node.start().await?;

let model = node
    .models()
    .download("Qwen3-0.6B-Q4_K_M", DownloadOptions::default())
    .await?;

node.serving()
    .load(model.model_ref, LoadModelOptions::default())
    .await?;

let request_id = node.inference().chat(request, listener).await?;
```

---

## Public API Shape

This is the target SDK shape. Exact type names may change during
implementation, but the ownership boundaries should not.

### MeshNode

```rust
impl MeshNode {
    pub fn builder() -> MeshNodeBuilder;

    pub async fn start(&self) -> Result<(), MeshError>;
    pub async fn stop(&self) -> Result<(), MeshError>;
    pub async fn reconnect(&self) -> Result<(), MeshError>;

    pub fn inference(&self) -> MeshInference;
    pub fn models(&self) -> MeshModels;
    pub fn serving(&self) -> MeshServing;
    pub fn status(&self) -> MeshStatusApi;
    pub fn events(&self) -> MeshEvents;
}
```

`MeshNode` replaces `MeshClient` as the primary SDK type. A transitional
`MeshClient` alias or wrapper may exist only while internal call sites and
generated bindings are moved.

### Builder

```rust
impl MeshNodeBuilder {
    pub fn identity(self, identity: OwnerKeypair) -> Self;
    pub fn join(self, token: InviteToken) -> Self;
    pub fn user_agent(self, user_agent: impl Into<String>) -> Self;
    pub fn connect_timeout(self, timeout: Duration) -> Self;
    pub fn cache_dir(self, path: impl Into<PathBuf>) -> Self;
    pub fn runtime_dir(self, path: impl Into<PathBuf>) -> Self;
    pub fn serving_enabled(self, enabled: bool) -> Self;
    pub fn device_policy(self, policy: DevicePolicy) -> Self;
    pub fn serving_controller(self, controller: Arc<dyn ServingController>) -> Self;
    pub fn build(self) -> Result<MeshNode, MeshError>;
}
```

The SDK does not read identity or credentials from ambient filesystem locations.
The embedding application passes identity and storage policy explicitly.
When the SDK is embedded in the host runtime, host startup must attach the live
`MeshApi` controller to the builder before `build()`. Setting
`serving_enabled(true)` without a controller only records desired configuration;
it does not provide load/unload capability.

### Inference API

```rust
impl MeshInference {
    pub async fn list_models(&self) -> Result<Vec<Model>, MeshError>;
    pub async fn chat(
        &self,
        request: ChatRequest,
        listener: Arc<dyn EventListener>,
    ) -> Result<RequestId, MeshError>;
    pub async fn responses(
        &self,
        request: ResponsesRequest,
        listener: Arc<dyn EventListener>,
    ) -> Result<RequestId, MeshError>;
    pub async fn cancel(&self, request_id: RequestId) -> Result<(), MeshError>;
}
```

`chat` maps to `/v1/chat/completions`. `responses` maps to `/v1/responses`.
Both deliver incremental output through the node event model.

### Model Management API

Model management is part of the node SDK because serving depends on local model
state and because the embedding application owns cache policy.

```rust
impl MeshModels {
    pub async fn recommended(&self) -> Result<Vec<ModelSummary>, MeshError>;
    pub async fn search(&self, query: ModelSearchQuery) -> Result<Vec<ModelSummary>, MeshError>;
    pub async fn show(&self, model_ref: impl AsRef<str>) -> Result<ModelDetails, MeshError>;

    pub async fn installed(&self) -> Result<Vec<InstalledModel>, MeshError>;
    pub async fn cache_status(&self) -> Result<ModelCacheStatus, MeshError>;

    pub async fn download(
        &self,
        model_ref: impl AsRef<str>,
        options: DownloadOptions,
    ) -> Result<DownloadedModel, MeshError>;
    pub async fn cancel_download(&self, download_id: DownloadId) -> Result<(), MeshError>;

    pub async fn delete(
        &self,
        model_ref: impl AsRef<str>,
        options: DeleteModelOptions,
    ) -> Result<DeleteModelResult, MeshError>;
    pub async fn cleanup(&self, policy: CleanupPolicy) -> Result<CleanupResult, MeshError>;
    pub async fn prune_derived_cache(
        &self,
        policy: PrunePolicy,
    ) -> Result<PruneResult, MeshError>;
}
```

Definitions:

- `download` puts source model artifacts on disk.
- `delete` removes source model artifacts and must refuse loaded models unless
  the caller explicitly forces the operation.
- `cleanup` removes managed source artifacts according to an unused/stale
  policy.
- `prune_derived_cache` removes derived runtime artifacts while preserving source
  model artifacts.

The SDK should accept the same model reference forms as the CLI:

- catalog ids such as `Qwen3-0.6B-Q4_K_M`
- Hugging Face refs such as `unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL`
- immutable `hf://namespace/repo@revision` refs
- local model paths where supported by the serving runtime

### Serving API

Serving lifecycle is separate from model management. Downloading a model does
not make it active. Loading a model makes it served by this running node.

```rust
impl MeshServing {
    pub async fn load(
        &self,
        model_ref: impl AsRef<str>,
        options: LoadModelOptions,
    ) -> Result<ServedModel, MeshError>;
    pub async fn unload(
        &self,
        target: UnloadTarget,
        options: UnloadModelOptions,
    ) -> Result<(), MeshError>;
    pub async fn unload_model(
        &self,
        model_id: impl AsRef<str>,
        options: UnloadModelOptions,
    ) -> Result<(), MeshError>;
    pub async fn unload_instance(
        &self,
        instance_id: impl AsRef<str>,
        options: UnloadModelOptions,
    ) -> Result<(), MeshError>;
    pub async fn served_models(&self) -> Result<Vec<ServedModel>, MeshError>;
    pub async fn status(&self) -> Result<ServingStatus, MeshError>;
    pub async fn set_device_policy(&self, policy: DevicePolicy) -> Result<(), MeshError>;
}

pub enum UnloadTarget {
    Model(String),
    Instance(String),
}

pub struct UnloadModelOptions {
    pub drain_timeout: Duration,
    pub force: bool,
}

pub struct ServedModel {
    pub model_ref: String,
    pub model_id: String,
    pub instance_id: Option<String>,
    pub state: ServingModelState,
    pub backend: Option<String>,
    pub capabilities: ModelCapabilities,
    pub context_length: Option<u32>,
    pub error: Option<String>,
}

pub enum ServingModelState {
    Loading,
    Ready,
    Failed,
    Unloading,
    Stopped,
    Unknown(String),
}

pub enum DevicePolicy {
    Auto,
    Cpu,
    Gpu { device_ids: Vec<String> },
}
```

The serving API advertises normal mesh model descriptors and capabilities. It
must preserve mixed-version mesh compatibility. Any serving SDK node is a real
mesh participant, not a side-channel runtime.

Load accepts model references only. It never accepts runtime instance ids.
Loading an already-loading or already-ready model should return the existing
`ServedModel`. Unload targets are explicit: callers unload either a model or a
specific runtime instance. Unloading a missing target succeeds by default so app
lifecycle cleanup can be idempotent. Default unload behavior drains active
requests before unloading; `force` bypasses that drain where the runtime
supports it.

Serving errors should be typed at the SDK boundary:

```rust
pub enum ServingError {
    ModelNotFound { model_ref: String },
    DownloadRequired { model_ref: String },
    LoadFailed { model_ref: String, message: String },
    UnloadFailed { target: UnloadTarget, message: String },
    UnsupportedDevicePolicy { policy: DevicePolicy },
    RuntimeUnavailable { message: String },
}
```

### Status API

```rust
impl MeshStatusApi {
    pub async fn node(&self) -> Result<NodeStatus, MeshError>;
    pub async fn peers(&self) -> Result<Vec<PeerStatus>, MeshError>;
    pub async fn models(&self) -> Result<Vec<Model>, MeshError>;
    pub async fn runtime(&self) -> Result<RuntimeStatus, MeshError>;
}
```

### Events

The SDK exposes one node event stream. Language-specific bindings may adapt that
stream to callbacks, async sequences, flows, or host framework events.

```rust
pub enum MeshEvent {
    Starting,
    Joined { node_id: String },
    PeerDiscovered { node_id: String },
    ModelsUpdated { models: Vec<Model> },
    Disconnected { reason: String },

    ModelDownloadStarted { download_id: DownloadId, model_ref: String },
    ModelDownloadProgress { download_id: DownloadId, received_bytes: u64, total_bytes: Option<u64> },
    ModelDownloadCompleted { download_id: DownloadId, model: DownloadedModel },
    ModelDownloadFailed { download_id: DownloadId, error: String },

    ModelInstalled { model: InstalledModel },
    ModelDeleted { model_ref: String },

    ModelLoading { model_ref: String },
    ModelReady { model: ServedModel },
    ModelUnloading { model_id: String },
    ModelUnloaded { model_id: String },
    ModelFailed { model_ref: String, error: String },

    RequestStarted { request_id: RequestId, model: String },
    TokenDelta { request_id: RequestId, delta: String },
    RequestCompleted { request_id: RequestId },
    RequestFailed { request_id: RequestId, error: String },
}
```

---

## Crate Structure

Target ownership:

- `mesh-sdk` or the reworked `mesh-llm-api-server`
  - public Rust SDK: `MeshNode`, namespaced APIs, public SDK types
- `mesh-client`
  - low-level client transport, routing, request forwarding, protocol handling
- `mesh-llm-node`
  - embeddable node model management and serving control behind SDK-safe APIs
- `mesh-llm-ffi`
  - UniFFI wrapper over the node SDK for Swift, Kotlin, and future native
    bindings
- `sdk/swift`
  - thin generated Swift package and platform-specific wrapper ergonomics
- `sdk/kotlin`
  - thin generated Kotlin/JVM/Android package and wrapper ergonomics

The existing `mesh-llm-api-server` crate may be renamed to `mesh-sdk` or reworked in place.
Because there are no known external users, keeping the old `MeshClient` naming
is not required.

CLI commands should become thin wrappers over library APIs where practical. The
CLI should not remain the owner of model search, show, download, installed,
delete, cleanup, prune, or local load/unload behavior.

---

## Extraction Map

Existing host-runtime code should move behind SDK-safe library boundaries in
phases.

| Responsibility | Current home | Target owner |
|---|---|---|
| Client join, model list, chat, responses, cancel, reconnect | `mesh-client`, `mesh-llm-api-server` | `MeshNode::inference()` backed by `mesh-client` |
| Model search, show, recommended, download | CLI/model modules in `mesh-llm-host-runtime` | `MeshNode::models()` backed by model-management library code |
| Installed model scanning and cache status | `models/local.rs` and related host modules | `MeshNode::models()` |
| Model delete, cleanup, derived-cache prune | CLI/model modules in `mesh-llm-host-runtime` | `MeshNode::models()` |
| Runtime load, unload, served models, runtime status | host runtime-control loop through `mesh-llm-host-runtime::api::MeshApi` | `MeshNode::serving()` backed by the in-process `ServingController` trait |
| Device policy and GPU inventory | host system/runtime modules | `MeshNode::serving()` and status APIs |
| Protocol, routing, compatibility types | shared protocol/client/host modules | shared crates consumed by SDK and host runtime |

Do not expose CLI parser types, terminal UI concerns, process-global config, or
host-runtime internals as SDK types.

---

## Platform Scope

Initial target:

- Rust SDK for embedded desktop/server applications
- macOS serving support where the native runtime is available
- Swift macOS bindings once the Rust node API is stable enough to wrap
- Kotlin/JVM and Android client/model-management support where platform
  constraints allow it

Out of initial serving scope:

1. iOS serving mode
2. Android serving mode
3. Browser SDK
4. Web console assets in the SDK
5. Plugin host in the SDK
6. Generic third-party model backend registration

The SDK is not a generic LLM provider framework. It embeds mesh-llm's native
model serving path behind product-level Mesh names.

---

## FFI Toolchain

`uniffi-rs v0.31+` remains the chosen FFI layer. It generates Swift and Kotlin
bindings from a single Rust interface definition, supports async functions
natively, and has an active maintenance track.

Language SDKs should stay thin:

- generated bindings map to the Rust node SDK
- platform wrappers provide idiomatic event and async integration
- shared behavior stays in Rust

---

## Credentials And Storage

The SDK does not silently load credentials from host-global paths. The embedding
application provides identity and persistence policy.

The node builder should accept explicit paths or storage policy for:

- model cache
- runtime directory
- metadata cache
- logs
- temporary derived artifacts

Defaults are allowed for developer ergonomics on desktop platforms, but those
defaults must be visible in status/config APIs and overridable by the caller.

---

## Protocol Compatibility

The mesh-llm control plane supports two protocol versions:

- `mesh-llm/0`: legacy JSON/raw payloads, preserved for backward compatibility
- `mesh-llm/1`: protobuf framing with `meshllm.node.v1` schema, preferred for
  all new connections

Mixed meshes containing `/0` and `/1` nodes are supported. SDK nodes are
first-class mesh participants and must preserve that compatibility. Gossip
fields, model descriptors, capability advertisements, stage/topology state, and
stream types must remain additive unless an explicit breaking protocol revision
is approved.

Any change to shared protocol, gossip, model capabilities, routing, or serving
state requires backward-compatibility tests against released mesh nodes.

---

## Test Strategy

Extraction and SDK changes should keep the existing TDD discipline:

1. Write failing tests in the destination crate.
2. Move or wrap the minimal implementation needed to pass.
3. Remove duplicated behavior from CLI/runtime owners once library APIs exist.

Minimum gates by change type:

- client/inference SDK changes: `cargo test -p mesh-client -p mesh-llm-api-server`
- FFI changes: `cargo test -p mesh-llm-ffi` plus Swift/Kotlin smoke where touched
- model-management changes: model search/show/download/installed/delete tests
- serving changes: runtime load/unload/status tests plus mixed-version protocol
  checks when gossip or routing changes

Before landing broad SDK API changes, run:

```bash
just build
cargo test --workspace
cargo clippy --workspace -- -D warnings
```

---

## Consequences

### Enables

- Applications can embed one mesh node rather than stitching together a client,
  CLI subprocesses, and management API calls.
- Model management becomes part of the supported app integration story.
- Local serving is designed as a first-class SDK role.
- Swift/Kotlin bindings can expose one coherent node API instead of a
  client-only surface that later grows incompatible host concepts.

### Costs

- The existing `MeshClient` SDK surface must be renamed or wrapped during
  migration.
- More host-runtime code must move behind library boundaries.
- Model management APIs must define stable identities and deletion safety rules.
- Serving APIs must make runtime state, device policy, and capability
  advertisement explicit.

### Alternatives Rejected

- Keep a client-only SDK and add a separate serving SDK. Rejected because model
  management, serving, and inference all share node identity, cache policy,
  eventing, and mesh lifecycle.
- Expose a generic third-party backend provider interface. Rejected for this SDK
  phase; the product is Mesh model serving, not a general provider framework.
- Leave model management as CLI-only. Rejected because an embedded serving app
  must manage local models without shelling out to the CLI.
