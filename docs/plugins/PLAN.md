# Plugins Plan

Track implementation work here for the `mesh-llm` plugin architecture defined
in [README.md](README.md).

## Sequencing Principles

The work should land in this order:

1. Define the protocol and manifest before refactoring plugin features onto it.
2. Keep the host control plane small and stable before adding higher-level projections like MCP and HTTP.
3. Add endpoint registration before building real provider plugins.
4. Validate the architecture with one real inference provider plugin before broadening the surface area further.
5. Add crypto as a host-owned service surface only after the rest of the plugin surfaces are stable.

## Plugin Manager, Catalog, And Blackboard Extraction

The plugin architecture should grow a first-class plugin management layer before
third-party plugins become a normal user workflow.

Create a new workspace crate:

```text
crates/mesh-llm-plugin-manager
```

This crate owns plugin package management and should not print directly. It
should expose structured progress and status events that the host CLI renders in
the existing `mesh-llm` style.

The manager crate should own:

- parsing plugin install references
- resolving bare plugin names through the public catalog
- resolving GitHub releases
- selecting native release assets for the local OS and CPU architecture
- downloading release assets
- extracting and validating plugin archives
- recording installed plugin metadata
- updating installed plugins
- enabling, disabling, and deleting installed plugins
- startup compatibility metadata used by the host to diagnose load failures

`mesh-llm-host-runtime` should own the CLI command shape, human output, JSON
output where supported, and host startup integration.

### CLI Surface

Initial commands:

```bash
mesh-llm plugins install <ref>
mesh-llm plugins update <plugin>
mesh-llm plugins enable <plugin>
mesh-llm plugins disable <plugin>
mesh-llm plugins delete <plugin>
mesh-llm plugins list
mesh-llm plugins info <plugin>
mesh-llm plugins search [query]
```

Supported install references:

```bash
mesh-llm plugins install https://github.com/mesh-llm/cool-plugin
mesh-llm plugins install mesh-llm/cool-plugin
mesh-llm plugins install mesh-llm/cool-plugin@1.1.0
mesh-llm plugins install https://github.com/mesh-llm/cool-plugin@1.1.0
mesh-llm plugins install cool-plugin
```

Bare names resolve through the catalog. Explicit GitHub URLs and `owner/repo`
references bypass catalog lookup.

CLI output should match other mesh commands:

- emoji-led status lines
- indeterminate progress while resolving catalog entries, GitHub releases, and
  updates
- progress bars for bounded downloads
- concise success, skip, and failure summaries
- machine-readable JSON output where plugin commands expose it

### Native GitHub Release Packages

Plugins are native binaries distributed through GitHub Releases. Release asset
selection is OS and CPU architecture only. Plugin packages must not introduce
GPU backend flavor suffixes such as CUDA, ROCm, Vulkan, Metal, or CPU.

Versioned assets use:

```text
<plugin-name>-<version>-<target-triple>.<archive-ext>
```

Stable latest aliases may use:

```text
<plugin-name>-<target-triple>.<archive-ext>
```

Supported targets:

| Platform | Target triple | Archive |
|---|---|---|
| macOS Apple Silicon | `aarch64-apple-darwin` | `.tar.gz` |
| macOS Intel | `x86_64-apple-darwin` | `.tar.gz` |
| Linux x86_64 | `x86_64-unknown-linux-gnu` | `.tar.gz` |
| Linux ARM64 | `aarch64-unknown-linux-gnu` | `.tar.gz` |
| Windows x86_64 | `x86_64-pc-windows-msvc` | `.zip` |
| Windows ARM64 | `aarch64-pc-windows-msvc` | `.zip` |

Archives should be rooted under one plugin directory and contain at least
`plugin.toml` plus the native executable. Windows executables use `.exe`.

Install selection:

1. Parse the install reference.
2. Resolve through the catalog if the reference is a bare plugin name.
3. Resolve the requested GitHub release.
4. Detect the local target triple.
5. Prefer a versioned asset for the target triple.
6. Fall back to the stable alias for the target triple.
7. Fail clearly if no compatible asset exists.

Installed metadata should include:

- plugin name
- source repository
- installed version
- target triple
- downloaded asset name
- install path
- enabled or disabled state
- last observed plugin protocol version
- last startup status
- last startup error

### Hugging Face Catalog Dataset

Create the public Hugging Face Dataset as part of the implementation plan. The
suggested dataset is:

```text
meshllm/plugin-catalog
```

The dataset is metadata only. GitHub releases remain the source of native plugin
archives for install and update.

The canonical file is `plugins.jsonl`, with one plugin per line:

```json
{"name":"blackboard","description":"Shared mesh blackboard for agent status, findings, questions, answers, and searchable coordination notes.","github_url":"https://github.com/mesh-llm/blackboard","author_email":"maintainers@meshllm.cloud","author_name":"Mesh LLM"}
```

Required fields:

- `name`
- `description`
- `github_url`
- `author_email`
- `author_name`

Catalog rules:

- `name` must be unique
- `name` should match the plugin manifest ID and release asset prefix
- unknown extra fields are ignored for forward compatibility
- catalog lookup never downloads binaries from Hugging Face
- the manager should support a configurable catalog repo for tests and private
  catalogs
- tests should use local `plugins.jsonl` fixtures rather than the live dataset

The first catalog entry should be `blackboard`, backed by the GitHub repository
`mesh-llm/blackboard`.

### Startup Compatibility

The plugin protocol is versioned. Normal mesh startup should never auto-update
plugins or block on network plugin checks.

Startup behavior:

1. Load installed plugin metadata.
2. Skip disabled plugins.
3. Preflight obvious local failures such as missing binaries or wrong target triples.
4. Start enabled plugins.
5. Send an initialize request with host protocol and version information.
6. Validate plugin identity, plugin version, protocol compatibility, capabilities,
   and manifest.
7. Register compatible plugins.
8. Mark incompatible optional plugins as `incompatible` and exclude their
   capabilities, routes, endpoints, and MCP contributions.
9. Fail startup only when a required plugin is incompatible or unavailable.

The recovery path for an incompatible plugin should be explicit:

```text
mesh-llm plugins update <plugin>
```

The host should eventually move from strict protocol equality to negotiated
compatibility ranges, for example:

```text
host protocol: 2
plugin supports: 1..=2
selected protocol: 2
```

### Blackboard As The First External Plugin

Use the plugin transport and capability-routing work from
`/Users/jdumay/.codex/worktrees/4d86/mesh-llm` as the base for extracting
blackboard.

Create the external first-party plugin repository at:

```text
~/code/mesh-blackboard
```

The upstream GitHub repository should be:

```text
mesh-llm/blackboard
```

Keep the plugin identity stable:

- plugin/catalog name: `blackboard`
- manifest ID: `blackboard`
- release asset prefix: `blackboard`
- GitHub repository: `mesh-llm/blackboard`

Move the blackboard implementation out of
`crates/mesh-llm-host-runtime/src/plugins/blackboard` and into the external repo.
The host runtime should retain only generic plugin hosting, capability routing,
and compatibility aliases.

Compatibility requirements:

- keep `/api/blackboard/*` as legacy aliases onto blackboard plugin
  capability/HTTP bindings while users migrate
- keep `mesh-llm blackboard ...` as a compatibility CLI shim if needed
- if blackboard is not installed, print a direct install hint:
  `mesh-llm plugins install blackboard`
- blackboard remains optional by default and must not block core mesh startup
  unless explicitly configured as required

Blackboard should be the first end-to-end validation plugin for:

- catalog install
- GitHub release install
- update
- enable
- disable
- delete
- startup compatibility diagnostics
- capability-backed routes
- MCP projection
- legacy API and CLI compatibility

## Proposed Sequence

### Phase 1: Protocol And Manifest

Define the v2 plugin control-plane protocol.

This phase should specify:

- plugin manifest schema
- plugin lifecycle messages
- request / response invocation model
- endpoint registration messages
- health and availability messages
- negotiated stream messages
- cancellation and error messages

Target outputs:

- manifest types
- protocol message types
- versioning / compatibility rules
- host and plugin runtime interfaces

This is the foundation for everything else.

### Phase 2: Host Runtime Core

Implement the new host/plugin runtime without changing every feature at once.

This phase should deliver:

- one long-lived control connection per plugin
- negotiated short-lived streams
- plugin manifest registration on startup
- plugin health supervision
- endpoint health supervision
- separation of plugin state and endpoint state

Target behavior:

- plugins stay loaded even when managed endpoints become unavailable
- endpoint recovery automatically restores availability
- large or streaming payloads do not block the control connection

### Phase 3: Manifest-Driven MCP

Implement MCP as a host projection over manifest-declared plugin services.

This phase should deliver:

- manifest-declared tools
- manifest-declared resources
- manifest-declared resource templates
- manifest-declared prompts
- manifest-declared completions
- namespaced MCP aggregation in the host

Target behavior:

- plugins do not implement MCP JSON-RPC directly
- `mesh-llm` remains the MCP server
- external MCP endpoints can be aggregated later through the same host surface

### Phase 4: Manifest-Driven HTTP Bindings

Implement HTTP as a host projection over manifest-declared plugin services.

This phase should deliver:

- plugin-defined HTTP bindings
- default plugin-owned route namespacing
- buffered request / response support
- streamed request / response support using negotiated streams
- validation and error mapping in the host

Target behavior:

- plugin authors do not implement HTTP servers
- plugin-specific host route code is no longer required for each new plugin

### Phase 5: Capability Resolution

Add capability-based routing for stable product contracts.

This phase should deliver:

- named capability registration in plugin manifests
- host resolution of one provider for a capability
- optional promoted product routes backed by capabilities

Examples:

- `object-store.v1`
- `inference-endpoint-provider.v1`
- `mcp-endpoint-provider.v1`

Target behavior:

- core depends on capability contracts, not plugin IDs
- top-level product APIs can remain stable even if providers change

### Phase 6: Endpoint Registration

Implement concrete endpoint registration support.

This phase should deliver:

- inference endpoint registration
- external MCP endpoint registration
- endpoint descriptors
- endpoint health and availability tracking
- optional lifecycle hooks for plugin-managed services

Target behavior:

- plugins can register local or managed OpenAI-compatible inference servers
- plugins can register external MCP servers
- `mesh-llm` talks directly to those endpoints
- plugin IPC remains the control plane, not the data path

### Phase 7: Migrate Existing Built-Ins

Move built-in plugin behavior onto the new architecture.

This phase should include:

- moving blackboard fully behind generic plugin transport
- removing plugin-specific core mesh stream behavior where generic plugin channels are sufficient
- moving plugin-specific HTTP behavior behind manifest-driven bindings or capability routes

Target behavior:

- bundled plugins remain auto-registered
- core mesh logic becomes plugin-agnostic

### Phase 8: Validation Plugins

Build real plugins that exercise the design.

The first plugin-hosted inference migration should be the current llama backend.

After that, build an MLX endpoint provider plugin.

After that, build at least one external MCP endpoint plugin.

These plugins should validate:

- endpoint registration
- endpoint health transitions
- direct host-to-endpoint communication
- capability resolution
- MCP aggregation
- HTTP binding ergonomics

The llama pluginization work should move the current local llama-style serving path behind the new plugin-hosted inference endpoint contract.

The MLX plugin should then take inspiration from the in-process inference-server work in [PR #103](https://github.com/Mesh-LLM/mesh-llm/pull/103), but implemented using the new plugin endpoint registration architecture rather than direct built-in runtime ownership in core.

After that, add an attached external inference plugin, with Lemonade as the first target for that mode. That should take inspiration from [PR #150](https://github.com/Mesh-LLM/mesh-llm/pull/150), but implemented using endpoint registration rather than ad hoc `inference/register` notifications in the transport layer.

### Phase 9: Host-Owned Plugin Crypto API

Add host-owned crypto services for plugins.

This phase should deliver:

- `crypto.get_identity`
- `crypto.seal`
- `crypto.open`

Target behavior:

- plugins do not read the owner keystore directly
- plugins do not receive owner secret keys
- secret-key operations remain in the host process

## Immediate Next Steps

The best near-term execution order is:

1. Write the manifest and protocol types.
2. Implement the new control connection and negotiated stream runtime.
3. Add manifest-driven MCP.
4. Add manifest-driven HTTP bindings.
5. Add endpoint registration and health tracking.
6. Pluginize the llama backend.
7. Build the MLX endpoint provider plugin.
8. Migrate blackboard off bespoke core behavior.
9. Add the host-owned crypto APIs.

## Test Strategy

The new plugin architecture needs explicit host/runtime integration tests in addition to unit tests.

### MCP And HTTP Projection Testing

Create fake MCP and HTTP servers plus dedicated test plugins that exercise projection behavior and failure modes.

This test setup should validate:

- manifest-declared MCP tools, resources, prompts, and completions
- manifest-declared HTTP bindings
- namespacing and collision handling
- buffered request / response behavior
- streamed request / response behavior
- negotiation and cleanup of short-lived streams
- cancellation behavior
- malformed payload handling
- timeout handling
- endpoint disappearance and recovery
- projection behavior when plugins are healthy but endpoints are not

Include corner cases such as:

- duplicate tool or route names
- invalid schemas or invalid manifests
- large request bodies
- large response bodies
- partial stream writes
- abrupt stream disconnects
- plugin restart while requests are in flight
- endpoint flapping between healthy and unhealthy

### Inference Plugin Testing

Use the pluginized llama backend first, then an MLX-backed inference plugin, to validate plugin-hosted inference endpoint registration end to end.

This should validate:

- endpoint registration
- model discovery
- request routing through the registered endpoint
- streaming response handling
- endpoint health transitions
- automatic endpoint recovery

The pluginized llama backend should prove that the current built-in serving path can move behind the plugin contract without changing the host-facing inference model.

The MLX plugin should then prove that a second plugin-hosted backend can use the same contract while owning its own runtime behavior.

Take implementation cues from the current llama runtime behavior first, and then from [PR #103](https://github.com/Mesh-LLM/mesh-llm/pull/103):

- plugin-hosted local model serving with llama semantics
- plugin-hosted local inference serving
- model discovery from the owned runtime
- direct routing through the registered endpoint
- endpoint health and lifecycle management separated from plugin liveness

After that, validate the attached-external-endpoint mode with Lemonade:

- connect to an already-running Lemonade endpoint
- perform health checks and model discovery
- register the endpoint and its models with the host
- mark the endpoint unavailable on health failure without unloading the plugin
- restore the endpoint automatically when health returns

If MLX or Lemonade is not available locally, keep a fallback test mode with a fake OpenAI-compatible inference server for protocol and routing validation.

### Explicit Follow-Up TODOs

- once the llama backend is pluginized, keep MLX aligned to the same plugin-hosted inference endpoint contract

### Additional Testing Needed

Beyond fake MCP/HTTP servers and the MLX/Lemonade providers, we should also test:

- backward compatibility of the plugin control protocol where required
- plugin startup and shutdown behavior
- host behavior when a plugin connects but never fully initializes
- host behavior when a plugin advertises endpoints and then disconnects
- capability resolution when zero, one, or multiple providers exist
- promoted product routes backed by capabilities
- plugin health vs endpoint health separation
- crypto host API behavior for `crypto.get_identity`, `crypto.seal`, and `crypto.open`
- security properties around short-lived stream naming, reuse, expiration, and cleanup
- concurrency with multiple plugins and multiple simultaneous streams
- platform behavior on both Unix sockets and Windows named pipes

## Plugin Crypto API

Plugins should not read the owner keystore directly and should not receive owner secret keys.

Instead, the host should expose crypto operations to plugins.

Initial API surface:

- `crypto.get_identity`
  - returns `owner_id`
  - returns `signing_public_key`
  - returns `encryption_public_key`
  - returns `node_id`

- `crypto.seal`
  - host signs and encrypts for a recipient using the local owner keys
  - returns a `SignedEncryptedEnvelope`

- `crypto.open`
  - host decrypts and verifies an incoming `SignedEncryptedEnvelope`
  - returns the verified `OpenedMessage`

This keeps owner secret-key operations inside the host process while still allowing plugins to use the signed+encrypted message primitives added by the owner keystore work.

## Inference Plugin Testing

When building and validating inference plugins, create an Ollama provider plugin first.

The purpose of the Ollama provider plugin is to validate the inference endpoint registration model end to end:

- plugin registers an inference endpoint with `mesh-llm`
- plugin reports endpoint health without becoming disabled when the endpoint is temporarily unavailable
- `mesh-llm` talks directly to the Ollama OpenAI-compatible endpoint rather than proxying inference through the plugin
- model discovery and routing work through the registered endpoint
- endpoint recovery makes the provider available again automatically

This should be the first concrete inference-plugin test target before building more specialized inference providers.
