# Managed provider models

Select a plugin under `[[models]]` to configure its models alongside built-in
models. Mesh sends each plugin only the selections naming that provider, skips
those entries during native Skippy startup, and supplies an asset service that
uses the host's Hugging Face credentials and shared cache.

For example, the ONNX provider accepts this pinned Indra configuration:

```toml
[[plugin]]
name = "onnx"
command = "/absolute/path/to/mesh-onnx"

[[models]]
model = "datalab-to/indra-ocr@v1.1.0"
provider = "onnx"
variant = "fp32"

[models.options]
threads = 4
math = false

[models.request_defaults]
dpi = 200
max_pages = 100

[models.lifecycle]
load = "on_demand"
idle_timeout_secs = 300
prefetch = true
```

The provider validates `variant`, `options` and `request_defaults` against its
implemented adapters. Built-in hardware, context and speculative-decoding
controls are rejected on provider entries. Native model defaults do not flow
into provider configuration. A missing or disabled provider fails resolution
instead of silently loading the model through Skippy.

## Asset acquisition

The Rust SDK exposes the typed selections through
`mesh_llm_plugin::models::configured_models_from_env()`. The host sets
`MESH_LLM_PLUGIN_MODELS` on the plugin process; operators configure `[[models]]`
rather than setting that environment variable themselves.

A provider declares an `AssetPlan` containing the model ID, repository,
revision and exact relative file list, then calls
`context.prepare_model_assets(&plan).await`. The SDK checks the optional
`model_assets.v1` host capability before sending the request. The host verifies
that the model belongs to that plugin and that the requested repository and
revision match its configured source. It rejects empty, duplicate or unsafe
asset paths.

The host downloads only the named files. It resolves the first file to an
immutable snapshot and uses that revision for the rest of the bundle, keeping
ONNX graphs beside external tensor data. `HF_HUB_OFFLINE=1` permits cached files
only; a missing asset is an error.

An operator can instead provide a complete local bundle:

```toml
[models.source]
local_dir = "models/indra"
```

Relative paths resolve beside the Mesh configuration file. Local bundles must
contain every planned file and cannot escape their directory through symlinks.
`local_dir` cannot be combined with remote `repo` or `revision` overrides.
Remote source overrides select compatible assets; they do not change the
adapter selected by the configured model ID.

## Loading and discovery

Mesh passes the typed lifecycle policy to the provider. The provider implements
session loading, request leases and idle eviction; the host supplies configuration,
assets and routing, rather than holding the inference engine itself.

The ONNX provider interprets `load = "startup"` as eager loading and
`load = "on_demand"` as loading on the first request. `prefetch = true` prepares
assets ahead of an on-demand load. An `idle_timeout_secs` of zero disables idle
eviction; a positive timeout releases an idle engine while retaining its cached
assets. Queued and active requests hold leases that prevent eviction.

The provider's `GET /v1/models` inventory includes `workload_class` and
`availability` per model. `loadable`, `ready` and `busy` entries remain eligible
for routing, partitioned by each registered task protocol. Unloaded therefore
does not mean undiscoverable. Mesh's public `/v1/models` remains a chat-compatible
listing and excludes these task-only entries.

See [the task contracts](document-ocr.md) for protocol labels and HTTP paths.
Register a separate endpoint binding for each supported protocol; matching a
model ID alone never grants another task's capabilities.
