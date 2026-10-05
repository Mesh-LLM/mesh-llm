# Skippy architecture and API boundaries

Skippy is a standalone model-serving product. MeshLLM is a separate product in
the same Cargo workspace and composes Skippy; dependencies point from Mesh to
Skippy, never the reverse. Both products share the release version and native
runtime artifacts. Each has its own CLI, configuration, and data paths; Mesh
translates those inputs into the Skippy-owned serving and KV behavior shared
by both products.

## Entry points and contracts

| Surface | Owner | Intended caller and contract |
|---|---|---|
| `skippy` CLI | `skippy-cli`, `skippy-commands` | Operators running model management, local OpenAI serving, or explicit stage workers. |
| HTTP `/v1` frontend | `skippy-inference-api` | OpenAI-compatible request/response and backend contract shared by standalone Skippy and Mesh. Mesh adds discovery, routing, and proxy policy around it. |
| Rust model lifecycle | `skippy-api::serving` | Embedding hosts load a model with `ModelLoadRequest` and retain `LoadedModelBackend` while serving. The options are currently a low-level integration contract, not a small stable SDK facade. |
| Native runtime selection | `skippy-api::native_runtime` | The caller supplies bundle and cache locations. The API neither chooses a product home directory nor downloads implicitly. |
| Package-v2 format | `skippy-package-format` | Producers and consumers share validated manifests and content identities. Format compatibility is separate from the native ABI. |
| Stage wire protocol | `skippy-protocol` | Internal, generation-gated stage traffic. Mixed generations fail closed; it is not a backwards-compatible public network API. |
| C ABI | `skippy/llama_cpp/patches`, `skippy-ffi` | Experimental version-0 capability API. The patched native header and Rust version constants advance together. |

`skippy-serving`, `skippy-runtime`, `skippy-cache`, and `skippy-scheduler` are
implementation layers. Embedding hosts may currently use their advanced types
through `ModelLoadRequest`, but those types are not a promise of a stable,
single-crate embedding SDK. A future narrower facade should accept validated
model/runtime locations and a small serving policy, then retain native resources
behind an opaque handle. Keep the advanced controls for Mesh and specialist
callers; do not duplicate the execution engine or add a crate merely to hide it.

## Ownership flow

```text
operator/client -> Skippy CLI or Mesh ingress
                -> shared OpenAI frontend / Skippy lifecycle API
                -> serving + scheduler + cache
                -> Rust runtime -> Skippy C ABI -> patched llama.cpp

model reference -> artifact/HF adapter -> package format and admission
                -> runtime load
```

Mesh owns its peer discovery, placement, identity, routing, plugins, console,
and management APIs. It translates those decisions into Skippy lifecycle,
protocol, and serving inputs. The lifecycle API takes explicit locations rather
than reading Mesh product configuration. Both CLIs use the same HF cache
preflight and fallback data roots, with `MESH_LLM_DATA_DIR` as the optional
fallback override. `HF_HUB_CACHE`,
`HUGGINGFACE_HUB_CACHE`, `HF_HOME`, `HF_XET_CACHE`, and `XDG_CACHE_HOME` still
take precedence where applicable.

`skippy-model-hf::store` uses the same application cache root for both CLIs.
Its `_in` operations take an explicit application cache root for tests and
embedding applications. The persisted `mesh_managed` usage-record field is retained
for compatibility with existing records; it does not select a product path.

Skippy's canonical HTTP request extension for enabling its serving hooks is
`skippy_hooks`. Readers also accept the older `mesh_hooks` field, and Mesh
currently writes both for mixed-version peers. If both are present, the
canonical Skippy field wins. The separate legacy `mesh_compact` and
`_mesh_respond` guardrail wire names remain unchanged; changing those is a
distinct client-compatibility decision.

The former [Skippy integration plan](SKIPPY.md) records migration history and
is not the current product/API architecture.
