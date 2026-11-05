# Python SDK source ownership

The Python package, generated UniFFI bindings, SDK tests and real OpenAI/LiteLLM/LangChain/embedding clients are owned by the separately versioned `Mesh-LLM/mesh-llm-python-sdk` project. Both extracted projects are published; fresh anonymous restore and native source admission passed for the [SDK descriptor](../../../ci/required-sdk-python/sdk-source.json) and [research descriptor](../../../ci/python-research-source.json). This source proof does not qualify hosted workflows, installed SDKs, native bridges, models or a new PyPI release.

Mesh preparation restores the exact commit in `ci/required-sdk-python/sdk-source.json` and admits its manifest and all tracked file hashes through `automation smoke-observation sdk-source`. Local preparation must set `MESH_PYTHON_SDK_SOURCE` to an absolute checkout of that pinned source. Runtime SDK execution is offline and does not resolve dependencies or compile a native bridge. The compatibility and embedding checks keep their required cadence.

SDK binding generation belongs to the external producer: pass an explicit Mesh source checkout and an existing UniFFI 0.32.0 generator. Packaging consumes an explicitly supplied, already built Mesh FFI library; it does not infer a neighboring Mesh checkout. The native bridge source/API and runtime ABI remain Mesh-owned and must match the generated binding artifact.

# Python SDK

The Python package exposes one `Node` with `client`, `serve`, and `combined`
roles. `serve` is serve-only; inference methods reject that role.

```python
from meshllm import Node

async with Node.create(mode="client", auto_join=True) as node:
    models = await node.inference.list_models()
    response = await node.inference.chat_completions({
        "model": models[0].id,
        "messages": [{"role": "user", "content": "Hello"}],
    })
```

For a private mesh, use `join_tokens=(token,)` instead of `auto_join=True`.
To serve, use `mode="serve"` or `mode="combined"` and pass
`models=("publisher/model:quant",)`. Serving requires an installed matching
native runtime. Set `owner_key_path` to a Mesh LLM owner keystore path when
owner identity is required. The old in-memory keypair hex constructor and
`Client` class have been removed.

Streaming is available through `node.inference.stream_chat_completions(...)`
and `stream_responses(...)`; each event carries the original SSE data and raw
frame. See [native runtime guidance](../SDK.md#native-runtime-artifacts).

## Thinking effort levels

`list_models()` reports the reasoning effort levels each advertised model
accepts, so an application can select a supported level before requesting
inference:

```python
models = await node.inference.list_models()
model = next(
    model for model in models
    if model.thinking_efforts and "high" in model.thinking_efforts
)
response = await node.inference.chat_completions({
    "model": model.id,
    "messages": [{"role": "user", "content": "Explain lock ordering."}],
    "reasoning_effort": "high",
})
```

`thinking_efforts` is `None` when the provider advertises no effort metadata —
an older provider or a model the host has not probed — and an empty tuple when
the provider reports that the model's template exposes no effort levels. The
list comes from the node's model catalog, which merges the mesh providers
serving that ID.

## Discover every advertised workload

`node.models` queries the primary mesh inventory directly through the native
bridge, including local and remote non-LLM providers. Existing
`node.inference.list_models()` remains the LLM-only compatibility list.

```python
page = await node.models.list()
ocr = await node.models.list(workload_class="document_ocr", available=True)
details = await node.models.get("org/model@revision#profile")
if page.next_cursor is not None:
    following = await node.models.list(cursor=page.next_cursor)
```

Results are `MeshModelPage`, `MeshModel`, and `MeshModelProvider` dataclasses.
Filters combine with AND. Keep filters unchanged when continuing a page, and
restart if its snapshot expires. Missing metadata is unknown. Unrecognized
workloads become `unknown`; the original peer string is not preserved.
Each provider retains its runtime limits, capabilities
and optional thinking/payment observations. See
[Mesh model discovery](../MODEL_DISCOVERY.md) for complete HTTP and SDK shapes.

Use a Python SDK release and native bridge containing these APIs together.
The public wrapper and generated bindings live in the companion Python SDK
repository; updating Mesh alone does not publish a new Python wheel.

## Invoke document OCR

```python
from meshllm import DocumentOcrOptions

response = await node.inference.document_ocr(
    pdf_bytes,
    DocumentOcrOptions(
        model="ocr-model",
        filename="scan.pdf",
        content_type="application/pdf",
        response_format="markdown",
        max_pages=2,
    ),
)
markdown = response.body
```

Document bytes are uploaded unchanged as multipart data. Files must be nonempty
and at most 64 MiB; `max_pages` must be positive when supplied. Formats are
`json`, `text`, or `markdown`. The response preserves `status_code`,
`content_type`, and `body`; use `.json()` for JSON output. Provider HTTP errors
raise `OpenAIRequestError` by default. Pass `raise_for_status=False` to inspect
the provider response directly.

## Native runtime management

Client mode needs no model-serving runtime. Serving applications explicitly
install or resolve one before starting the node. With application-owned
`app_cache_dir` and a bundled `runtime_bundle_dir`:

```python
from meshllm import (
    NativeRuntimeInstallOptions,
    current_mesh_version,
    current_skippy_abi_version,
    install_native_runtime,
    installed_native_runtimes,
    resolve_native_runtime,
    validate_native_runtime,
)

options = NativeRuntimeInstallOptions(
    mesh_version=current_mesh_version(),
    skippy_abi_version=current_skippy_abi_version(),
    cache_dir=str(app_cache_dir),
    bundle_dirs=(str(runtime_bundle_dir),),
    allow_download=False,
)
outcome = await install_native_runtime(options)
cached = await installed_native_runtimes(options.cache_dir)
resolved = await resolve_native_runtime(options)
validated = await validate_native_runtime(str(runtime_bundle_dir), options)
```

`install_native_runtime` can report `NativeRuntimeDownloadProgress` through its
`on_progress` callback. Online installs allow downloads when explicitly invoked
with `allow_download=True`; node startup never installs implicitly.
`resolve_native_runtime` defaults to `allow_download=False`, and validation
always disables downloads. Outcomes and runtime records are typed dataclasses.

`remove_native_runtime(mesh_version, native_runtime_id, cache_dir=...)` removes
a selected cached runtime. `prune_native_runtimes(cache_dir=...,
active_mesh_version=...)` defaults to `keep_active_and_previous`; `active_only`
is also supported. Runtime version/ABI queries use the matching native bridge.

## Additional memberships

```python
from meshllm import MeshMembershipConfig

team = await node.add_membership(MeshMembershipConfig(
    name="team",
    api_port=9338,
    join_tokens=(team_token,),
    trusted_owners=(team_owner,),
))
memberships = await node.memberships()
removed = await node.remove_membership("team")
```

Each membership has its own identity, allowlist, peers and API port and shares
the node's loaded models. `node.models` remains scoped to the primary mesh.

## Live transcription

```python
from meshllm import LiveTranscriptionOptions

session = await node.start_live_transcription(LiveTranscriptionOptions(
    model="speech-model", sample_rate=16000, turn_detection=None,
))
try:
    await session.send_audio(pcm16_samples)
    await session.commit()
    event = await session.next_event()
finally:
    remaining_events = await session.stop()
```

Use signed 16-bit mono PCM for `send_audio`, or floating point samples for
`send_audio_f32`. Defaults use 16 kHz audio and server-side turn detection;
`turn_detection=None` ends a turn on explicit `commit`. `events()` provides an
async event iterator. `stop` returns final remaining events; `close` abandons
the session. Session transport uses the embedded realtime endpoint.
