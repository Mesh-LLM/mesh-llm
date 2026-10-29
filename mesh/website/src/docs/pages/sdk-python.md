---
title: Python SDK
---

# Python SDK

The `mesh-llm` package embeds a Rust-backed `Node` with async Python APIs.
Python 3.10 or newer is supported. Select `mode="client"`, `"serve"` (serve-only),
or `"combined"`. The old `Client` class and in-memory owner-key constructor have
been removed.

```bash
pip install mesh-llm
```

Use a release whose Python wrapper, generated bindings and native bridge match.
Source changes do not themselves publish a new wheel.

## Join and run inference

```python
import asyncio
from meshllm import Node

async def main() -> None:
    async with Node.create(mode="client", auto_join=True) as node:
        llms = await node.inference.list_models()
        if not llms:
            raise RuntimeError("No LLM is currently available")
        response = await node.inference.chat_completions({
            "model": llms[0].id,
            "messages": [{"role": "user", "content": "Hello"}],
        })
        print(response["choices"][0]["message"])

asyncio.run(main())
```

For a private mesh, use `join_tokens=(token,)` instead of `auto_join=True`.
Pass `owner_key_path` when an owner keystore is required. To serve local models,
select `mode="serve"` or `"combined"` and pass
`models=("publisher/model:quant",)`. Local serving requires a compatible installed
native runtime. Client mode does not load a model-serving runtime. A serve-only
node rejects SDK inference calls.

## Discover every advertised workload

`node.models` queries the node's primary mesh inventory in process, including
local and remote non-LLM models. Existing `node.inference.list_models()` remains
the LLM-only compatibility list.

```python
page = await node.models.list()
ocr = await node.models.list(workload_class="document_ocr", available=True)
details = await node.models.get("org/model@revision#profile")
if page.next_cursor is not None:
    following = await node.models.list(cursor=page.next_cursor)
```

Results are `MeshModelPage`, `MeshModel`, and `MeshModelProvider` dataclasses.
Filters combine with AND. Preserve filters while continuing a cursor; restart
the list when the snapshot expires. Workload strings preserve future values.
Provider runtime limits, capabilities and optional thinking/payment observations
remain separate; absent facts are unknown. See
[Mesh model discovery](/docs/pages/model-discovery/) for HTTP shapes and migration.

## Protocol-preserving agent requests

`node.inference.chat_completions(body)` and `responses(body)` send the complete
JSON object and return the complete provider response. Tools, tool results,
multimodal parts, structured outputs, reasoning, usage and future fields are not
reduced to a text-only SDK object.

```python
raw = await node.inference.request(
    "/v1/chat/completions",
    {"model": "llm-model", "messages": messages, "stream": False},
)
payload = raw.json()
```

Streaming methods yield complete SSE events:

```python
from meshllm import OpenAIStreamChunk

async for event in node.inference.stream_chat_completions({
    "model": "llm-model", "messages": messages, "tools": tools,
}):
    if isinstance(event, OpenAIStreamChunk) and not event.done:
        payload = event.json()
        # Inspect text, tool-call deltas, usage and provider extensions.
```

Use `stream_responses` for the Responses API. Named events, their JSON data,
and original frames remain available. Closing an iterator early cancels the
native request. Native bridge work runs off the asyncio event-loop thread.
Non-success inference responses raise `OpenAIRequestError` with status and
body; raw request callers can disable automatic status checking.

## Document OCR

Discover an available `document_ocr` model through `node.models`, then send its
exact ID with the document bytes:

```python
from meshllm import DocumentOcrOptions

response = await node.inference.document_ocr(
    pdf_bytes,
    DocumentOcrOptions(
        model="ocr-model", filename="scan.pdf", content_type="application/pdf",
        response_format="markdown", max_pages=2,
    ),
)
markdown = response.body
```

Uploads are multipart and preserve binary bytes. Files must be nonempty and
at most 64 MiB. Optional `max_pages` must be positive; output formats are `json`,
`text`, and `markdown`. The result preserves `status_code`, `content_type`, and
`body`, with `.json()` available for JSON. Provider errors raise
`OpenAIRequestError` by default; `raise_for_status=False` returns their status
and body for application handling.

## Memberships and live transcription

```python
from meshllm import MeshMembershipConfig

team = await node.add_membership(MeshMembershipConfig(
    name="team", api_port=9338,
    join_tokens=(team_token,), trusted_owners=(team_owner,),
))
memberships = await node.memberships()
removed = await node.remove_membership("team")
```

Additional memberships share loaded models while retaining their own mesh
identity, allowlist, peers and API port. Native `node.models` discovery remains
scoped to the primary mesh.

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

`send_audio` accepts signed 16-bit mono PCM; `send_audio_f32` accepts floating
point samples. Defaults use 16 kHz and server-side turn detection. Setting
`turn_detection=None` makes turns explicit with `commit`. Read individual
events with `next_event` or iterate `events()`. `stop` returns remaining final
events; `close` abandons the session.

## Native runtime management

Client mode needs no model-serving runtime. A serving application explicitly
installs or resolves a matching runtime before starting its node. This offline
example uses application-owned `app_cache_dir` and bundled `runtime_bundle_dir`:

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

Installation supports an `on_progress` callback with typed download progress.
Explicit online installation can opt into `allow_download=True`; node startup
does not install implicitly. Resolution defaults to downloads disabled, and
validation always disables downloads. Results are typed runtime/outcome records.

Use `remove_native_runtime(mesh_version, native_runtime_id, cache_dir=...)` to
remove a selected cached runtime. `prune_native_runtimes(cache_dir=...,
active_mesh_version=...)` defaults to `keep_active_and_previous`, with
`active_only` available when requested.

## Source and native bindings

The public Python wrapper, generated UniFFI bindings and component tests are
maintained in [Mesh-LLM/mesh-llm-python-sdk](https://github.com/Mesh-LLM/mesh-llm-python-sdk).
Mesh admits an immutable SDK source checkpoint for its compatibility workloads.
Generation requires explicit absolute paths to Mesh source, the existing
UniFFI 0.32.0 generator, and an already built native library:

```bash
"$SDK_SOURCE/sdk/scripts/generate-python-bindings.sh" "$MESH_SOURCE" "$UNIFFI_BINDGEN" "$PREBUILT_MESH_FFI_LIBRARY"
"$SDK_SOURCE/sdk/scripts/build-native.sh" "$PREBUILT_MESH_FFI_LIBRARY"
python3 -I -m pip install -e "$SDK_SOURCE/sdk"
```

The stager never compiles a missing native bridge. Generate from the exact
library's metadata so the wrapper checksums match the packaged artifact.
Catalog queries are native; compatibility inference and some lifecycle queries
still use the embedded listener. See [SDK overview](/docs/pages/sdk/).
