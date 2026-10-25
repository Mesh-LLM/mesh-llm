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
