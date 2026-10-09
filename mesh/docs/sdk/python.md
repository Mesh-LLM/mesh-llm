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
