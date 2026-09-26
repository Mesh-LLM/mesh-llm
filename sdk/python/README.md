# MeshLLM Python SDK

The `mesh-llm` package embeds a Mesh client directly in Python. It is designed
for agent runtimes such as Hermes that need private inference without managing
a loopback HTTP sidecar.

```bash
pip install mesh-llm
```

```python
import asyncio
import os

from meshllm import Client, generate_owner_keypair_hex


async def main() -> None:
    async with Client.create(
        owner_keypair_hex=generate_owner_keypair_hex(),
        invite_token=os.environ["MESH_INVITE_TOKEN"],
    ) as client:
        response = await client.inference.chat_completions({
            "model": "Qwen3-8B",
            "messages": [{"role": "user", "content": "What is the weather?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }],
        })
        print(response["choices"][0]["message"])


asyncio.run(main())
```

`chat_completions()` and `responses()` use the protocol-preserving request
path: the SDK serializes the mapping without narrowing it to text-only fields
and returns the full response object. This is the recommended API for agents.

The typed `chat()` and `text_response()` async iterators remain available as
simple text conveniences. Their current native contract emits text deltas only;
they do not represent tool-call deltas. The protocol-preserving API currently
uses non-streaming OpenAI responses. Incremental rich SSE delivery is planned
as a follow-up contract extension.

## Building from a checkout

Generate bindings and stage the current-platform native library:

```bash
sdk/python/scripts/generate-python-bindings.sh
sdk/python/scripts/build-native.sh
python3 -m pip install -e sdk/python
```

Set `MESH_PYTHON_EMBEDDED_RUNTIME=1` before `build-native.sh` to include local
model serving. Client-only builds are smaller and are sufficient for Hermes.

The generated Python source is committed. Release wheels additionally package
the matching `libuniffi` native library beside it for macOS, Linux, or Windows.
