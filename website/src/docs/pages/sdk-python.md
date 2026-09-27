---
title: Python SDK
---

# Python SDK

Use the `mesh-llm` package to embed a private Mesh client in Python services and agent runtimes. The SDK calls the Rust core through a generated UniFFI binding, so the Mesh transport and identity stay in-process—there is no loopback HTTP sidecar to install or supervise.

## Install

```bash
pip install mesh-llm
```

Release wheels contain the generated binding and a native library for the wheel's target platform. Python 3.10 or newer is supported.

For a repository checkout:

```bash
sdk/python/scripts/generate-python-bindings.sh
sdk/python/scripts/build-native.sh
python3 -m pip install -e sdk/python
```

## Connect to a mesh

```python
import asyncio
import os

from meshllm import Client, generate_owner_keypair_hex


async def main() -> None:
    owner = generate_owner_keypair_hex()
    # Public Mesh: discover and connect to the best published mesh.
    client = await Client.connect_public(owner_keypair_hex=owner)
    # Private Mesh instead:
    # client = Client.create(
    #     owner_keypair_hex=owner,
    #     invite_token=os.environ["MESH_INVITE_TOKEN"],
    # )
    async with client:
        models = await client.inference.list_models()
        if not models:
            raise RuntimeError("The selected Mesh has no available models")
        response = await client.inference.chat_completions({
            "model": models[0].id,
            "messages": [{"role": "user", "content": "Say hello from Python."}],
        })
        print(response["choices"][0]["message"])


asyncio.run(main())
```

Persist the owner keypair in the host application's secure storage. Generating one during every startup creates a new Mesh identity and is suitable only for examples.

`Client.connect_public()` uses Nostr discovery and selects the best matching
published Mesh. Pass `model`, `region`, `target_name`, or custom `relays` to
narrow discovery. `Client.create()` connects directly to a specific public or
private Mesh using its invite token.

Each model returned by `list_models()` includes `context_length` when the Mesh
advertises its actual served window. Agent runtimes should budget against that
value rather than a model architecture's theoretical maximum. Legacy servers
that omit served-context metadata return `None`.

## Agent requests

`chat_completions()` and `responses()` use the protocol-preserving request path. The SDK sends the complete JSON object through Mesh and returns the complete OpenAI-compatible response, instead of converting it to a text-only SDK model. Their streaming counterparts preserve complete SSE events the same way.

This is the recommended path for Hermes and other agents because it preserves:

- tool definitions, `tool_choice`, assistant tool calls, and tool results;
- multipart text, image, audio, and file content supported by the selected model;
- structured-output and JSON-schema settings;
- finish reasons, usage, log probabilities, reasoning fields, and future JSON additions.

```python
response = await client.inference.chat_completions({
    "model": "Qwen3-8B",
    "messages": [{"role": "user", "content": "What is the weather in Sydney?"}],
    "tools": [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }],
})

message = response["choices"][0]["message"]
for call in message.get("tool_calls", []):
    print(call["function"]["name"], call["function"]["arguments"])
```

For an endpoint not covered by a convenience method, call an OpenAI-compatible path directly:

```python
raw = await client.inference.request(
    "/v1/chat/completions",
    {"model": "Qwen3-8B", "messages": messages, "stream": False},
)
payload = raw.json()
```

## Stream agent events

`stream_chat_completions()` yields complete SSE events, including incremental tool-call arguments, reasoning, text, usage, finish reasons, and provider extensions. The SDK does not flatten the event into a text token, so new OpenAI-compatible fields remain available without an SDK release.

```python
from meshllm import OpenAIStreamChunk

arguments = ""
async for event in client.inference.stream_chat_completions({
    "model": "Qwen3-8B",
    "messages": [{"role": "user", "content": "What is the weather in Sydney?"}],
    "tools": tools,
}):
    if not isinstance(event, OpenAIStreamChunk) or event.done:
        continue

    chunk = event.json()
    for choice in chunk.get("choices", []):
        delta = choice.get("delta", {})
        if content := delta.get("content"):
            print(content, end="", flush=True)
        for call in delta.get("tool_calls", []):
            arguments += call.get("function", {}).get("arguments", "")

print(arguments)
```

For the Responses API, use `stream_responses()`. Its named SSE event is available as `event.event`, its `data:` payload through `event.json()`, and the exact original frame through `event.raw`. A `[DONE]` sentinel has `event.done == True`.

Closing either iterator early cancels the native request and interrupts a blocked transport read. Network and bridge work runs off the asyncio event-loop thread.

## Embed a node

`Node` shares the same lifecycle and inference API:

```python
from meshllm import Node

node = Node.create(
    owner_keypair_hex=owner_keypair,
    invite_token=invite_token,
    cache_dir=cache_dir,
    runtime_dir=runtime_dir,
    serving_enabled=True,
)

async with node:
    response = await node.inference.responses({
        "model": "local-model",
        "input": "Summarize this document.",
    })
```

Local serving requires a Python wheel built with the native bridge's `embedded-runtime` feature plus a compatible native runtime artifact. Client-only wheels are smaller and are the preferred shape for a Hermes private-compute provider.

## Errors and status

`status()` reports Mesh connection state and peer count. Non-2xx OpenAI-compatible responses raise `OpenAIRequestError`, which includes `status_code` and the original response body. Invalid identities, invite tokens, join failures, and transport failures surface from the native bridge as Python exceptions.
