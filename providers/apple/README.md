# Experimental Apple runtime

This directory contains one macOS Swift sidecar for MeshLLM's Apple-native
model integrations. Its first logical model is Apple's on-device system model,
exposed as `apple/system`. Named Core AI models will be added to the same
runtime and protocol rather than shipped as a second sidecar.

This work is **experimental**. It is not connected to MeshLLM's production
OpenAI frontend, model gossip, or private-mesh routing yet. It implements the
Milestone 0 evidence and an experimental local REST vertical slice from
[issue #1246](https://github.com/Mesh-LLM/mesh-llm/issues/1246).

## Requirements

You must have all of the following:

- Apple silicon;
- **macOS Golden Gate** (macOS 27);
- full Xcode 27 selected with `xcode-select`, not Command Line Tools alone;
- Apple Intelligence enabled;
- the system model downloaded and reported as available.

Confirm the developer environment:

```bash
xcode-select -p
xcodebuild -version
xcrun --sdk macosx --show-sdk-version
```

The expected Xcode path ends in `Xcode.app/Contents/Developer` or
`Xcode-beta.app/Contents/Developer`, and the SDK must be 27.x.

## Try it locally

Run all commands from the repository root.

### 1. Build and test the runtime

```bash
just apple::build
just apple::test
```

### 2. Check system-model availability

```bash
just apple::run status
```

An eligible machine reports one logical model inside the shared runtime:

```json
{
  "runtimeID": "apple/runtime",
  "protocolVersion": "0.1",
  "models": [{
    "modelID": "apple/system",
    "providerKind": "system",
    "availability": "available",
    "contextSize": 4096,
    "variant": "system-default-unversioned",
    "capabilities": ["guided_generation", "tool_calling", "vision"]
  }]
}
```

### 3. Start the loopback REST server

```bash
just apple::run serve --port 11435
```

The server prints its bound address:

```json
{"host":"127.0.0.1","port":"11435","type":"ready"}
```

The listener is an experimental diagnostic surface. It binds through Apple's
loopback stack and should not be exposed to an untrusted network.

### 4. List models over REST

```bash
curl -s http://127.0.0.1:11435/v1/models | jq
```

Example:

```json
{
  "object": "list",
  "data": [{
    "id": "apple/system",
    "object": "model",
    "owned_by": "apple",
    "availability": "available",
    "context_length": 4096,
    "variant": "system-default-unversioned",
    "capabilities": ["guided_generation", "tool_calling", "vision"]
  }]
}
```

### 5. Run a completion over REST

```bash
curl -s http://127.0.0.1:11435/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "apple/system",
    "messages": [{
      "role": "user",
      "content": "Reply with exactly: apple runtime REST ready"
    }],
    "temperature": 0,
    "max_tokens": 32
  }' | jq
```

Captured output from the Golden Gate test machine:

```json
{
  "model": "apple/system",
  "object": "chat.completion",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "apple runtime REST ready"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 65,
    "completion_tokens": 9,
    "total_tokens": 74
  },
  "mesh_timing": {
    "elapsed_ms": 1845,
    "time_to_first_token_ms": 1752
  }
}
```

### 6. Stream a completion

```bash
curl -sN http://127.0.0.1:11435/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "apple/system",
    "messages": [{"role":"user","content":"Reply with exactly: streaming REST ready"}],
    "temperature": 0,
    "max_tokens": 32,
    "stream": true
  }'
```

Example SSE output:

```text
data: {"object":"chat.completion.chunk","model":"apple/system","choices":[{"delta":{"content":"streaming REST"},"finish_reason":null,"index":0}]}

data: {"object":"chat.completion.chunk","model":"apple/system","choices":[{"delta":{"content":" ready"},"finish_reason":null,"index":0}]}

data: {"object":"chat.completion.chunk","model":"apple/system","choices":[{"delta":{},"finish_reason":"stop","index":0}]}

data: [DONE]
```

### 7. Exercise a tool call over REST

The experimental REST surface recognizes one deterministic fixture tool so the
Foundation Models tool path can be tested without external side effects:

```bash
curl -s http://127.0.0.1:11435/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "apple/system",
    "messages": [{"role":"user","content":"Use the tool with key: rest-demo"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "mesh_fixture_lookup",
        "description": "Look up a fixture",
        "parameters": {
          "type": "object",
          "properties": {"key": {"type": "string"}},
          "required": ["key"]
        }
      }
    }]
  }' | jq
```

Captured output:

```json
{
  "model": "apple/system",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "mesh-fixture-value-for-rest-demo"
    },
    "finish_reason": "stop"
  }],
  "mesh_tool_executions": [{
    "name": "mesh_fixture_lookup",
    "arguments": {"key": "rest-demo"},
    "result": "mesh-fixture-value-for-rest-demo"
  }],
  "usage": {
    "prompt_tokens": 327,
    "completion_tokens": 13,
    "total_tokens": 340
  }
}
```

`mesh_tool_executions` is an experimental MeshLLM extension showing the
server-executed tool. Arbitrary OpenAI tool schemas are not implemented yet.

### 8. Run the automated REST smoke

```bash
just apple::rest
```

This verifies model listing, buffered completion, SSE streaming, the fixture
tool, client-disconnect cancellation, and slot reuse after cancellation.

## Other validation commands

```bash
just apple::live
just apple::carriers
just apple::launchd
just apple::instruments
just apple::orphan
```

`just apple::instruments` writes unencrypted prompts and responses into ignored
files under `target/apple-runtime/instruments/`. Only its aggregate
`summary.json` is suitable for sharing.

## Packaging and signing

`just apple::package` signs ad hoc by default. To use a local signing identity:

```bash
MESH_APPLE_RUNTIME_CODESIGN_IDENTITY="Mesh-LLM Local Codesign" \
  just apple::package
```

The background continued-processing inference entitlement is included as a
review artifact. Packaging refuses to apply it unless provisioning has been
independently validated. A locally created certificate is not sufficient:
macOS terminates that entitlement-bearing binary before `main`.

See [the Apple runtime design and evidence](../../docs/design/APPLE_RUNTIME.md)
for the entitlement result, Instruments evidence, SDK carrier boundary, quality
caveats, and rollout gates.

## Delivery status

| Phase | Deliverable | Status |
|---|---|---|
| 0 | Policy, entitlement, packaging, signing, launchd, and accelerator spike | complete |
| 1 | Local `apple/system` REST vertical slice | experimental implementation complete |
| 2 | All host-capable macOS SDKs drive the same runtime lifecycle | not implemented |
| 3 | Private-mesh routing, load, failover, affinity, and withdrawal | not implemented |

This runtime does not alter the Skippy ABI or use Skippy stage execution.
