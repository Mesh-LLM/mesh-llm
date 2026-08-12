# Experimental Apple runtime

This directory contains one macOS Swift sidecar for MeshLLM's Apple-native
model integrations. Its first logical model is Apple's on-device system model,
exposed as `apple/system`. Named Core AI models will be added to the same
runtime and protocol rather than shipped as a second sidecar.

This work is **experimental**. A macOS `mesh-llm serve` host can supervise the
runtime and expose `apple/system` through its normal local OpenAI frontend. A
release-shaped macOS product can now embed and auto-discover the provider; the
provider is not enabled in published releases, model gossip, or private-mesh
routing yet. It implements the Milestone 0 evidence, local REST vertical slice,
Rust host-supervision layer, and CLI product-packaging layer from
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
    "variant": "system-default-27.0",
    "modelVersion": "27.0",
    "versionSource": "apple_os_release_band",
    "versionedModelID": "apple/system@27.0",
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
    "variant": "system-default-27.0",
    "model_version": "27.0",
    "version_source": "apple_os_release_band",
    "resolved_model": "apple/system@27.0",
    "capabilities": ["guided_generation", "tool_calling", "vision"]
  }, {
    "id": "apple/system@27.0",
    "object": "model",
    "owned_by": "apple",
    "alias_of": "apple/system",
    "model_version": "27.0",
    "version_source": "apple_os_release_band"
  }]
}
```

`apple/system` follows the system model installed by Apple. The versioned ID
matches only the documented 27.0 generation; it is not an immutable checkpoint
and MeshLLM cannot install or roll back it. Apple exposes no public checkpoint
or model-build identifier. Unknown future OS generations remain unversioned
until Apple publishes their release-band mapping.

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
tool, client-disconnect cancellation, slot reuse after cancellation, and a
completion addressed specifically to the resolved model generation.

### 9. Exercise the MeshLLM host supervisor

```bash
just apple::mesh
```

This builds an ad-hoc-signed local provider package and the normal dynamic Rust
host, starts `mesh-llm serve` with an isolated config, waits for `apple/system`
on the host's ordinary `/v1/models`, and sends the same completion, SSE, tool,
and cancellation probes through MeshLLM. It also verifies the provider process
in `/api/runtime/processes`, kills the child to prove target withdrawal and
restart, and terminates the host to prove child cleanup.

Captured from the Golden Gate host through MeshLLM's REST API:

```json
{
  "status": "pass",
  "model": "apple/system",
  "versioned_model": "apple/system@27.0",
  "completion_content": "apple runtime REST ready",
  "tool_executions": [{
    "name": "mesh_fixture_lookup",
    "arguments": {"key": "rest-demo"},
    "result": "mesh-fixture-value-for-rest-demo"
  }],
  "stream_done": true,
  "client_disconnect_cancelled": true,
  "provider_restarted_after_crash": true,
  "provider_exited_with_meshllm": true
}
```

The recipe sets `MESH_LLM_APPLE_PROVIDER_ALLOW_AD_HOC=1` only for this local QA
artifact. Product builds must use a trusted signature. Manual host testing can
select artifacts and policy with:

- `MESH_LLM_PROVIDER_RUNTIME_BUNDLE_DIR` for one or more carrier roots;
- `MESH_LLM_PROVIDER_RUNTIME_INDEX` plus
  `MESH_LLM_PROVIDER_RUNTIME_DOWNLOAD=1` for an opt-in release index;
- `MESH_LLM_PROVIDER_RUNTIME_CACHE_DIR` for an isolated immutable cache; and
- `MESH_LLM_APPLE_PROVIDER_ALLOW_AD_HOC=1` only for local development.

## Other validation commands

```bash
just apple::live
just apple::contract
just apple::mesh
just apple::rust-sdk
just apple::carriers
just apple::launchd
just apple::instruments
just apple::orphan
```

`just apple::instruments` writes unencrypted prompts and responses into ignored
files under `target/apple-runtime/instruments/`. Only its aggregate
`summary.json` is suitable for sharing.

`just apple::contract` verifies `provider-runtime.json`, every declared file
digest, and the executable bit through the shared Rust provider-runtime crate.

`just apple::rust-sdk` starts a provider-only embedded MeshLLM node through the
public Rust SDK builder. It supplies the carrier root through typed
configuration while setting invalid process discovery variables, then proves
`apple/system@27.0`, completion, and tool execution. This path intentionally
does not install or load a Skippy runtime.

## Packaging and signing

`just apple::package` signs ad hoc by default. To use a local signing identity:

```bash
MESH_APPLE_RUNTIME_CODESIGN_IDENTITY="Mesh-LLM Local Codesign" \
  just apple::package
```

Build and exercise a release-shaped CLI product on Golden Gate:

```bash
just apple::product-qa 0.72.1 target/apple-runtime/product
```

Use `just apple::product 0.72.1 target/apple-runtime/product` when only the
composed archives are needed; `product-qa` already performs that build before
running the smoke.

The product contains the host, one native Metal runtime, and the same Apple
sidecar under
`provider-runtimes/apple/meshllm-apple-runtime-darwin-arm64`. Product QA runs
the host from that layout with both provider bundle and provider index
overrides unset, then repeats the completion, streaming, tool, cancellation,
restart, and shutdown checks. The resulting
`target/apple-runtime/product-qa/summary.json` records
`provider_discovery=adjacent_product_bundle` and
`provider_bundle_override_used=false`.

The public release lane is intentionally stricter:

```bash
MESH_APPLE_RUNTIME_CODESIGN_IDENTITY="Developer ID Application: Example (TEAMID)" \
MESH_APPLE_RUNTIME_NOTARY_PROFILE=mesh-llm-notary \
  just apple::release-product \
    v0.72.1 0.1.0 \
    https://github.com/Mesh-LLM/mesh-llm/releases/download/v0.72.1/meshllm-apple-runtime-darwin-arm64.zip \
    dist
```

This signs with hardened runtime and a secure timestamp, submits the exact ZIP
to Apple's notary service, requires an `Accepted` result, checks it with
`spctl`, and only then composes the provider into the product. It requires a
Developer ID Application certificate and a `notarytool` keychain profile. The
notarization submission ID/status are written to
`target/apple-runtime/package/notarization.json`; credentials are never written
to the artifact.

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
| 2 | All host-capable macOS SDKs drive the same runtime lifecycle | CLI and Rust SDK carriers implemented; Swift, Node/Electron, and Kotlin/JVM pending |
| 3 | Private-mesh routing, load, failover, affinity, and withdrawal | not implemented |

This runtime does not alter the Skippy ABI or use Skippy stage execution.
