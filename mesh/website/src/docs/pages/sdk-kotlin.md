---
title: Java, Kotlin, and Android SDK
---

# Java, Kotlin, and Android SDK

Install `ai.meshllm:meshllm-android:{{ site.sdkVersion }}` from the GitHub
Packages registry at `https://maven.pkg.github.com/Mesh-LLM/mesh-llm`.
Configure the registry credentials through your Gradle properties or environment.
The Kotlin API uses coroutines and Flow and one embedded `Node` in
`CLIENT`, `SERVE` or `COMBINED` mode. `SERVE` is serve-only.

## Join and infer

```kotlin
import ai.meshllm.Node
import ai.meshllm.NodeMode
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject

val node = Node(mode = NodeMode.CLIENT, autoJoin = true)
node.start()
try {
    val llms = node.inference.listModels()
    val model = llms.firstOrNull() ?: error("No LLM is currently available")
    val body = Json.parseToJsonElement(
        """{"model":"${model.id}","messages":[{"role":"user","content":"Hello from Kotlin."}]}"""
    ).jsonObject
    println(node.inference.chatCompletions(body).json())
} finally {
    node.stop()
}
```

Use `joinTokens = listOf(token)` for a private mesh and `ownerKeyPath` for an
existing Mesh LLM owner keystore. Client mode loads no serving runtime.

## Discover advertised models

`node.models` queries the embedded mesh directly, including all advertised
LLM, OCR, speech, embedding, decision and unknown models.

```kotlin
val page = node.models.list()
val query = MeshModelQuery(workloadClass = "document_ocr", available = true, limit = 50)
val ocr = node.models.list(query)
val model = node.models.get("org/model@revision#profile")
model.providers.forEach { provider ->
    println(provider.providerId)
    println(provider.runtime)
}
ocr.nextCursor?.let { cursor ->
    val next = node.models.list(query.copy(cursor = cursor))
}
```

Results use `MeshModelPage`, `MeshModel`, `MeshModelProvider` and
`MeshModelRuntime`. Filters combine with AND; keep them when continuing a page.
Unrecognized workloads become `unknown`; the original peer string is not
preserved. Kinds are string-valued. Unknown optional
facts remain null; open-ended metadata uses `JsonElement`.

A model's operations mean at least one provider advertises each operation.
Runtime limits, capabilities, thinking controls and payment offers belong to
their provider. `node.inference.listModels()` stays the LLM-only compatibility
view. Full discovery does not call the OpenAI model endpoint.

## Streaming

`node.inference.streamChatCompletions(bodyJson)` and
`streamResponses(bodyJson)` return SSE flows. Their events preserve the event
name, data and raw frame, including tools, reasoning, structured outputs, usage
and future fields. Cancelling the flow cancels its native request.

## Serving and native runtime

Resolve a matching native runtime explicitly before starting a serving node:

```kotlin
val runtime = NativeRuntime.resolve(NativeRuntimeResolveOptions(
    artifactDir = java.io.File("/path/to/bundled/runtime"),
    allowDownload = false,
))
val servingNode = Node(mode = NodeMode.COMBINED, models = listOf("<model-reference>"))
servingNode.start()
```

Node configuration selects loaded models. Discovery does not download or load
them. Use client mode on platforms whose release artifact does not support
local serving.

## Local development

Build the AAR with `./gradlew assembleAar` from `mesh/sdk/kotlin`. Bindings
must match a prebuilt native FFI library; generation reads its metadata rather
than the legacy UDL:

```bash
export MESH_LLM_FFI_LIBRARY=/absolute/path/libmeshllm_ffi.a
mesh/sdk/kotlin/scripts/generate-kotlin-bindings.sh
```

The generator uses UniFFI 0.32.0. The same environment variable selects the
native metadata input for Gradle's binding generation task.

## Document OCR

```kotlin
val page = node.models.list(MeshModelQuery(workloadClass = "document_ocr", available = true))
val model = page.data.firstOrNull() ?: error("No OCR provider is available")
val pdfBytes = java.io.File("scan.pdf").readBytes()
val response = node.inference.documentOcr(pdfBytes,
    DocumentOcrOptions(model = model.id, filename = "scan.pdf",
        contentType = "application/pdf", responseFormat = "json", maxPages = 10))
val pages = response.json()
```

OCR uploads the original binary bytes to `/mesh/v1/documents/ocr`; an empty document
is rejected. Provider status, content type and body are preserved. OCR raises
`OpenAIRequestException` for non-2xx by default; `raiseForStatus = false`
returns the raw response. Generic `inference.request`/`requestJson` also check status by default,
matching the other SDKs. **Migration:** pass `raiseForStatus = false` to retain
the previous raw-response behavior, then optionally call
`response.errorForStatus()` later. For plain text or Markdown,
read `response.body` instead of decoding JSON.

## Additional mesh memberships

```kotlin
val team = node.addMembership(MeshMembershipConfig(name = "team",
    joinTokens = listOf(inviteToken), trustedOwners = listOf(ownerId), apiPort = 9338u))
val memberships = node.memberships()
val removed = node.removeMembership("team")
```

Memberships have independent identities, owner allowlists and peers. An empty
allowlist admits nobody. Adding an existing name replaces the membership under
the Rust core's validation/recovery rules. The native model catalog queries the
primary mesh; it never combines independent memberships implicitly.

## Live transcription

Use signed 16-bit mono PCM recorded at 16 kHz. Disable turn detection to commit
turns explicitly:

```kotlin
val session = node.startLiveTranscription(LiveTranscriptionOptions(
    model = "speech-model", sampleRate = 16000u, turnDetection = null))
try {
    session.sendAudio(pcm16Samples)
    session.commit()
    println(session.nextEvent())
    val remainingEvents = session.stop()
    println(remainingEvents)
} finally {
    session.close()
}
```

`sendAudioF32` accepts floating point samples. Defaults use 16 kHz and server-side
turn detection. `stop()` returns remaining final events; `close()` abandons the
session. Stop the node when the application exits.

## Full embedded-node configuration

`Node.fromConfig(JsonObject)` forwards the complete Rust
`EmbeddedMeshNodeConfig` schema, including relay/bind settings, publication,
trust/admission, storage, serving limits and timeout configuration.

```kotlin
val config = Json.parseToJsonElement("""
    {"mode":"client","network":{"enumerate_host":false},
     "storage":{"isolated_config":true},"startup_timeout":{"secs":30,"nanos":0}}
""").jsonObject
val node = Node.fromConfig(config)
node.start()
```

The whole object is passed unchanged: keys and enum strings use snake_case,
omitted fields use native defaults, and unknown keys are rejected by the core.
Full-config `mode` is `client`, `serve_only`, or `serve` (combined); the
full-config default is combined, matching Rust. The existing `Node(...)`
constructor keeps its client default and Kotlin convenience arguments.
