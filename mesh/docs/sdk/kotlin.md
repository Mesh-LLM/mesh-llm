# Kotlin/Android SDK

## Read local runtime capacity

For a started serving node and an exact local serving ID, within a coroutine:

```kotlin
node.runtimeCapacity("exact-serving-id")?.let { capacity ->
    println("${capacity.executionLanes} lanes, ${capacity.contextTokensPerLane} tokens per lane")
    println("${capacity.contextPoolTokens} pooled tokens, observed at ${capacity.capturedAtUnixMs}")
}
```

`LaneCapacity` fields use `ULong`. Each call reads the latest published
observation without HTTP. Busy lanes count toward capacity; this does not reserve
resources. `null` means unknown, unsupported, unavailable, or incomplete capacity.
Stopped nodes throw. The timestamp is the oldest observation in Unix milliseconds;
callers choose a freshness policy for long-running inference. Do not use the total
pool as a per-request context limit.

Install `ai.meshllm:meshllm-android:<version>` from GitHub Packages. Configure
the repository and credentials in Gradle:

```kotlin
repositories {
    maven {
        url = uri("https://maven.pkg.github.com/Mesh-LLM/mesh-llm")
        credentials {
            username = providers.gradleProperty("gpr.user")
                .orElse(System.getenv("GITHUB_ACTOR")).get()
            password = providers.gradleProperty("gpr.key")
                .orElse(System.getenv("GITHUB_TOKEN")).get()
        }
    }
}
```

The Kotlin package exposes one embedded `Node` with `CLIENT`, `SERVE`, and
`COMBINED` roles. `SERVE` is serve-only.

```kotlin
val node = Node(mode = NodeMode.CLIENT, autoJoin = true)
node.start()
val models = node.inference.listModels()
val response = node.inference.chatCompletions(
    Json.parseToJsonElement("""{"model":"${models.first().id}","messages":[{"role":"user","content":"Hello"}]}""").jsonObject
)
node.stop()
```

Use `joinTokens = listOf(token)` for a private mesh. To serve local models,
choose `SERVE` or `COMBINED` and set `models = listOf(modelRef)`. Serving needs a
matching native runtime. `ownerKeyPath` accepts a Mesh LLM keystore path; the
old in-memory owner keypair constructor and `Client` class are removed.

`node.inference.streamChatCompletions(bodyJson)` and `streamResponses(bodyJson)`
return SSE flows. See [runtime artifacts](../SDK.md#native-runtime-artifacts).

## Thinking effort levels

`listModels()` returns each model with the effort strings the provider observed
in its loaded chat template:

```kotlin
val model = node.inference.listModels()
    .firstOrNull { it.thinkingEfforts?.contains("high") == true }
if (model != null) {
    val response = node.inference.chatCompletions(
        Json.parseToJsonElement(
            """{"model":"${model.id}","messages":[{"role":"user","content":"Explain lock ordering."}],"reasoning_effort":"high"}"""
        ).jsonObject
    )
}
```

`thinkingEfforts` is `null` when the provider is older or has not probed the
model, and empty when the template exposes no effort levels. The list is an
observation of the template, not a guarantee that generation obeys it.

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
Runtime limits, capabilities, thinking controls and plugin requirements belong to
their provider. `node.inference.listModels()` stays the LLM-only compatibility
view. Full discovery does not call the OpenAI model endpoint.

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

## Live transcription

The node must have a local speech model loaded. Embedded transcription uses
typed in-process audio/event channels; remote-only models fail at startup.
Closing a session, dropping either native half, or shutting down the node
cancels the backend. Explicit Rust network sessions remain available through
`LiveTranscription::start` when mesh relaying is needed.

```kotlin
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.withContext

val session = node.startLiveTranscription(LiveTranscriptionOptions(
    model = "speech-model", sampleRate = 16000u, turnDetection = null,
    stepMs = 400u, tentative = true))
try {
    session.sendAudio(pcm16Samples)
    session.commit()
    println(session.nextEvent())
    val remainingEvents = session.stop()
    println(remainingEvents)
} finally {
    withContext(NonCancellable) { session.close() }
}
```

Supply signed 16-bit mono samples, or use `sendAudioF32` for float samples.
`stepMs` accepts 200 to 5000 ms; `null` keeps the node's 1000 ms default.
Shorter steps run more transcription passes. `tentative` defaults to `false`.

Confirmed deltas are append-only. Recognition errors in confirmed text cannot
be corrected by later passes or the completed event. Shorter transcription
intervals do not guarantee better recognition accuracy.

Native transcription failures end the session after an error event. Start a
new session to retry. Request-validation errors may leave the session open.

Each `tentative` event replaces the previous unconfirmed text for its `itemId`;
empty text clears it. Append `delta` text to the confirmed transcript and let
`completed` replace the whole turn, including tentative text. Read events
concurrently with microphone input. `stop()` finishes the turn and returns
unread events; `close()` abandons the session.
Cancelling `events()` also closes the session without waiting for a pending
native read; native cleanup finishes in the background.
## Structured inference failures

`OpenAIRequestException.details` and `OpenAIStreamException.details` expose
typed `InferenceErrorDetails` with `message`, `type`, `code` and `param`:

```kotlin
try {
    node.inference.chatCompletions(request)
} catch (error: OpenAIRequestException) {
    if (error.details?.code == "context_length_exceeded") {
        // Compact the conversation before retrying.
    } else {
        throw error
    }
}
```

Inspect stream failures the same way. The original status and body remain
available. `details` is `null` for transport failures or an unstructured body;
missing optional fields stay `null`. Unknown codes are preserved.
