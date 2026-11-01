# MeshLLM Kotlin SDK

The package exposes one real embedded `Node` with client, serve-only, and
combined roles. The legacy thin `Client` is removed. See the
[kotlin usage guide](../../docs/sdk/kotlin.md) for current constructors,
streaming, role selection, and native runtime requirements.

The `serve` role serves local models; `combined` also permits mesh inference.
Client mode starts no native serving runtime. Supply a Mesh LLM owner keystore
path through `ownerKeyPath`/`owner_key_path` if owner identity is required.

Discover every workload advertised in the node's mesh through the native API:

```kotlin
val page = node.models.list()
val ocr = node.models.list(MeshModelQuery(workloadClass = "document_ocr", available = true))
val model = node.models.get("org/model@revision#profile")
page.nextCursor?.let { cursor ->
    val next = node.models.list(MeshModelQuery(cursor = cursor))
}
```

Results use typed model, provider, runtime and page objects. Retain the same
filters when continuing a page. Unrecognized peer workloads become `unknown`;
the original peer string is not preserved; optional metadata remains absent
when unknown. Open-ended capabilities,
thinking, payment and plugin metadata uses `JsonElement`. Runtime observations
apply to their provider. `inference.listModels()` remains the LLM-only
compatibility view. Full model discovery queries the embedded node directly.

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
