# Kotlin/Android SDK

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
