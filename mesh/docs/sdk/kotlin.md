# Kotlin/Android SDK

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
