package ai.meshllm.example

import ai.meshllm.Node
import ai.meshllm.NodeMode
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject

fun main(args: Array<String>) = runBlocking {
    val token = args.firstOrNull() ?: System.getenv("MESH_SDK_INVITE_TOKEN")
    val node = Node(
        mode = NodeMode.CLIENT,
        joinTokens = token?.let(::listOf) ?: emptyList(),
        autoJoin = token == null,
        ownerKeyPath = System.getenv("MESH_SDK_OWNER_KEY_PATH"),
    )
    node.start()
    try {
        val models = node.inference.listModels()
        println("[models] ${models.size}")
        if (models.isNotEmpty()) {
            val body = Json.parseToJsonElement(
                """{"model":"${models.first().id}","messages":[{"role":"user","content":"hello"}]}"""
            ).jsonObject
            val response = node.inference.chatCompletions(body)
            println(response.body)
        }
    } finally {
        node.stop()
    }
}
