package ai.meshllm

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long

/** An additional mesh with an independent owner allowlist and local API port. */
data class MeshMembershipConfig(
    val name: String,
    val joinTokens: List<String> = emptyList(),
    val trustedOwners: List<String> = emptyList(),
    val apiPort: UShort,
    val nodeKeyPath: String? = null,
    val enumerateHost: Boolean = false,
) {
    internal fun json(): String = JsonObject(mapOf(
        "name" to JsonPrimitive(name),
        "join_tokens" to JsonArray(joinTokens.map(::JsonPrimitive)),
        "trusted_owners" to JsonArray(trustedOwners.map(::JsonPrimitive)),
        "api_port" to JsonPrimitive(apiPort.toInt()),
        "node_key_path" to (nodeKeyPath?.let(::JsonPrimitive) ?: JsonNull),
        "enumerate_host" to JsonPrimitive(enumerateHost),
    )).toString()
}

data class MeshMembershipStatus(
    val name: String,
    val apiBaseUrl: String,
    val inviteToken: String,
    val endpointId: String,
    val peers: Long,
    val joinError: String?,
) {
    companion object {
        internal fun decode(value: JsonObject) = MeshMembershipStatus(
            name = value.getValue("name").jsonPrimitive.content,
            apiBaseUrl = value.getValue("api_base_url").jsonPrimitive.content,
            inviteToken = value.getValue("invite_token").jsonPrimitive.content,
            endpointId = value.getValue("endpoint_id").jsonPrimitive.content,
            peers = value.getValue("peers").jsonPrimitive.long,
            joinError = value["join_error"]?.jsonPrimitive?.contentOrNull,
        )
    }
}

internal class Memberships(
    private val addJson: suspend (String) -> String,
    private val removeNative: suspend (String) -> Boolean,
    private val listJson: suspend () -> String,
) {
    suspend fun add(config: MeshMembershipConfig): MeshMembershipStatus =
        MeshMembershipStatus.decode(Json.parseToJsonElement(addJson(config.json())).jsonObject)

    suspend fun remove(name: String): Boolean = removeNative(name)

    suspend fun list(): List<MeshMembershipStatus> =
        Json.parseToJsonElement(listJson()).jsonArray.map { MeshMembershipStatus.decode(it.jsonObject) }
}
