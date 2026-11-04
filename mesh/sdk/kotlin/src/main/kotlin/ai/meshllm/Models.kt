package ai.meshllm

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long

/** Filters combine with AND; omitted filters include all advertised workloads. */
data class MeshModelQuery(
    val workloadClass: String? = null,
    val kind: String? = null,
    val available: Boolean? = null,
    val limit: Int? = null,
    val cursor: String? = null,
) {
    internal fun json(): String = JsonObject(buildMap {
        workloadClass?.let { put("workload_class", JsonPrimitive(it)) }
        kind?.let { put("kind", JsonPrimitive(it)) }
        available?.let { put("available", JsonPrimitive(it)) }
        limit?.let { put("limit", JsonPrimitive(it)) }
        cursor?.let { put("cursor", JsonPrimitive(it)) }
    }).toString()
}

data class MeshModelRuntime(
    val modelName: String,
    val identityHash: String?,
    val contextLength: Long?,
    val ready: Boolean,
)

/** Observations apply to this provider, not to every replica of a model. */
data class MeshModelProvider(
    val providerId: String,
    val local: Boolean,
    val endpointId: String?,
    val plugin: String?,
    val workloadClass: String?,
    val available: Boolean,
    val operations: List<String>,
    val capabilities: JsonElement?,
    val metadata: JsonElement?,
    val runtime: MeshModelRuntime?,
    val thinking: JsonElement?,
    val requires: JsonElement?,
)

/** String-valued kinds and workloads preserve advertisements from newer peers. */
data class MeshModel(
    val id: String,
    val displayName: String,
    val kind: String,
    val workloadClass: String?,
    val availability: String,
    val operations: List<String>,
    val metadata: JsonElement?,
    val providers: List<MeshModelProvider>,
    val virtualModel: JsonElement?,
)

data class MeshModelPage(val data: List<MeshModel>, val nextCursor: String?)

/** Native mesh discovery, including LLM, speech, embedding, decision and OCR models. */
class Models internal constructor(
    private val listJson: suspend (String) -> String,
    private val modelJson: suspend (String) -> String,
) {
    suspend fun list(query: MeshModelQuery = MeshModelQuery()): MeshModelPage {
        val page = Json.parseToJsonElement(listJson(query.json())).jsonObject
        return MeshModelPage(
            page.getValue("data").jsonArray.map { decodeModel(it.jsonObject) },
            page.optionalString("next_cursor"),
        )
    }

    suspend fun get(id: String): MeshModel =
        decodeModel(Json.parseToJsonElement(modelJson(id)).jsonObject)

    private fun decodeModel(value: JsonObject): MeshModel = MeshModel(
        value.string("id"), value.string("display_name"), value.string("kind"),
        value.optionalString("workload_class"), value.string("availability"),
        value.strings("operations"), value.optional("metadata"),
        value.getValue("providers").jsonArray.map { decodeProvider(it.jsonObject) },
        value.optional("virtual_model"),
    )

    private fun decodeProvider(value: JsonObject): MeshModelProvider = MeshModelProvider(
        value.string("provider_id"), value.getValue("local").jsonPrimitive.boolean,
        value.optionalString("endpoint_id"), value.optionalString("plugin"),
        value.optionalString("workload_class"), value.getValue("available").jsonPrimitive.boolean,
        value.strings("operations"), value.optional("capabilities"), value.optional("metadata"),
        value.optional("runtime")?.jsonObject?.let { runtime ->
            MeshModelRuntime(
                runtime.string("model_name"), runtime.optionalString("identity_hash"),
                runtime.optional("context_length")?.jsonPrimitive?.long,
                runtime.getValue("ready").jsonPrimitive.boolean,
            )
        }, value.optional("thinking"), value.optional("requires"),
    )
}

private fun JsonObject.optional(name: String): JsonElement? = get(name)?.takeUnless { it == JsonNull }
private fun JsonObject.string(name: String): String = getValue(name).jsonPrimitive.content
private fun JsonObject.optionalString(name: String): String? = optional(name)?.jsonPrimitive?.contentOrNull
private fun JsonObject.strings(name: String): List<String> = getValue(name).jsonArray.map { it.jsonPrimitive.content }
