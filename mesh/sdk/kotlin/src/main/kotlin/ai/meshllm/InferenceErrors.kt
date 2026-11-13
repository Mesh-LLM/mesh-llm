package ai.meshllm

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

/** Backend error fields, preserved independently of display text. */
data class InferenceErrorDetails(
    val message: String,
    val type: String,
    val code: String? = null,
    val param: String? = null,
) {
    companion object {
        internal fun decode(body: String?): InferenceErrorDetails? {
            val envelope = body?.let { runCatching { Json.parseToJsonElement(it) }.getOrNull() }
                as? JsonObject ?: return null
            val error = envelope["error"] as? JsonObject ?: return null
            fun string(key: String) = (error[key] as? JsonPrimitive)?.takeIf { it.isString }?.content
            val message = string("message") ?: return null
            val type = string("type") ?: return null
            for (key in listOf("code", "param")) {
                val value = error[key]
                if (value != null && value != JsonNull && string(key) == null) return null
            }
            return InferenceErrorDetails(message, type, string("code"), string("param"))
        }
    }
}
