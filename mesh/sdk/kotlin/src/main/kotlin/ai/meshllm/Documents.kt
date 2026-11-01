package ai.meshllm

import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

/** Select an OCR model and describe its multipart PDF/image upload. */
data class DocumentOcrOptions(
    val model: String,
    val filename: String = "document",
    val contentType: String = "application/octet-stream",
    val responseFormat: String? = null,
    val maxPages: Int? = null,
) {
    internal fun json(): String = JsonObject(buildMap {
        put("model", JsonPrimitive(model))
        put("filename", JsonPrimitive(filename))
        put("content_type", JsonPrimitive(contentType))
        responseFormat?.let { put("response_format", JsonPrimitive(it)) }
        maxPages?.let { put("max_pages", JsonPrimitive(it)) }
    }).toString()
}
