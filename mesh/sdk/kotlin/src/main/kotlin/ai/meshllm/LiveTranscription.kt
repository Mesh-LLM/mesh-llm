package ai.meshllm

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long

data class LiveTurnDetection(
    val threshold: Float = 0.5f,
    val prefixPaddingMs: UInt = 300u,
    val silenceDurationMs: UInt = 500u,
)

data class LiveTranscriptionOptions(
    val model: String? = null,
    val language: String? = null,
    val prompt: String? = null,
    val sampleRate: UInt = 16000u,
    /** null ends turns only when commit() is called. */
    val turnDetection: LiveTurnDetection? = LiveTurnDetection(),
    /** Audio between transcription passes, 200 to 5000 ms; null keeps the node's 1000 ms default. */
    val stepMs: UInt? = null,
    /** Report replaceable, unconfirmed text before confirmed deltas. Defaults to false. */
    val tentative: Boolean = false,
) {
    internal fun json(): String = JsonObject(buildMap {
        model?.let { put("model", JsonPrimitive(it)) }
        language?.let { put("language", JsonPrimitive(it)) }
        prompt?.let { put("prompt", JsonPrimitive(it)) }
        put("sample_rate", JsonPrimitive(sampleRate.toLong()))
        stepMs?.let { put("step_ms", JsonPrimitive(it.toLong())) }
        put("tentative", JsonPrimitive(tentative))
        put("turn_detection", turnDetection?.let { detection -> JsonObject(mapOf(
            "threshold" to JsonPrimitive(detection.threshold),
            "prefix_padding_ms" to JsonPrimitive(detection.prefixPaddingMs.toLong()),
            "silence_duration_ms" to JsonPrimitive(detection.silenceDurationMs.toLong()),
        )) } ?: JsonNull)
    }).toString()
}

 /**
 * Types: speech_started, speech_stopped, delta, tentative, committed, completed, error.
 * Tentative text replaces the turn's previous unconfirmed text; empty text clears it.
 * Completed supersedes tentative text. Delta text is confirmed and should be appended.
 */
data class LiveTranscriptionEvent(
    val type: String,
    val itemId: String?,
    val audioStartMs: Long?,
    val audioEndMs: Long?,
    val text: String?,
    val transcript: String?,
    val message: String?,
) {
    companion object {
        internal fun decode(value: JsonObject): LiveTranscriptionEvent = LiveTranscriptionEvent(
            type = value.getValue("type").jsonPrimitive.content,
            itemId = value["item_id"]?.jsonPrimitive?.contentOrNull,
            audioStartMs = value["audio_start_ms"]?.takeUnless { it == JsonNull }?.jsonPrimitive?.long,
            audioEndMs = value["audio_end_ms"]?.takeUnless { it == JsonNull }?.jsonPrimitive?.long,
            text = value["text"]?.jsonPrimitive?.contentOrNull,
            transcript = value["transcript"]?.jsonPrimitive?.contentOrNull,
            message = value["message"]?.jsonPrimitive?.contentOrNull,
        )
    }
}

/** Audio writes and event reads can run concurrently; stop or close when done. */
class LiveTranscription internal constructor(
    private val audio: suspend (List<Short>) -> Unit,
    private val floatAudio: suspend (List<Float>) -> Unit,
    private val commitNative: suspend () -> Unit,
    private val eventJson: suspend () -> String?,
    private val stopJson: suspend () -> String,
    private val closeNative: () -> Unit,
) {
    suspend fun sendAudio(samples: List<Short>) = audio(samples)
    suspend fun sendAudioF32(samples: List<Float>) = floatAudio(samples)
    suspend fun commit() = commitNative()
    suspend fun nextEvent(): LiveTranscriptionEvent? = eventJson()?.let {
        LiveTranscriptionEvent.decode(Json.parseToJsonElement(it).jsonObject)
    }
    suspend fun stop(): List<LiveTranscriptionEvent> = Json.parseToJsonElement(stopJson()).jsonArray.map {
        LiveTranscriptionEvent.decode(it.jsonObject)
    }
    suspend fun close(): Unit = withContext(Dispatchers.IO) { closeNative() }
    /** Emits the session events until the session closes. */
    fun events(): Flow<LiveTranscriptionEvent> = callbackFlow {
        // A blocking native read cannot be cancelled until native close returns.
        // Keep its lifetime outside the collector's structured cancellation join.
        val nativeScope = CoroutineScope(Dispatchers.IO)
        nativeScope.launch {
            try {
                while (true) { send(nextEvent() ?: break) }
                channel.close()
            } catch (error: Throwable) { channel.close(error) }
        }
        awaitClose {
            // closeNative() blocks until the native session pump drains — the
            // same wait close() performs on Dispatchers.IO — and unblocks a
            // reader parked in a pending read. Run it on IO so cancellation
            // never blocks the collector's dispatcher.
            nativeScope.launch {
                try {
                    closeNative()
                } finally {
                    nativeScope.cancel()
                }
            }
        }
    }
}
