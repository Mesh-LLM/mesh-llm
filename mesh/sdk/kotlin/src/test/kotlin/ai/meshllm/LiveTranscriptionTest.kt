package ai.meshllm

import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class LiveTranscriptionTest {
    @Test
    fun optionsPreserveExplicitPushToTalk() {
        val options = Json.parseToJsonElement(LiveTranscriptionOptions(sampleRate = 48000u, turnDetection = null).json()).jsonObject
        assertEquals(JsonPrimitive(48000), options["sample_rate"])
        assertEquals(JsonNull, options["turn_detection"])
    }

    @Test
    fun audioAndTranscriptEventsRemainLossless() = runTest {
        val live = LiveTranscription(
            audio = { samples -> assertEquals(listOf<Short>(1, -2), samples) },
            floatAudio = { samples -> assertEquals(listOf(0.5f), samples) },
            commitNative = {},
            eventJson = { """{"type":"delta","item_id":"turn","text":"hello"}""" },
            stopJson = { """[{"type":"completed","item_id":"turn","transcript":"hello"}]""" },
            closeNative = {},
        )
        live.sendAudio(listOf(1, -2))
        live.sendAudioF32(listOf(0.5f))
        live.commit()
        assertEquals("hello", live.nextEvent()?.text)
        assertEquals("hello", live.stop().first().transcript)
        live.close()
    }

    @Test(expected = IllegalStateException::class)
    fun nativeFailuresPropagate() = runTest {
        val live = LiveTranscription(audio = { error("closed") }, floatAudio = {}, commitNative = {}, eventJson = { null }, stopJson = { "[]" }, closeNative = {})
        assertNull(live.nextEvent())
        live.sendAudio(emptyList())
    }
}
