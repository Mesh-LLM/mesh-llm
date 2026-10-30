package ai.meshllm

import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.withTimeout
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

    @Test
    fun cancellingEventsDuringPendingReadClosesOnIoWithoutBlockingCollection() = runTest {
        val readPending = CompletableDeferred<Unit>()
        val closeStarted = CompletableDeferred<Unit>()
        val release = CompletableDeferred<Unit>()
        val live = LiveTranscription(
            audio = {}, floatAudio = {}, commitNative = {},
            // The reader parks in a pending native read until close releases it.
            eventJson = { readPending.complete(Unit); release.await(); null },
            stopJson = { "[]" },
            // Native close waits for the session pump, like a pending final transcript.
            closeNative = { closeStarted.complete(Unit); release.await() },
        )
        val collection = launch { live.events().collect { } }
        readPending.await()
        withTimeout(5_000) {
            // Cancellation must return even though the native close is parked:
            // the collector's dispatcher never runs the blocking cleanup.
            collection.cancelAndJoin()
        }
        release.complete(Unit)
        withTimeout(5_000) { closeStarted.await() }
    }
}
