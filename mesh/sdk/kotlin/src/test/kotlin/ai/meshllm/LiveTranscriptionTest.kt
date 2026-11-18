package ai.meshllm

import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
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
    /** Explicit push-to-talk options survive the native round trip. */
    @Test
    fun optionsPreserveExplicitPushToTalk() {
        val options = Json.parseToJsonElement(LiveTranscriptionOptions(model = "speech#fast", language = "de", prompt = "jargon", sampleRate = 48000u, turnDetection = null).json()).jsonObject
        assertEquals(JsonPrimitive("speech#fast"), options["model"])
        assertEquals(JsonPrimitive("de"), options["language"])
        assertEquals(JsonPrimitive("jargon"), options["prompt"])
        assertEquals(JsonPrimitive(48000), options["sample_rate"])
        assertEquals(JsonNull, options["turn_detection"])
        val defaults = Json.parseToJsonElement(LiveTranscriptionOptions().json()).jsonObject
        assertNull(defaults["step_ms"])
        assertEquals(JsonPrimitive(false), defaults["tentative"])
    }

    /** Incremental options serialize under their native keys. */
    @Test
    fun incrementalOptionsUseNativeKeys() {
        val options = Json.parseToJsonElement(LiveTranscriptionOptions(stepMs = 400u, tentative = true).json()).jsonObject
        assertEquals(JsonPrimitive(400), options["step_ms"])
        assertEquals(JsonPrimitive(true), options["tentative"])
        assertNull(options["stepMs"])
    }

    /** Tentative replacements and clear remain distinct from final text. */
    @Test
    fun tentativeReplacementsAndClearRemainDistinctFromFinalText() = runTest {
        for (text in listOf("hel", "hello", "")) {
            val live = LiveTranscription(
                audio = {}, floatAudio = {}, commitNative = {},
                eventJson = { """{"type":"tentative","item_id":"turn","text":"$text"}""" },
                stopJson = { """[{"type":"completed","item_id":"turn","transcript":"Hello!"}]""" },
                closeNative = {},
            )
            val event = live.nextEvent()
            assertEquals("tentative", event?.type)
            assertEquals("turn", event?.itemId)
            assertEquals(text, event?.text)
            assertNull(event?.transcript)
            val remaining = live.stop()
            assertEquals("completed", remaining.first().type)
            assertEquals("Hello!", remaining.first().transcript)
            live.close()
        }
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

    /** Cancelling a pending read closes on IO without blocking the collection. */
    @Test
    fun cancellingEventsDuringPendingReadClosesOnIoWithoutBlockingCollection() = runBlocking {
        val readPending = CompletableDeferred<Unit>()
        val closeStarted = CompletableDeferred<Unit>()
        val readFinished = CompletableDeferred<Unit>()
        val closeFinished = CompletableDeferred<Unit>()
        // closeNative is a plain () -> Unit, so the park must be a blocking
        // wait rather than a suspending await; the reader callback stays
        // suspending either way.
        val release = java.util.concurrent.CountDownLatch(1)
        val live = LiveTranscription(
            audio = {}, floatAudio = {}, commitNative = {},
            // The reader parks in a pending native read until close releases it.
            eventJson = {
                readPending.complete(Unit)
                try { release.await(); null }
                finally { readFinished.complete(Unit) }
            },
            stopJson = { "[]" },
            // Native close waits for the session pump, like a pending final transcript.
            closeNative = {
                closeStarted.complete(Unit)
                try { release.await() }
                finally { closeFinished.complete(Unit) }
            },
        )
        val collection = launch { live.events().collect { } }
        try {
            withTimeout(5_000) {
                readPending.await()
                // Cancellation must return even though the native close is parked:
                // the collector's dispatcher never runs the blocking cleanup.
                collection.cancelAndJoin()
                closeStarted.await()
            }
        } finally {
            // A failed cancellation assertion must still release both native calls.
            release.countDown()
        }
        withTimeout(5_000) {
            readFinished.await()
            closeFinished.await()
        }
    }
}
