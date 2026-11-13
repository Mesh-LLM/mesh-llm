package ai.meshllm

import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class StatusTest {
    private fun snapshot(phase: String, error: String = "") = """{
        "state":{"phase":"$phase","error":"$error"}, "serving_models":["chat"],
        "models":[], "local_models":[{"identity":{"model_name":"chat"}}],
        "capacities":[{"model":"chat","context_per_lane":[4096,8192],"last_success_unix_ms":123}],
        "download":{"file":"weights","downloaded_bytes":10,"total_bytes":null}}
    """

    @Test
    fun nativeSnapshotsPreserveProgressCapacityFailureAndClosure() = runTest {
        val values = ArrayDeque(listOf(snapshot("loading"), snapshot("downloading"), snapshot("ready"), snapshot("failed", "runtime failed")))
        var closed = false
        val events = StatusEvents(nextJson = { values.removeFirstOrNull() }, closeNative = { closed = true })
        val seen = events.events().toList()
        assertEquals(listOf(NodeLifecycle.Loading, NodeLifecycle.Downloading, NodeLifecycle.Ready,
            NodeLifecycle.Failed("runtime failed")), seen.map { it.state })
        assertEquals(10L, seen.first().download?.downloadedBytes)
        assertNull(seen.first().download?.totalBytes)
        assertEquals(listOf(4096L, 8192L), seen.first().capacities.first().contextPerLane)
        assertEquals(123L, seen.first().capacities.first().lastSuccessUnixMs)
        // Native close runs on IO after the flow channel finishes.
        events.close()
        assertTrue(closed)
    }

    @Test
    fun nativeSourceErrorsPropagate() = runTest {
        val failure = IllegalStateException("native source closed")
        val events = StatusEvents(nextJson = { throw failure }, closeNative = {})
        try {
            events.events().toList()
            throw AssertionError("expected native failure")
        } catch (error: IllegalStateException) { assertEquals(failure.message, error.message) }
    }

    @Test
    fun cancellationClosesAPendingNativeReadWithoutWaitingForAnotherUpdate() = runBlocking {
        val reading = CompletableDeferred<Unit>()
        val release = CompletableDeferred<Unit>()
        val closed = CompletableDeferred<Unit>()
        val events = StatusEvents(nextJson = {
            reading.complete(Unit)
            release.await()
            null
        }, closeNative = { release.complete(Unit); closed.complete(Unit) })
        val collector = launch(Dispatchers.Default) { events.events().toList() }
        withTimeout(5000) { reading.await() }
        collector.cancelAndJoin()
        withTimeout(5000) { closed.await() }
    }
}
