package ai.meshllm

import io.mockk.every
import io.mockk.just
import io.mockk.mockk
import io.mockk.runs
import io.mockk.slot
import io.mockk.verify
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Test
import uniffi.mesh_ffi.EventDto
import uniffi.mesh_ffi.EventListener as FfiEventListener
import uniffi.mesh_ffi.MeshClientHandleInterface

@OptIn(ExperimentalCoroutinesApi::class)
class MeshClientTest {

    private val simpleRequest = ChatRequest(
        model = "test-model",
        messages = listOf(ChatMessage(role = "user", content = "hi")),
    )

    @Test
    fun chatFlowCancellationCallsCancelWithRequestId() = runTest {
        val handle = mockk<MeshClientHandleInterface>()
        val requestIdStr = "req-cancel-123"

        every { handle.chat(any(), any()) } returns requestIdStr
        every { handle.cancel(requestIdStr) } just runs

        val client = MeshClient(handle)
        val job = launch { client.chatFlow(simpleRequest).collect {} }

        advanceUntilIdle()
        job.cancel()
        advanceUntilIdle()

        verify { handle.cancel(requestIdStr) }
    }

    @Test
    fun chatFlowEmitsEventsInOrder() = runTest {
        val handle = mockk<MeshClientHandleInterface>()
        val capturedListener = slot<FfiEventListener>()
        val requestIdStr = "req-order-456"

        every { handle.chat(any(), capture(capturedListener)) } answers {
            capturedListener.captured.onEvent(EventDto.Connecting)
            capturedListener.captured.onEvent(EventDto.Joined("node-abc"))
            capturedListener.captured.onEvent(EventDto.TokenDelta(requestIdStr, "hello "))
            capturedListener.captured.onEvent(EventDto.Completed(requestIdStr))
            requestIdStr
        }
        every { handle.cancel(requestIdStr) } just runs

        val client = MeshClient(handle)
        val events = client.chatFlow(simpleRequest).take(4).toList()

        assertEquals(Event.Connecting, events[0])
        assertEquals(Event.Joined("node-abc"), events[1])
        assertEquals(Event.TokenDelta(RequestId(requestIdStr), "hello "), events[2])
        assertEquals(Event.Completed(RequestId(requestIdStr)), events[3])
    }

    @Test
    fun chatFlowClosesOnCompletedEventWithoutCancelling() = runTest {
        val handle = mockk<MeshClientHandleInterface>()
        val capturedListener = slot<FfiEventListener>()
        val requestIdStr = "req-finish-789"

        every { handle.chat(any(), capture(capturedListener)) } answers {
            capturedListener.captured.onEvent(EventDto.TokenDelta(requestIdStr, "done"))
            capturedListener.captured.onEvent(EventDto.Completed(requestIdStr))
            requestIdStr
        }
        every { handle.cancel(any()) } just runs

        val client = MeshClient(handle)
        val events = client.chatFlow(simpleRequest).toList()

        assertEquals(
            listOf(
                Event.TokenDelta(RequestId(requestIdStr), "done"),
                Event.Completed(RequestId(requestIdStr)),
            ),
            events
        )
        verify(exactly = 0) { handle.cancel(requestIdStr) }
    }
}
