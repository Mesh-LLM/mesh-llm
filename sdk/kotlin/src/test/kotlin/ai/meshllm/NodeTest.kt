package ai.meshllm

import io.mockk.every
import io.mockk.just
import io.mockk.mockk
import io.mockk.runs
import io.mockk.slot
import io.mockk.verify
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import uniffi.mesh_ffi.ClientEvent
import uniffi.mesh_ffi.EventListener as FfiEventListener
import uniffi.mesh_ffi.MeshNodeHandleInterface
import uniffi.mesh_ffi.OpenAiResponseNative
import uniffi.mesh_ffi.OpenAiStreamEventNative
import uniffi.mesh_ffi.OpenAiStreamListener as FfiOpenAiStreamListener

@OptIn(ExperimentalCoroutinesApi::class)
class NodeTest {
    private fun jsonObject(source: String) = Json.parseToJsonElement(source).jsonObject

    private val simpleRequest = ChatRequest(
        model = "test-model",
        messages = listOf(ChatMessage(role = "user", content = "hi")),
    )

    @Test
    fun chatFlowCancellationCallsCancelWithRequestId() = runTest {
        val handle = mockk<MeshNodeHandleInterface>()
        val requestIdStr = "req-cancel-123"

        every { handle.chat(any(), any()) } returns requestIdStr
        every { handle.cancel(requestIdStr) } just runs

        val node = Node(handle)
        val job = launch { node.inference.chatFlow(simpleRequest).collect {} }

        advanceUntilIdle()
        job.cancel()
        advanceUntilIdle()

        verify { handle.cancel(requestIdStr) }
    }

    @Test
    fun chatFlowEmitsEventsInOrder() = runTest {
        val handle = mockk<MeshNodeHandleInterface>()
        val capturedListener = slot<FfiEventListener>()
        val requestIdStr = "req-order-456"

        every { handle.chat(any(), capture(capturedListener)) } answers {
            capturedListener.captured.onEvent(ClientEvent.Connecting)
            capturedListener.captured.onEvent(ClientEvent.Joined("node-abc"))
            capturedListener.captured.onEvent(ClientEvent.TokenDelta(requestIdStr, "hello "))
            capturedListener.captured.onEvent(ClientEvent.Completed(requestIdStr))
            requestIdStr
        }
        every { handle.cancel(requestIdStr) } just runs

        val node = Node(handle)
        val events = node.inference.chatFlow(simpleRequest).take(4).toList()

        assertEquals(Event.Connecting, events[0])
        assertEquals(Event.Joined("node-abc"), events[1])
        assertEquals(Event.TokenDelta(RequestId(requestIdStr), "hello "), events[2])
        assertEquals(Event.Completed(RequestId(requestIdStr)), events[3])
    }

    @Test
    fun chatFlowClosesOnCompletedEventWithoutCancelling() = runTest {
        val handle = mockk<MeshNodeHandleInterface>()
        val capturedListener = slot<FfiEventListener>()
        val requestIdStr = "req-finish-789"

        every { handle.chat(any(), capture(capturedListener)) } answers {
            capturedListener.captured.onEvent(ClientEvent.TokenDelta(requestIdStr, "done"))
            capturedListener.captured.onEvent(ClientEvent.Completed(requestIdStr))
            requestIdStr
        }
        every { handle.cancel(any()) } just runs

        val node = Node(handle)
        val events = node.inference.chatFlow(simpleRequest).toList()

        assertEquals(
            listOf(
                Event.TokenDelta(RequestId(requestIdStr), "done"),
                Event.Completed(RequestId(requestIdStr)),
            ),
            events,
        )
        verify(exactly = 0) { handle.cancel(requestIdStr) }
    }

    @Test
    fun bufferedRequestPreservesAgentPayloadAndResponse() = runTest {
        val handle = mockk<MeshNodeHandleInterface>()
        val body = """{"model":"test","tools":[{"type":"function"}],"response_format":{"type":"json_schema"}}"""
        every { handle.openaiRequest("/v1/chat/completions", body) } returns OpenAiResponseNative(
            statusCode = 200u.toUShort(),
            contentType = "application/json",
            body = """{"choices":[{"message":{"tool_calls":[{"id":"call-1"}]}}]}""",
        )

        val response = Node(handle).inference.chatCompletions(jsonObject(body))

        assertEquals(200u.toUShort(), response.statusCode)
        assertTrue(response.body.contains("tool_calls"))
        verify { handle.openaiRequest("/v1/chat/completions", body) }
    }

    @Test
    fun responsesIsTheCanonicalRichResponsesAPI() = runTest {
        val handle = mockk<MeshNodeHandleInterface>()
        val body = """{"model":"test","input":"weather"}"""
        every { handle.openaiRequest("/v1/responses", body) } returns OpenAiResponseNative(
            statusCode = 200u.toUShort(),
            contentType = "application/json",
            body = """{"output":[{"type":"function_call","name":"weather"}]}""",
        )

        val response = Node(handle).inference.responses(jsonObject(body))

        assertEquals(
            "function_call",
            response.json().jsonObject["output"]?.jsonArray?.first()?.jsonObject?.get("type")?.jsonPrimitive?.content,
        )
        verify { handle.openaiRequest("/v1/responses", body) }
    }

    @Test
    fun openAIStreamPreservesToolCallDeltaRawFrameAndDone() = runTest {
        val handle = mockk<MeshNodeHandleInterface>()
        val listener = slot<FfiOpenAiStreamListener>()
        val requestId = "agent-request"
        val data = """{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]}}]}"""
        every { handle.openaiStream("/v1/chat/completions", any(), capture(listener)) } answers {
            listener.captured.onEvent(OpenAiStreamEventNative.Started(requestId, 200u.toUShort(), "text/event-stream"))
            listener.captured.onEvent(OpenAiStreamEventNative.Sse(requestId, null, data, "data: $data\n\n"))
            listener.captured.onEvent(OpenAiStreamEventNative.Sse(requestId, null, "[DONE]", "data: [DONE]\n\n"))
            listener.captured.onEvent(OpenAiStreamEventNative.Completed(requestId))
            requestId
        }
        every { handle.cancel(any()) } just runs

        val events = Node(handle).inference.streamChatCompletions(jsonObject("""{"model":"test"}""")).toList()

        assertEquals(3, events.size)
        val chunk = events[1] as OpenAIStreamEvent.Sse
        assertTrue(chunk.data.contains("tool_calls"))
        assertTrue(chunk.json().toString().contains("tool_calls"))
        assertTrue(chunk.raw.startsWith("data: "))
        assertTrue((events[2] as OpenAIStreamEvent.Sse).isDone)
        verify(exactly = 0) { handle.cancel(requestId) }
    }

    @Test
    fun responsesStreamPreservesNamedEvent() = runTest {
        val handle = mockk<MeshNodeHandleInterface>()
        val listener = slot<FfiOpenAiStreamListener>()
        every { handle.openaiStream("/v1/responses", any(), capture(listener)) } answers {
            listener.captured.onEvent(
                OpenAiStreamEventNative.Sse(
                    "responses-request",
                    "response.function_call_arguments.delta",
                    """{"delta":"{\\"city\\":"}""",
                    "event: response.function_call_arguments.delta\ndata: {}\n\n",
                ),
            )
            listener.captured.onEvent(OpenAiStreamEventNative.Completed("responses-request"))
            "responses-request"
        }
        every { handle.cancel(any()) } just runs

        val event = Node(handle).inference.streamResponses(jsonObject("""{"model":"test"}""")).first()

        assertEquals("response.function_call_arguments.delta", (event as OpenAIStreamEvent.Sse).event)
    }

    @Test
    fun openAIStreamFailurePreservesStatusAndBody() = runTest {
        val handle = mockk<MeshNodeHandleInterface>()
        val listener = slot<FfiOpenAiStreamListener>()
        every { handle.openaiStream(any(), any(), capture(listener)) } answers {
            listener.captured.onEvent(
                OpenAiStreamEventNative.Failed(
                    "failed-request",
                    429u.toUShort(),
                    "rate limited",
                    """{"error":"slow down"}""",
                ),
            )
            "failed-request"
        }
        every { handle.cancel(any()) } just runs

        val failure = runCatching {
            Node(handle).inference.streamChatCompletions(jsonObject("{}")).collect {}
        }.exceptionOrNull() as OpenAIStreamException

        assertEquals(429u.toUShort(), failure.statusCode)
        assertEquals("""{"error":"slow down"}""", failure.responseBody)
        assertEquals("rate limited", failure.message)
    }

    @Test
    fun closingOpenAIStreamCancelsNativeRequest() = runTest {
        val handle = mockk<MeshNodeHandleInterface>()
        val listener = slot<FfiOpenAiStreamListener>()
        every { handle.openaiStream(any(), any(), capture(listener)) } answers {
            listener.captured.onEvent(OpenAiStreamEventNative.Started("open-request", 200u.toUShort(), "text/event-stream"))
            "open-request"
        }
        every { handle.cancel("open-request") } just runs

        Node(handle).inference.streamChatCompletions(jsonObject("{}")).first()

        verify { handle.cancel("open-request") }
    }
}
