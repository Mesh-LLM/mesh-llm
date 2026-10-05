package ai.meshllm

import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import uniffi.mesh_ffi.MeshNodeHandleInterface
import uniffi.mesh_ffi.NodeStatusNative
import uniffi.mesh_ffi.OpenAiResponseNative

class NodeTest {
    @Test
    fun rolesHaveDistinctWireValues() {
        assertEquals(listOf("client", "serve", "combined"), NodeMode.entries.map { it.wireValue })
    }

    @Test
    fun statusReportsEmbeddedRoleAndEndpoint() = runTest {
        val handle = mockk<MeshNodeHandleInterface>()
        every { handle.status() } returns NodeStatusNative(
            running = true,
            mode = "combined",
            apiBaseUrl = "http://127.0.0.1:9337/v1",
            consoleUrl = "http://127.0.0.1:3131",
            payloadJson = "{}",
        )
        val status = Node(handle).status()
        assertTrue(status.running)
        assertEquals(NodeMode.COMBINED, status.mode)
        assertEquals("http://127.0.0.1:9337/v1", status.apiBaseUrl)
    }

    @Test
    fun requestPreservesAgentPayload() = runTest {
        val handle = mockk<MeshNodeHandleInterface>()
        val source = """{"model":"test","tools":[{"type":"function"}]}"""
        every { handle.openaiRequest("/v1/chat/completions", source) } returns OpenAiResponseNative(
            statusCode = 200u.toUShort(),
            contentType = "application/json",
            body = """{"choices":[{"message":{"tool_calls":[{"id":"call-1"}]}}]}""",
        )
        val body = Json.parseToJsonElement(source).jsonObject
        val response = Node(handle).inference.chatCompletions(body)
        assertEquals(200u.toUShort(), response.statusCode)
        assertTrue(response.body.contains("tool_calls"))
        verify { handle.openaiRequest("/v1/chat/completions", source) }
    }
}
