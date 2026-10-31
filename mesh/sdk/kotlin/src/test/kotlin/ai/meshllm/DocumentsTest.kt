package ai.meshllm

import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test
import uniffi.mesh_ffi.MeshNodeHandleInterface
import uniffi.mesh_ffi.OpenAiResponseNative

class DocumentsTest {
    @Test
    fun genericRequestChecksStatusByDefaultAndOffersRawMode() = runTest {
        val handle = mockk<MeshNodeHandleInterface>()
        every { handle.openaiRequest(any(), any()) } returns
            OpenAiResponseNative(422u.toUShort(), "text/plain", "provider rejected request")
        val inference = Node(handle).inference
        try {
            inference.requestJson("/v1/chat/completions", "{}")
            org.junit.Assert.fail("expected checked failure")
        } catch (failure: OpenAIRequestException) {
            assertEquals(422u.toUShort(), failure.statusCode)
            assertEquals("provider rejected request", failure.body)
        }
        val raw = inference.requestJson("/v1/chat/completions", "{}", raiseForStatus = false)
        assertEquals(422u.toUShort(), raw.statusCode)
        assertEquals("provider rejected request", raw.body)
    }

    @Test
    fun binaryUploadOptionsAndProviderErrorsRemainUnchanged() = runTest {
        val handle = mockk<MeshNodeHandleInterface>()
        val file = byteArrayOf(0, -1, 13, 10)
        every { handle.documentOcr(any(), any()) } answers {
            assertArrayEquals(file, firstArg<ByteArray>())
            val options = Json.parseToJsonElement(secondArg<String>()).jsonObject
            assertEquals(JsonPrimitive("ocr"), options["model"])
            assertEquals(JsonPrimitive("application/pdf"), options["content_type"])
            assertEquals(JsonPrimitive("markdown"), options["response_format"])
            assertEquals(JsonPrimitive(2), options["max_pages"])
            OpenAiResponseNative(422u.toUShort(), "text/plain", "bad PDF")
        }
        val response = Node(handle).inference.documentOcr(file,
            DocumentOcrOptions(model = "ocr", filename = "scan.pdf", contentType = "application/pdf",
                responseFormat = "markdown", maxPages = 2), raiseForStatus = false)
        assertEquals(422u.toUShort(), response.statusCode)
        assertEquals("text/plain", response.contentType)
        assertEquals("bad PDF", response.body)
    }

    @Test(expected = OpenAIRequestException::class)
    fun ocrChecksProviderStatusByDefault() = runTest {
        val handle = mockk<MeshNodeHandleInterface>()
        every { handle.documentOcr(any(), any()) } returns OpenAiResponseNative(422u.toUShort(), "text/plain", "bad PDF")
        Node(handle).inference.documentOcr(byteArrayOf(1), DocumentOcrOptions(model = "ocr"))
    }

    @Test(expected = IllegalStateException::class)
    fun nativeOcrFailuresPropagate() = runTest {
        val handle = mockk<MeshNodeHandleInterface>()
        every { handle.documentOcr(any(), any()) } throws IllegalStateException("node stopped")
        Node(handle).inference.documentOcr(byteArrayOf(), DocumentOcrOptions(model = "ocr"))
    }
}
