package ai.meshllm

import org.junit.Assert.*
import org.junit.Test

class InferenceErrorTest {
    @Test
    fun requestAndStreamErrorsExposeBackendFields() {
        val body = """{"error":{"message":"too much context","type":"invalid_request_error","code":"context_length_exceeded","param":"messages","future":true}}"""
        val failure = assertThrows(OpenAIRequestException::class.java) {
            OpenAIResponse(400u, "application/json", body).errorForStatus()
        }
        assertEquals(400.toUShort(), failure.statusCode)
        assertEquals(body, failure.body)
        assertEquals(InferenceErrorDetails("too much context", "invalid_request_error",
            "context_length_exceeded", "messages"), failure.details)
        val stream = OpenAIStreamException("request", 400u, body, "stream failed")
        assertEquals(failure.details, stream.details)
    }

    @Test
    fun unstructuredErrorsKeepRawDetailsWithoutInventedClassification() {
        for (body in listOf(null, "bad gateway", """{"error":"failed"}""",
            """{"error":{"message":"failed","type":"server_error","code":2}}""")) {
            assertNull(OpenAIStreamException("request", null, body, "transport failed").details)
        }
        val error = OpenAIRequestException(500u,
            """{"error":{"message":"failed","type":"server_error"}}""")
        assertEquals(InferenceErrorDetails("failed", "server_error"), error.details)
    }
}
