package ai.meshllm

import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.fail
import org.junit.Test

class ModelsTest {
    @Test
    fun filtersAndProviderObservationsSurviveNativeDiscovery() = runTest {
        val models = Models(listJson = { request ->
            val query = Json.parseToJsonElement(request).jsonObject
            assertEquals(JsonPrimitive("document_ocr"), query["workload_class"])
            assertEquals(JsonPrimitive(false), query["available"])
            assertEquals(JsonPrimitive("opaque"), query["cursor"])
            """{"data":[{"id":"ocr","display_name":"OCR","kind":"model","workload_class":"document_ocr","availability":"unavailable","operations":["document_ocr"],"providers":[{"provider_id":"peer","local":false,"available":false,"operations":["document_ocr"],"metadata":{"architecture":"future"},"runtime":{"model_name":"ocr","ready":false}}]}],"next_cursor":"next"}"""
        }, modelJson = { error("list must not invoke lookup") })
        val page = models.list(MeshModelQuery(workloadClass = "document_ocr", available = false, cursor = "opaque"))
        assertEquals("next", page.nextCursor)
        assertEquals("peer", page.data.first().providers.first().providerId)
        assertNull(page.data.first().providers.first().runtime?.contextLength)
        assertEquals(JsonPrimitive("future"), page.data.first().providers.first().metadata?.jsonObject?.get("architecture"))
    }

    @Test
    fun lookupPreservesExactIdsAndNormalizedUnknownClasses() = runTest {
        val models = Models(listJson = { error("lookup must not invoke list") }, modelJson = { id ->
            assertEquals("org/model@revision#profile", id)
            // Native deserialization normalizes unrecognized peer workloads and
            // kinds to "unknown" before this surface sees them, so realistic
            // input carries the normalized values.
            """{"id":"org/model@revision#profile","display_name":"Future","kind":"unknown","workload_class":"unknown","availability":"unknown","operations":[],"providers":[]}"""
        })
        val model = models.get("org/model@revision#profile")
        assertEquals("unknown", model.workloadClass)
        assertEquals("unknown", model.kind)
        assertNull(model.metadata)
    }

    @Test
    fun nativeFailurePropagates() = runTest {
        val failure = IllegalStateException("model not found")
        val models = Models(listJson = { throw failure }, modelJson = { throw failure })
        try { models.get("missing"); fail("lookup should fail") }
        catch (error: IllegalStateException) { assertEquals(failure, error) }
        try { models.list(); fail("list should fail") }
        catch (error: IllegalStateException) { assertEquals(failure, error) }
    }
}
