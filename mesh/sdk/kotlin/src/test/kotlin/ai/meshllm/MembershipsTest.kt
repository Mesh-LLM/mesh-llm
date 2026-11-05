package ai.meshllm

import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Test

class MembershipsTest {
    private val status = """{"name":"team","api_base_url":"http://localhost:9338/v1","invite_token":"invite","endpoint_id":"node-a","peers":2,"join_error":null}"""

    @Test
    fun controlsPreserveWireConfigStatusAndExactNames() = runTest {
        val memberships = Memberships(addJson = { source ->
            val config = Json.parseToJsonElement(source).jsonObject
            assertEquals(JsonPrimitive("team"), config["name"])
            assertEquals(JsonPrimitive(9338), config["api_port"])
            assertEquals(JsonPrimitive(false), config["enumerate_host"])
            status
        }, removeNative = { name -> assertEquals("team", name); false }, listJson = { "[$status]" })
        val added = memberships.add(MeshMembershipConfig(name = "team", apiPort = 9338u))
        assertEquals("http://localhost:9338/v1", added.apiBaseUrl)
        assertEquals("node-a", added.endpointId)
        assertNull(added.joinError)
        assertEquals(2L, memberships.list().first().peers)
        assertFalse(memberships.remove("team"))
    }

    @Test(expected = IllegalStateException::class)
    fun nativeRejectionPropagates() = runTest {
        val memberships = Memberships(addJson = { error("rejected") }, removeNative = { error("rejected") }, listJson = { error("rejected") })
        memberships.add(MeshMembershipConfig(name = "?", apiPort = 9338u))
    }
}
