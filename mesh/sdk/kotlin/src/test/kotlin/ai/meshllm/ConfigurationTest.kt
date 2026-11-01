package ai.meshllm

import io.mockk.mockk
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import org.junit.Assert.assertEquals
import org.junit.Test
import uniffi.mesh_ffi.MeshNodeHandleInterface

class ConfigurationTest {
    @Test
    fun completeConfigurationIsPassedWithoutDroppingFields() {
        val config = Json.parseToJsonElement("""{"mode":"client","network":{"mesh_name":"team","enumerate_host":false},"admission":{"trust_policy":"allowlist","trusted_owners":["owner"]},"startup_timeout":{"secs":10,"nanos":0},"future_field":true}""").jsonObject
        Node.fromConfig(config) { source ->
            assertEquals(config, Json.parseToJsonElement(source).jsonObject)
            mockk<MeshNodeHandleInterface>()
        }
    }

    @Test(expected = IllegalArgumentException::class)
    fun nativeValidationFailurePropagates() {
        Node.fromConfig(Json.parseToJsonElement("""{"future_field":true}""").jsonObject) { error ->
            throw IllegalArgumentException(error)
        }
    }
}
