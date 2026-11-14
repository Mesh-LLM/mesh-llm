package ai.meshllm

import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.fail
import org.junit.Test
import uniffi.mesh_ffi.FfiException
import uniffi.mesh_ffi.MeshNodeHandleInterface
import uniffi.mesh_ffi.RuntimeCapacityNative

class RuntimeCapacityTest {
    @Test
    fun capacityPreservesUnsignedValuesChangesUnavailabilityAndErrors() = runTest {
        val handle = mockk<MeshNodeHandleInterface>()
        val id = "org/model@revision#profile"
        val failure = FfiException.HostUnavailable("node is not running")
        every { handle.runtimeCapacity(id) } returnsMany listOf(
            RuntimeCapacityNative(2uL, ULong.MAX_VALUE / 2uL, ULong.MAX_VALUE - 1uL, ULong.MAX_VALUE),
            RuntimeCapacityNative(1uL, 4096uL, 4096uL, 123uL),
            null,
        ) andThenThrows failure
        val node = Node(handle)
        assertEquals(LaneCapacity(2uL, ULong.MAX_VALUE / 2uL, ULong.MAX_VALUE - 1uL, ULong.MAX_VALUE), node.runtimeCapacity(id))
        assertEquals(LaneCapacity(1uL, 4096uL, 4096uL, 123uL), node.runtimeCapacity(id))
        assertNull(node.runtimeCapacity(id))
        try {
            node.runtimeCapacity(id)
            fail("stopped nodes must fail")
        } catch (error: FfiException.HostUnavailable) {
            assertEquals(failure.javaClass, error.javaClass)
            assertEquals(failure.message, error.message)
        }
    }
}
