package ai.meshllm

import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.File
import uniffi.mesh_ffi.NativeRuntimeVerificationPolicyNative

class NativeRuntimeTest {
    @Test
    fun defaultInstallOptionsUseRecommendedSelection() {
        val options = NativeRuntime.defaultInstallOptions(
            cacheDir = "/cache",
            allowDownload = false,
        )

        assertEquals("recommended", options.selection)
        assertEquals("/cache", options.cacheDir)
        assertEquals(false, options.allowDownload)
        assertEquals(NativeRuntimeVerificationPolicyNative.REQUIRE_CHECKSUM, options.verificationPolicy)
    }

    @Test
    fun resolveOptionsDefaultToOfflineRecommended() {
        val options = NativeRuntimeResolveOptions()

        assertEquals("recommended", options.selection)
        assertEquals(false, options.allowDownload)
        assertEquals(null, options.artifactDir)
        assertEquals(emptyList<File>(), options.searchDirs)
    }
}
