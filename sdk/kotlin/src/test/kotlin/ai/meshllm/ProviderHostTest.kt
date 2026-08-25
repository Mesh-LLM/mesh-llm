package ai.meshllm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class ProviderHostTest {
    @Test
    fun explicitProviderOptionsOwnCarrierPathsAndPolicy() {
        val options = ProviderRuntimeOptions(
            bundleRoots = listOf(File("/app/provider-runtimes/apple")),
            releaseManifest = File("/app/provider-runtimes.json"),
            cacheDir = File("/cache/providers"),
            allowDownload = true,
            startupTimeoutMs = 45_000UL,
        )

        assertEquals(listOf(File("/app/provider-runtimes/apple")), options.bundleRoots)
        assertEquals(File("/app/provider-runtimes.json"), options.releaseManifest)
        assertEquals(File("/cache/providers"), options.cacheDir)
        assertTrue(options.allowDownload)
        assertEquals(45_000UL, options.startupTimeoutMs)
        assertFalse(options.cleanupBundleRoots.isNotEmpty())
    }
}
