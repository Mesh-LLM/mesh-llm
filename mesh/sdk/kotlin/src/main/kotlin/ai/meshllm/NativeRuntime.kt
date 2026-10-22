package ai.meshllm

import java.io.File
import uniffi.mesh_ffi.InstalledNativeRuntimeNative
import uniffi.mesh_ffi.NativeRuntimeDownloadProgressNative
import uniffi.mesh_ffi.NativeRuntimeInstallOptionsNative
import uniffi.mesh_ffi.NativeRuntimeInstallOutcomeNative
import uniffi.mesh_ffi.NativeRuntimeProgressListener
import uniffi.mesh_ffi.NativeRuntimePruneModeNative
import uniffi.mesh_ffi.NativeRuntimePruneResultNative
import uniffi.mesh_ffi.NativeRuntimeVerificationPolicyNative
import uniffi.mesh_ffi.currentMeshVersion
import uniffi.mesh_ffi.currentSkippyAbiVersion
import uniffi.mesh_ffi.installNativeRuntime
import uniffi.mesh_ffi.installedNativeRuntimes
import uniffi.mesh_ffi.pruneNativeRuntimes
import uniffi.mesh_ffi.removeNativeRuntime

typealias NativeRuntimeInstallOptions = NativeRuntimeInstallOptionsNative
typealias NativeRuntimeInstallOutcome = NativeRuntimeInstallOutcomeNative
typealias NativeRuntimeDownloadProgress = NativeRuntimeDownloadProgressNative
typealias InstalledNativeRuntime = InstalledNativeRuntimeNative
typealias NativeRuntimePruneMode = NativeRuntimePruneModeNative
typealias NativeRuntimePruneResult = NativeRuntimePruneResultNative
typealias NativeRuntimeVerificationPolicy = NativeRuntimeVerificationPolicyNative

data class NativeRuntimeResolveOptions(
    val artifactDir: File? = null,
    val searchDirs: List<File> = emptyList(),
    val cacheDir: File? = null,
    val allowDownload: Boolean = false,
    val selection: String = "recommended",
)

object NativeRuntime {
    val meshVersion: String
        get() = currentMeshVersion()

    val skippyAbiVersion: String
        get() = currentSkippyAbiVersion()

    fun install(
        options: NativeRuntimeInstallOptions = defaultInstallOptions(),
        onProgress: ((NativeRuntimeDownloadProgress) -> Unit)? = null,
    ): NativeRuntimeInstallOutcome =
        installNativeRuntime(options, onProgress?.let(::ProgressListener))

    fun resolve(
        options: NativeRuntimeResolveOptions = NativeRuntimeResolveOptions(),
        onProgress: ((NativeRuntimeDownloadProgress) -> Unit)? = null,
    ): InstalledNativeRuntime {
        val bundleDirs = buildList {
            options.artifactDir?.let { add(it.path) }
            addAll(options.searchDirs.map(File::getPath))
        }
        return install(
            defaultInstallOptions(
                selection = options.selection,
                bundleDirs = bundleDirs,
                cacheDir = options.cacheDir?.path,
                allowDownload = options.allowDownload,
            ),
            onProgress,
        ).runtime
    }

    fun validate(artifactDir: File, cacheDir: File? = null): InstalledNativeRuntime =
        resolve(
            NativeRuntimeResolveOptions(
                artifactDir = artifactDir,
                cacheDir = cacheDir,
                allowDownload = false,
            )
        )

    fun installed(cacheDir: File? = null): List<InstalledNativeRuntime> =
        installedNativeRuntimes(cacheDir?.path)

    fun remove(
        meshVersion: String,
        nativeRuntimeId: String,
        cacheDir: File? = null,
    ): Boolean =
        removeNativeRuntime(cacheDir?.path, meshVersion, nativeRuntimeId)

    fun prune(
        cacheDir: File? = null,
        activeMeshVersion: String? = null,
        mode: NativeRuntimePruneMode = NativeRuntimePruneMode.KEEP_ACTIVE_AND_PREVIOUS,
    ): NativeRuntimePruneResult =
        pruneNativeRuntimes(cacheDir?.path, activeMeshVersion, mode)

    fun defaultInstallOptions(
        meshVersion: String? = null,
        skippyAbiVersion: String? = null,
        selection: String = "recommended",
        manifestPath: String? = null,
        manifestUrl: String? = null,
        bundleDirs: List<String> = emptyList(),
        cacheDir: String? = null,
        verificationPolicy: NativeRuntimeVerificationPolicy = NativeRuntimeVerificationPolicy.REQUIRE_CHECKSUM,
        allowDownload: Boolean = true,
    ): NativeRuntimeInstallOptions =
        NativeRuntimeInstallOptions(
            meshVersion = meshVersion,
            skippyAbiVersion = skippyAbiVersion,
            selection = selection,
            manifestPath = manifestPath,
            manifestUrl = manifestUrl,
            bundleDirs = bundleDirs,
            cacheDir = cacheDir,
            verificationPolicy = verificationPolicy,
            allowDownload = allowDownload,
        )
}

private class ProgressListener(
    private val onProgress: (NativeRuntimeDownloadProgress) -> Unit,
) : NativeRuntimeProgressListener {
    override fun onProgress(event: NativeRuntimeDownloadProgressNative) {
        onProgress.invoke(event)
    }
}
