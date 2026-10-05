import Foundation

#if canImport(MeshLLMFFI)
import MeshLLMFFI

public typealias NativeRuntimeInstallOptions = NativeRuntimeInstallOptionsNative
public typealias NativeRuntimeInstallOutcome = NativeRuntimeInstallOutcomeNative
public typealias NativeRuntimeDownloadProgress = NativeRuntimeDownloadProgressNative
public typealias InstalledNativeRuntime = InstalledNativeRuntimeNative
public typealias NativeRuntimePruneResult = NativeRuntimePruneResultNative
public typealias NativeRuntimeVerificationPolicy = NativeRuntimeVerificationPolicyNative
public typealias NativeRuntimePruneMode = NativeRuntimePruneModeNative

public struct NativeRuntimeResolveOptions: Sendable {
    public let artifactDirectory: URL?
    public let searchDirectories: [URL]
    public let cacheDirectory: URL?
    public let allowDownload: Bool
    public let selection: String

    public init(
        artifactDirectory: URL? = nil,
        searchDirectories: [URL] = [],
        cacheDirectory: URL? = nil,
        allowDownload: Bool = false,
        selection: String = "recommended"
    ) {
        self.artifactDirectory = artifactDirectory
        self.searchDirectories = searchDirectories
        self.cacheDirectory = cacheDirectory
        self.allowDownload = allowDownload
        self.selection = selection
    }
}

public enum NativeRuntime {
    public static var meshVersion: String {
        currentMeshVersion()
    }

    public static var skippyAbiVersion: String {
        currentSkippyAbiVersion()
    }

    public static func install(
        _ options: NativeRuntimeInstallOptions,
        onProgress: (@Sendable (NativeRuntimeDownloadProgress) -> Void)? = nil
    ) async throws -> NativeRuntimeInstallOutcome {
        let progress = onProgress.map(ProgressListener.init)
        return try await runBlocking {
            try installNativeRuntime(options: options, progress: progress)
        }
    }

    public static func resolve(
        _ options: NativeRuntimeResolveOptions = NativeRuntimeResolveOptions(),
        onProgress: (@Sendable (NativeRuntimeDownloadProgress) -> Void)? = nil
    ) async throws -> InstalledNativeRuntime {
        var bundleDirs: [String] = options.searchDirectories.map(\.path)
        if let artifactDirectory = options.artifactDirectory {
            bundleDirs.insert(artifactDirectory.path, at: 0)
        }
        let outcome = try await install(
            NativeRuntimeInstallOptions(
                meshVersion: nil,
                skippyAbiVersion: nil,
                selection: options.selection,
                manifestPath: nil,
                manifestUrl: nil,
                bundleDirs: bundleDirs,
                cacheDir: options.cacheDirectory?.path,
                verificationPolicy: .requireChecksum,
                allowDownload: options.allowDownload
            ),
            onProgress: onProgress
        )
        return outcome.runtime
    }

    public static func validate(
        artifactDirectory: URL,
        cacheDirectory: URL? = nil
    ) async throws -> InstalledNativeRuntime {
        try await resolve(
            NativeRuntimeResolveOptions(
                artifactDirectory: artifactDirectory,
                cacheDirectory: cacheDirectory,
                allowDownload: false
            )
        )
    }

    public static func installed(cacheDirectory: URL? = nil) async throws -> [InstalledNativeRuntime] {
        try await runBlocking {
            try installedNativeRuntimes(cacheDir: cacheDirectory?.path)
        }
    }

    @discardableResult
    public static func remove(
        meshVersion: String,
        nativeRuntimeId: String,
        cacheDirectory: URL? = nil
    ) async throws -> Bool {
        try await runBlocking {
            try removeNativeRuntime(
                cacheDir: cacheDirectory?.path,
                meshVersion: meshVersion,
                nativeRuntimeId: nativeRuntimeId
            )
        }
    }

    public static func prune(
        cacheDirectory: URL? = nil,
        activeMeshVersion: String? = nil,
        mode: NativeRuntimePruneMode = .keepActiveAndPrevious
    ) async throws -> NativeRuntimePruneResult {
        try await runBlocking {
            try pruneNativeRuntimes(
                cacheDir: cacheDirectory?.path,
                activeMeshVersion: activeMeshVersion,
                mode: mode
            )
        }
    }
}

private final class ProgressListener: NativeRuntimeProgressListener, @unchecked Sendable {
    private let onProgress: @Sendable (NativeRuntimeDownloadProgress) -> Void

    init(_ onProgress: @escaping @Sendable (NativeRuntimeDownloadProgress) -> Void) {
        self.onProgress = onProgress
    }

    func onProgress(event: NativeRuntimeDownloadProgressNative) {
        onProgress(event)
    }
}

private func runBlocking<T>(_ work: @escaping () throws -> T) async throws -> T {
    try await withCheckedThrowingContinuation { continuation in
        DispatchQueue.global().async(flags: .inheritQoS) {
            do {
                continuation.resume(returning: try work())
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
}
#else
#error("MeshLLM Swift SDK requires MeshLLMFFI.xcframework. Build it with sdk/swift/scripts/build-xcframework.sh before building, testing, or integrating the SDK.")
#endif
