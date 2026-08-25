import Foundation
#if os(macOS)
import MeshLLMAppleProviderResources
#endif

public struct ProviderRuntimeOptions: Sendable {
    public let bundleRoots: [URL]
    public let releaseManifest: URL?
    public let cacheDirectory: URL?
    public let allowDownload: Bool
    public let startupTimeout: Duration

    public init(
        bundleRoots: [URL],
        releaseManifest: URL? = nil,
        cacheDirectory: URL? = nil,
        allowDownload: Bool = false,
        startupTimeout: Duration = .seconds(30)
    ) {
        self.bundleRoots = bundleRoots
        self.releaseManifest = releaseManifest
        self.cacheDirectory = cacheDirectory
        self.allowDownload = allowDownload
        self.startupTimeout = startupTimeout
    }

    public static func packagedAppleSystem(
        cacheDirectory: URL? = nil,
        allowDownload: Bool = false
    ) throws -> ProviderRuntimeOptions {
        #if os(macOS)
        guard let root = MeshLLMAppleProviderResources.appleRuntimeRoot else {
            throw ProviderRuntimeAssetError.packagedAppleRuntimeMissing
        }
        return ProviderRuntimeOptions(
            bundleRoots: [root],
            cacheDirectory: cacheDirectory,
            allowDownload: allowDownload
        )
        #else
        throw ProviderRuntimeAssetError.packagedAppleRuntimeMissing
        #endif
    }
}

public enum ProviderRuntimeAssetError: Error, CustomStringConvertible {
    case packagedAppleRuntimeMissing

    public var description: String {
        switch self {
        case .packagedAppleRuntimeMissing:
            return "Packaged Apple provider runtime is missing from the MeshLLM SwiftPM resources."
        }
    }
}

#if canImport(MeshLLMFFI)
public final class ProviderHost: @unchecked Sendable {
    private let handle: ProviderHostHandle

    public var apiBaseURL: URL {
        URL(string: handle.apiBaseUrl())!
    }

    private init(handle: ProviderHostHandle) {
        self.handle = handle
    }

    public static func start(_ options: ProviderRuntimeOptions) async throws -> ProviderHost {
        let timeout = options.startupTimeout.components
        let timeoutMilliseconds = UInt64(max(1, timeout.seconds * 1_000
            + Int64(timeout.attoseconds / 1_000_000_000_000_000)))
        let handle = try await runProviderBlocking {
            try startProviderHost(
                options: ProviderRuntimeOptionsNative(
                    bundleRoots: options.bundleRoots.map(\.path),
                    releaseManifest: options.releaseManifest?.path,
                    cacheDir: options.cacheDirectory?.path,
                    allowDownload: options.allowDownload,
                    startupTimeoutMs: timeoutMilliseconds
                )
            )
        }
        return ProviderHost(handle: handle)
    }

    public func statusJSON() async throws -> String {
        let handle = self.handle
        return try await runProviderBlocking {
            try handle.statusJson()
        }
    }

    public func stop() async throws {
        let handle = self.handle
        try await runProviderBlocking {
            try handle.stop()
        }
    }
}

private func runProviderBlocking<T>(_ work: @escaping () throws -> T) async throws -> T {
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
#error("MeshLLM Swift SDK requires MeshLLMFFI.xcframework. Build it before using provider hosts.")
#endif
