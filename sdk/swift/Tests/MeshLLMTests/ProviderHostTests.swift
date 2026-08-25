import XCTest
@testable import MeshLLM

final class ProviderHostTests: XCTestCase {
    func testExplicitCarrierConfigurationUsesURLs() {
        let root = URL(fileURLWithPath: "/app/Resources/provider-runtimes/apple")
        let options = ProviderRuntimeOptions(
            bundleRoots: [root],
            releaseManifest: URL(fileURLWithPath: "/app/Resources/provider-runtimes.json"),
            cacheDirectory: URL(fileURLWithPath: "/cache/providers"),
            allowDownload: true,
            startupTimeout: .seconds(45)
        )

        XCTAssertEqual(options.bundleRoots, [root])
        XCTAssertTrue(options.allowDownload)
        XCTAssertEqual(options.startupTimeout, .seconds(45))
    }
}
