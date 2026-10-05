import XCTest
@testable import MeshLLM

final class NativeRuntimeTests: XCTestCase {
    func testResolveOptionsDefaultToOfflineRecommended() {
        let options = NativeRuntimeResolveOptions()

        XCTAssertEqual(options.selection, "recommended")
        XCTAssertFalse(options.allowDownload)
        XCTAssertNil(options.artifactDirectory)
        XCTAssertTrue(options.searchDirectories.isEmpty)
    }

    func testSdkRuntimeVersionsAreExposed() {
        XCTAssertFalse(NativeRuntime.meshVersion.isEmpty)
        XCTAssertFalse(NativeRuntime.skippyAbiVersion.isEmpty)
    }
}
