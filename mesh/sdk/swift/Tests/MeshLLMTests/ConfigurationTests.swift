import Foundation
import XCTest
@testable import MeshLLM

final class ConfigurationTests: XCTestCase {
    func testFullConfigurationReachesNativeConstructorWithoutFiltering() throws {
        let config: [String: ModelJSONValue] = [
            "mode": .string("client"),
            "network": .object(["mesh_name": .string("team"), "enumerate_host": .bool(false)]),
            "admission": .object(["trust_policy": .string("allowlist"), "trusted_owners": .array([.string("owner")])]),
            "startup_timeout": .object(["secs": .number(10), "nanos": .number(0)]),
            "future_field": .bool(true),
        ]
        _ = try Node.fromConfig(config, create: { source in
            let decoded = try JSONDecoder().decode([String: ModelJSONValue].self, from: Data(source.utf8))
            XCTAssertEqual(decoded, config)
            return TestMeshNodeHandle()
        })
    }

    func testNativeConfigurationErrorIsNotSwallowed() throws {
        enum Failure: Error { case invalid }
        XCTAssertThrowsError(try Node.fromConfig(["future_field": .bool(true)], create: { _ in throw Failure.invalid }))
    }
}
