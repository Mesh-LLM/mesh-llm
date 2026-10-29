import Foundation
import XCTest
@testable import MeshLLM

final class ModelsTests: XCTestCase {
    func testDiscoveryUsesNativeQueryAndRetainsProviderMetadata() async throws {
        let models = Models(listJSON: { request in
            let query = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(request.utf8)) as? [String: Any])
            XCTAssertEqual(query["workload_class"] as? String, "document_ocr")
            XCTAssertEqual(query["available"] as? Bool, false)
            XCTAssertEqual(query["cursor"] as? String, "opaque")
            return #"{"data":[{"id":"ocr","display_name":"OCR","kind":"model","workload_class":"document_ocr","availability":"unavailable","operations":["document_ocr"],"providers":[{"provider_id":"peer","local":false,"available":false,"operations":["document_ocr"],"runtime":{"model_name":"ocr","ready":false},"metadata":{"architecture":"future"}}]}],"next_cursor":"next"}"#
        }, modelJSON: { _ in XCTFail("list must not invoke lookup"); return "{}" })
        let page = try await models.list(MeshModelQuery(workloadClass: "document_ocr", available: false, cursor: "opaque"))
        XCTAssertEqual(page.nextCursor, "next")
        XCTAssertEqual(page.data.first?.providers.first?.providerId, "peer")
        XCTAssertNil(page.data.first?.providers.first?.runtime?.contextLength)
        XCTAssertEqual(page.data.first?.providers.first?.metadata, .object(["architecture": .string("future")]))
    }

    func testExactLookupPreservesFutureWorkloadsAndMissingMetadata() async throws {
        let models = Models(listJSON: { _ in XCTFail("lookup must not invoke list"); return "{}" }, modelJSON: { id in
            XCTAssertEqual(id, "org/model@revision#profile")
            return #"{"id":"org/model@revision#profile","display_name":"Future","kind":"future_kind","workload_class":"future_workload","availability":"unknown","operations":[],"providers":[]}"#
        })
        let model = try await models.get("org/model@revision#profile")
        XCTAssertEqual(model.workloadClass, "future_workload")
        XCTAssertEqual(model.kind, "future_kind")
        XCTAssertNil(model.metadata)
    }

    func testNativeFailurePropagates() async throws {
        enum Failure: Error { case absent }
        let models = Models(listJSON: { _ in throw Failure.absent }, modelJSON: { _ in throw Failure.absent })
        do { _ = try await models.get("missing"); XCTFail("lookup should fail") }
        catch Failure.absent { }
        do { _ = try await models.list(); XCTFail("list should fail") }
        catch Failure.absent { }
    }
}
