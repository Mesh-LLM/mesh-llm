import Foundation
import XCTest
@testable import MeshLLM

private final class DocumentHandle: MeshNodeHandle, @unchecked Sendable {
    init() { super.init(noHandle: MeshNodeHandle.NoHandle()) }
    required init(unsafeFromHandle handle: UInt64) { super.init(unsafeFromHandle: handle) }

    override func openaiRequest(path: String, bodyJson: String) throws -> OpenAiResponseNative {
        OpenAiResponseNative(statusCode: 422, contentType: "text/plain", body: "provider rejected request")
    }

    override func documentOcr(file: Data, optionsJson: String) throws -> OpenAiResponseNative {
        XCTAssertEqual(file, Data([0, 255, 13, 10]))
        let options = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(optionsJson.utf8)) as? [String: Any])
        XCTAssertEqual(options["model"] as? String, "ocr")
        XCTAssertEqual(options["filename"] as? String, "scan.pdf")
        XCTAssertEqual(options["content_type"] as? String, "application/pdf")
        XCTAssertEqual(options["response_format"] as? String, "markdown")
        XCTAssertEqual(options["max_pages"] as? Int, 2)
        return OpenAiResponseNative(statusCode: 422, contentType: "text/plain", body: "bad PDF")
    }
}

final class DocumentsTests: XCTestCase {
    func testGenericRequestChecksStatusByDefaultAndOffersExplicitRawMode() async throws {
        let node = Node(handle: DocumentHandle())
        do {
            _ = try await node.inference.request(path: "/v1/chat/completions", body: [:])
            XCTFail("expected checked error")
        } catch let failure as OpenAIRequestFailure {
            XCTAssertEqual(failure.statusCode, 422)
            XCTAssertEqual(failure.body, "provider rejected request")
        }
        let raw = try await node.inference.request(path: "/v1/chat/completions", body: [:], raiseForStatus: false)
        XCTAssertEqual(raw.statusCode, 422)
        XCTAssertEqual(raw.body, "provider rejected request")
    }

    func testOcrPreservesBinaryUploadOptionsAndRawProviderErrors() async throws {
        let node = Node(handle: DocumentHandle())
        let response = try await node.inference.documentOcr(Data([0, 255, 13, 10]),
            options: .init(model: "ocr", filename: "scan.pdf", contentType: "application/pdf",
                responseFormat: "markdown", maxPages: 2), raiseForStatus: false)
        XCTAssertEqual(response.statusCode, 422)
        XCTAssertEqual(response.contentType, "text/plain")
        XCTAssertEqual(response.body, "bad PDF")
    }

    func testOcrChecksStatusByDefaultWithoutDiscardingProviderBody() async throws {
        let node = Node(handle: DocumentHandle())
        do {
            _ = try await node.inference.documentOcr(Data([0, 255, 13, 10]), options:
                .init(model: "ocr", filename: "scan.pdf", contentType: "application/pdf",
                    responseFormat: "markdown", maxPages: 2))
            XCTFail("expected provider error")
        } catch let failure as OpenAIRequestFailure {
            XCTAssertEqual(failure.statusCode, 422)
            XCTAssertEqual(failure.body, "bad PDF")
        }
    }
}
