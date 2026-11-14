import XCTest
@testable import MeshLLM

final class InferenceErrorTests: XCTestCase {
    func testRequestAndStreamErrorsExposeBackendFields() throws {
        let body = #"{"error":{"message":"too much context","type":"invalid_request_error","code":"context_length_exceeded","param":"messages","future":true}}"#
        let response = OpenAIResponse(statusCode: 400, contentType: "application/json", body: body)
        XCTAssertThrowsError(try response.errorForStatus()) { error in
            let failure = error as! OpenAIRequestFailure
            XCTAssertEqual(failure.statusCode, 400)
            XCTAssertEqual(failure.body, body)
            XCTAssertEqual(failure.details?.code, "context_length_exceeded")
            XCTAssertEqual(failure.details?.type, "invalid_request_error")
            XCTAssertEqual(failure.details?.message, "too much context")
            XCTAssertEqual(failure.details?.param, "messages")
            let stream = OpenAIStreamFailure(requestId: "request", statusCode: 400,
                message: "stream failed", body: body)
            XCTAssertEqual(stream.details, failure.details)
        }
    }

    func testUnstructuredErrorsRemainReadableWithoutInventedClassification() {
        for body in [nil, "bad gateway", #"{"error":"failed"}"#,
                     #"{"error":{"message":"failed","type":"server_error","code":2}}"#] {
            XCTAssertNil(OpenAIStreamFailure(requestId: "request", statusCode: nil,
                message: "transport failed", body: body).details)
        }
        let error = OpenAIRequestFailure(statusCode: 500,
            body: #"{"error":{"message":"failed","type":"server_error"}}"#)
        XCTAssertEqual(error.details?.message, "failed")
        XCTAssertNil(error.details?.code)
        XCTAssertNil(error.details?.param)
    }
}
