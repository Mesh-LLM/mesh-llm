import Foundation
import XCTest
@testable import MeshLLM

final class LiveTranscriptionTests: XCTestCase {
    func testOptionsPreserveSampleRateAndExplicitPushToTalk() throws {
        let options = LiveTranscriptionOptions(sampleRate: 48000, turnDetection: nil)
        let value = try JSONSerialization.jsonObject(with: Data(options.json().utf8)) as! [String: Any]
        XCTAssertEqual(value["sample_rate"] as? Int, 48000)
        XCTAssertTrue(value["turn_detection"] is NSNull)
        let defaults = try JSONSerialization.jsonObject(with: Data(LiveTranscriptionOptions().json().utf8)) as! [String: Any]
        XCTAssertNotNil(defaults["turn_detection"] as? [String: Any])
    }

    func testSessionPreservesAudioEventsAndRemainingTranscript() async throws {
        let live = LiveTranscription(
            audio: { samples in XCTAssertEqual(samples, [1, -2]) },
            floatAudio: { samples in XCTAssertEqual(samples, [0.5]) },
            commit: {},
            eventJSON: { #"{"type":"delta","item_id":"turn","text":"hello"}"# },
            stopJSON: { #"[{"type":"completed","item_id":"turn","transcript":"hello"}]"# },
            close: {}
        )
        try await live.sendAudio([1, -2])
        try await live.sendAudioF32([0.5])
        try await live.commit()
        let event = try await live.nextEvent()
        XCTAssertEqual(event?.itemId, "turn")
        XCTAssertEqual(event?.text, "hello")
        let remaining = try await live.stop()
        XCTAssertEqual(remaining.first?.transcript, "hello")
        await live.close()
    }

    func testNativeErrorsAndEndOfStreamArePreserved() async throws {
        enum Failure: Error { case closed }
        let live = LiveTranscription(audio: { _ in throw Failure.closed }, floatAudio: { _ in }, commit: {}, eventJSON: { nil }, stopJSON: { "[]" }, close: {})
        let event = try await live.nextEvent()
        XCTAssertNil(event)
        do { try await live.sendAudio([]); XCTFail("expected native failure") }
        catch Failure.closed {}
    }
}
