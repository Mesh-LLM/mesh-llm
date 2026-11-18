import Foundation

/// Structured backend error fields, preserved independently of display text.
public struct InferenceErrorDetails: Decodable, Equatable, Sendable {
    public let message: String
    public let type: String
    public let code: String?
    public let param: String?

    internal static func decode(_ body: String?) -> InferenceErrorDetails? {
        struct Envelope: Decodable { let error: InferenceErrorDetails }
        guard let body else { return nil }
        return try? JSONDecoder().decode(Envelope.self, from: Data(body.utf8)).error
    }
}
