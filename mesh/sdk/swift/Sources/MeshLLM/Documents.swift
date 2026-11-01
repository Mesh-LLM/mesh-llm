import Foundation

/// Document OCR selection and multipart file metadata.
public struct DocumentOcrOptions: Encodable, Sendable {
    public var model: String
    public var filename: String
    public var contentType: String
    public var responseFormat: String?
    public var maxPages: UInt32?

    public init(model: String, filename: String = "document",
                contentType: String = "application/octet-stream",
                responseFormat: String? = nil, maxPages: UInt32? = nil) {
        self.model = model
        self.filename = filename
        self.contentType = contentType
        self.responseFormat = responseFormat
        self.maxPages = maxPages
    }

    func json() throws -> String {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        return String(decoding: try encoder.encode(self), as: UTF8.self)
    }
}
