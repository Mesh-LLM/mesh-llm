import Foundation

/// Structured JSON for open-ended capability, thinking, channel requirements and plugin metadata.
public enum ModelJSONValue: Codable, Sendable, Equatable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([ModelJSONValue])
    case object([String: ModelJSONValue])

    public init(from decoder: Decoder) throws {
        let value = try decoder.singleValueContainer()
        if value.decodeNil() { self = .null }
        else if let bool = try? value.decode(Bool.self) { self = .bool(bool) }
        else if let number = try? value.decode(Double.self) { self = .number(number) }
        else if let string = try? value.decode(String.self) { self = .string(string) }
        else if let array = try? value.decode([ModelJSONValue].self) { self = .array(array) }
        else { self = .object(try value.decode([String: ModelJSONValue].self)) }
    }

    public func encode(to encoder: Encoder) throws {
        var value = encoder.singleValueContainer()
        switch self {
        case .null: try value.encodeNil()
        case .bool(let bool): try value.encode(bool)
        case .number(let number): try value.encode(number)
        case .string(let string): try value.encode(string)
        case .array(let array): try value.encode(array)
        case .object(let object): try value.encode(object)
        }
    }
}

/// All filters apply together. Omitted filters include every advertised workload.
public struct MeshModelQuery: Encodable, Sendable {
    public var workloadClass: String?
    public var kind: String?
    public var available: Bool?
    public var limit: UInt32?
    public var cursor: String?

    public init(workloadClass: String? = nil, kind: String? = nil, available: Bool? = nil,
                limit: UInt32? = nil, cursor: String? = nil) {
        self.workloadClass = workloadClass
        self.kind = kind
        self.available = available
        self.limit = limit
        self.cursor = cursor
    }
}

public struct MeshModelRuntime: Decodable, Sendable {
    public let modelName: String
    public let identityHash: String?
    public let contextLength: UInt32?
    public let ready: Bool
}

/// Runtime limits and controls apply to this provider, not to every model replica.
public struct MeshModelProvider: Decodable, Sendable {
    public let providerId: String
    public let local: Bool
    public let endpointId: String?
    public let plugin: String?
    public let workloadClass: String?
    public let available: Bool
    public let operations: [String]
    public let capabilities: ModelJSONValue?
    public let metadata: ModelJSONValue?
    public let runtime: MeshModelRuntime?
    public let thinking: ModelJSONValue?
    public let requires: ModelJSONValue?
}

/// One advertised identity. String-valued classes preserve future peer values.
public struct MeshModel: Decodable, Sendable {
    public let id: String
    public let displayName: String
    public let kind: String
    public let workloadClass: String?
    public let availability: String
    public let operations: [String]
    public let metadata: ModelJSONValue?
    public let providers: [MeshModelProvider]
    public let virtualModel: ModelJSONValue?
}

public struct MeshModelPage: Decodable, Sendable {
    public let data: [MeshModel]
    public let nextCursor: String?
}

/// Native mesh discovery, including LLM, speech, embedding, decision and OCR models.
public final class Models: @unchecked Sendable {
    private let listJSON: @Sendable (String) async throws -> String
    private let modelJSON: @Sendable (String) async throws -> String

    internal init(
        listJSON: @escaping @Sendable (String) async throws -> String,
        modelJSON: @escaping @Sendable (String) async throws -> String
    ) {
        self.listJSON = listJSON
        self.modelJSON = modelJSON
    }

    public func list(_ query: MeshModelQuery = MeshModelQuery()) async throws -> MeshModelPage {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let request = String(decoding: try encoder.encode(query), as: UTF8.self)
        return try Self.decode(MeshModelPage.self, json: await listJSON(request))
    }

    public func get(_ id: String) async throws -> MeshModel {
        try Self.decode(MeshModel.self, json: await modelJSON(id))
    }

    private static func decode<T: Decodable>(_ type: T.Type, json: String) throws -> T {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(type, from: Data(json.utf8))
    }
}
