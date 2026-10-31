import Foundation

/// An additional mesh served by this node, with its own owner allowlist.
public struct MeshMembershipConfig: Encodable, Sendable {
    public var name: String
    public var joinTokens: [String]
    public var trustedOwners: [String]
    public var apiPort: UInt16
    public var nodeKeyPath: String?
    public var enumerateHost: Bool

    public init(name: String, joinTokens: [String] = [], trustedOwners: [String] = [],
                apiPort: UInt16, nodeKeyPath: String? = nil, enumerateHost: Bool = false) {
        self.name = name
        self.joinTokens = joinTokens
        self.trustedOwners = trustedOwners
        self.apiPort = apiPort
        self.nodeKeyPath = nodeKeyPath
        self.enumerateHost = enumerateHost
    }
}

public struct MeshMembershipStatus: Decodable, Sendable {
    public let name: String
    public let apiBaseUrl: String
    public let inviteToken: String
    public let endpointId: String
    public let peers: UInt64
    public let joinError: String?
}

final class Memberships: @unchecked Sendable {
    private let addJSON: @Sendable (String) async throws -> String
    private let removeNative: @Sendable (String) async throws -> Bool
    private let listJSON: @Sendable () async throws -> String

    init(addJSON: @escaping @Sendable (String) async throws -> String,
         removeNative: @escaping @Sendable (String) async throws -> Bool,
         listJSON: @escaping @Sendable () async throws -> String) {
        self.addJSON = addJSON
        self.removeNative = removeNative
        self.listJSON = listJSON
    }

    func add(_ config: MeshMembershipConfig) async throws -> MeshMembershipStatus {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let request = String(decoding: try encoder.encode(config), as: UTF8.self)
        return try decode(MeshMembershipStatus.self, from: await addJSON(request))
    }

    func remove(_ name: String) async throws -> Bool { try await removeNative(name) }

    func list() async throws -> [MeshMembershipStatus] {
        try decode([MeshMembershipStatus].self, from: await listJSON())
    }

    private func decode<T: Decodable>(_ type: T.Type, from json: String) throws -> T {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(type, from: Data(json.utf8))
    }
}
