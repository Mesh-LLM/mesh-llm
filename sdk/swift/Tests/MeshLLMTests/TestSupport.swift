import XCTest
@testable import MeshLLM

func makeOwnerKeypairBytesHex() -> String {
    #if canImport(mesh_ffiFFI)
    return generateOwnerKeypairHex()
    #else
    return "test-owner-keypair"
    #endif
}
