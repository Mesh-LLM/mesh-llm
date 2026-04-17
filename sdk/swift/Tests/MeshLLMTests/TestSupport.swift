import XCTest
@testable import MeshLLM

func makeOwnerKeypairBytesHex() -> String {
    #if canImport(MeshLLMFFI)
    return generateOwnerKeypairHex()
    #else
    return "test-owner-keypair"
    #endif
}
