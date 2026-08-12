import Foundation
import Testing

@testable import MeshAppleRuntime

@Test func incrementalSnapshotsBecomeDeltas() {
  #expect(incrementalDelta(previous: "mesh", snapshot: "mesh-llm") == "-llm")
}

@Test func nonPrefixSnapshotIsPreserved() {
  #expect(incrementalDelta(previous: "old", snapshot: "replacement") == "replacement")
}

@Test func runtimeStatusIsStableJSON() throws {
  let model = AppleModelStatus(
    modelID: AppleRuntimeIdentifiers.systemModelID,
    providerKind: "system",
    availability: "available",
    unavailableReason: nil,
    contextSize: 4_096,
    supportedLanguages: ["en"],
    variant: "test",
    capabilities: ["tool_calling"]
  )
  let status = AppleRuntimeStatus(
    runtimeID: AppleRuntimeIdentifiers.runtimeID,
    protocolVersion: AppleRuntimeIdentifiers.protocolVersion,
    operatingSystem: "macOS",
    models: [model]
  )
  let data = try JSONEncoder().encode(status)
  let decoded = try JSONDecoder().decode(AppleRuntimeStatus.self, from: data)
  #expect(decoded == status)
}

@Test func generationDefaultsToSystemModel() {
  let request = AppleGenerationRequest(prompt: "hello")
  #expect(request.modelID == AppleRuntimeIdentifiers.systemModelID)
}
