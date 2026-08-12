import Foundation

public actor AppleRuntime {
  private let systemModel: SystemModelProvider

  public init() {
    systemModel = SystemModelProvider()
  }

  public func status() async -> AppleRuntimeStatus {
    AppleRuntimeStatus(
      runtimeID: AppleRuntimeIdentifiers.runtimeID,
      protocolVersion: AppleRuntimeIdentifiers.protocolVersion,
      operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
      models: [await systemModel.status()]
    )
  }

  public func prewarm(modelID: String, promptPrefix: String? = nil) async throws {
    try requireSystemModel(modelID)
    try await systemModel.prewarm(promptPrefix: promptPrefix)
  }

  public func generate(
    request: AppleGenerationRequest,
    onEvent: @Sendable (AppleRuntimeEvent) -> Void
  ) async throws -> AppleGenerationResult {
    try requireSystemModel(request.modelID)
    return try await systemModel.generate(request: request, onEvent: onEvent)
  }

  public func generateStructured(
    modelID: String,
    prompt: String
  ) async throws -> AppleStructuredResult {
    try requireSystemModel(modelID)
    return try await systemModel.generateStructured(prompt: prompt)
  }

  public func exerciseTool(modelID: String, key: String) async throws -> AppleToolResult {
    try requireSystemModel(modelID)
    return try await systemModel.exerciseTool(key: key)
  }

  private func requireSystemModel(_ modelID: String) throws {
    guard modelID == AppleRuntimeIdentifiers.systemModelID else {
      throw AppleRuntimeFailure(
        code: "model_not_found",
        message: "Apple runtime does not provide model '\(modelID)'",
        retryable: false
      )
    }
  }
}
