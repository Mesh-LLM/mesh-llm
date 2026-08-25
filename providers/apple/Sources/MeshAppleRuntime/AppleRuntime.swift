import Foundation

public actor AppleRuntime {
  private let systemModel: SystemModelProvider
  private let coreAIModel: CoreAIModelProvider
  private let scheduler: ProviderRequestScheduler

  public init() {
    systemModel = SystemModelProvider()
    coreAIModel = CoreAIModelProvider()
    scheduler = ProviderRequestScheduler()
  }

  public func status() async -> AppleRuntimeStatus {
    let load = await scheduler.snapshot()
    let systemModelStatus = await systemModel.status(load: load)
    let coreAIModelStatus = await coreAIModel.status(load: load)
    return AppleRuntimeStatus(
      runtimeID: AppleRuntimeIdentifiers.runtimeID,
      protocolVersion: AppleRuntimeIdentifiers.protocolVersion,
      operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
      models: [systemModelStatus, coreAIModelStatus].compactMap { $0 }
    )
  }

  public func supports(modelID: String) async -> Bool {
    if AppleRuntimeIdentifiers.isSystemModelID(modelID) { return true }
    return await coreAIModel.accepts(modelID)
  }

  public func prewarm(modelID: String, promptPrefix: String? = nil) async throws {
    try await requireModel(modelID)
    try await scheduler.withPermit {
      if AppleRuntimeIdentifiers.isSystemModelID(modelID) {
        try await self.systemModel.prewarm(promptPrefix: promptPrefix)
      } else {
        try await self.coreAIModel.prewarm()
      }
    }
  }

  public func generate(
    request: AppleGenerationRequest,
    onEvent: @Sendable (AppleRuntimeEvent) -> Void
  ) async throws -> AppleGenerationResult {
    try await requireModel(request.modelID)
    return try await scheduler.withPermit {
      if AppleRuntimeIdentifiers.isSystemModelID(request.modelID) {
        return try await self.systemModel.generate(request: request, onEvent: onEvent)
      }
      return try await self.coreAIModel.generate(request: request, onEvent: onEvent)
    }
  }

  public func generateStructured(
    modelID: String,
    prompt: String
  ) async throws -> AppleStructuredResult {
    try await requireModel(modelID)
    return try await scheduler.withPermit {
      if AppleRuntimeIdentifiers.isSystemModelID(modelID) {
        return try await self.systemModel.generateStructured(prompt: prompt)
      }
      return try await self.coreAIModel.generateStructured(modelID: modelID, prompt: prompt)
    }
  }

  public func exerciseTool(modelID: String, key: String) async throws -> AppleToolResult {
    try await requireModel(modelID)
    return try await scheduler.withPermit {
      if AppleRuntimeIdentifiers.isSystemModelID(modelID) {
        return try await self.systemModel.exerciseTool(key: key)
      }
      return try await self.coreAIModel.exerciseTool(modelID: modelID, key: key)
    }
  }

  private func requireModel(_ modelID: String) async throws {
    if AppleRuntimeIdentifiers.isSystemModelID(modelID) { return }
    guard await coreAIModel.accepts(modelID) else {
      throw AppleRuntimeFailure(
        code: "model_not_found",
        message: "Apple runtime does not provide model '\(modelID)'",
        retryable: false
      )
    }
  }
}
