import CoreAILanguageModels
import Foundation
import FoundationModels

private struct CoreAIArtifactConfiguration: Decodable {
  let id: String?
  let version: String?
  let path: String?
  let reference: String?
  let contextSize: Int?
  let languages: [String]?
}

/// Runs one explicitly configured `.aimodel` bundle through Apple's Core AI
/// Foundation Models adapter. The serving process owns one model instance;
/// Mesh scales independent requests across Macs instead of splitting a token
/// stream across nodes.
public actor CoreAIModelProvider {
  private let modelRoot: URL?
  private let artifactReference: CoreAIArtifactReference?
  private let artifactCacheDirectory: URL?
  private let modelID: String?
  private let modelVersion: String?
  private let contextSize: Int
  private let supportedLanguages: [String]
  private var model: CoreAILanguageModel?
  private var loadTask: Task<CoreAILanguageModel, Error>?
  private var artifactRoot: URL?
  private var artifactTask: Task<URL, Error>?

  public init(environment: [String: String] = ProcessInfo.processInfo.environment) {
    let artifact = Self.packageArtifactConfiguration()
    let configuredRoot = environment["MESH_APPLE_COREAI_MODEL_ROOT"]
      .map(URL.init(fileURLWithPath:))
    let packageRoot = artifact?.path.map { Self.packageRoot().appendingPathComponent($0) }
    modelRoot = configuredRoot ?? packageRoot
    artifactReference = environment["MESH_APPLE_COREAI_MODEL_REF"]
      .flatMap(CoreAIArtifactReference.init)
      ?? artifact?.reference.flatMap(CoreAIArtifactReference.init)
    artifactCacheDirectory = environment["MESH_APPLE_COREAI_MODEL_CACHE_DIR"]
      .map(URL.init(fileURLWithPath:))
    modelID = environment["MESH_APPLE_COREAI_MODEL_ID"]
      ?? artifact?.id
      ?? artifactReference?.repository
    modelVersion = environment["MESH_APPLE_COREAI_MODEL_VERSION"]
      ?? artifact?.version
      ?? artifactReference?.revision
    contextSize = Int(environment["MESH_APPLE_COREAI_CONTEXT_SIZE"] ?? "")
      ?? artifact?.contextSize
      ?? 4096
    supportedLanguages = (environment["MESH_APPLE_COREAI_LANGUAGES"]
      ?? artifact?.languages?.joined(separator: ",")
      ?? "en")
      .split(separator: ",")
      .map(String.init)
      .sorted()
  }

  public var isConfigured: Bool {
    (modelRoot != nil || artifactReference != nil)
      && modelID.map(AppleRuntimeIdentifiers.isArtifactModelID) == true
  }

  public func accepts(_ requestedModelID: String) -> Bool {
    isConfigured && (requestedModelID == modelID || requestedModelID == versionedModelID)
  }

  public func status(load: AppleProviderLoad) async -> AppleModelStatus? {
    guard isConfigured, let modelID, let modelVersion, let versionedModelID else { return nil }
    do {
      let model = try await loadModel()
      return AppleModelStatus(
        modelID: modelID,
        providerKind: "coreai",
        availability: "available",
        unavailableReason: nil,
        contextSize: contextSize,
        supportedLanguages: supportedLanguages,
        variant: "coreai-auto",
        modelVersion: modelVersion,
        versionSource: AppleRuntimeIdentifiers.coreAIModelVersionSource,
        versionedModelID: versionedModelID,
        capabilities: capabilities(for: model),
        load: load
      )
    } catch {
      return AppleModelStatus(
        modelID: modelID,
        providerKind: "coreai",
        availability: "unavailable",
        unavailableReason: "model_load_failed",
        contextSize: 0,
        supportedLanguages: [],
        variant: "coreai-auto",
        modelVersion: modelVersion,
        versionSource: AppleRuntimeIdentifiers.coreAIModelVersionSource,
        versionedModelID: versionedModelID,
        capabilities: [],
        load: load
      )
    }
  }

  public func prewarm() async throws {
    _ = try await loadModel()
  }

  public func generate(
    request: AppleGenerationRequest,
    onEvent: @Sendable (AppleRuntimeEvent) -> Void
  ) async throws -> AppleGenerationResult {
    let model = try await loadModel()
    let prompt = coreAIInputPrompt(
      modelIsReasoning: model.capabilities.contains(.reasoning),
      prompt: request.prompt
    )
    try await validateContextBudget(
      prompt: prompt,
      instructions: request.instructions,
      maximumResponseTokens: request.maximumResponseTokens
    )
    let session = LanguageModelSession(model: model, instructions: request.instructions)
    let options = GenerationOptions(
      temperature: request.temperature,
      maximumResponseTokens: request.maximumResponseTokens
    )
    let clock = ContinuousClock()
    let started = clock.now
    var firstTokenAt: ContinuousClock.Instant?
    var previous = ""
    var latestUsage = AppleUsage.zero

    do {
      let stream = session.streamResponse(to: prompt, options: options)
      for try await snapshot in stream {
        try Task.checkCancellation()
        let content = snapshot.content
        let delta = incrementalDelta(previous: previous, snapshot: content)
        if !delta.isEmpty {
          firstTokenAt = firstTokenAt ?? clock.now
          onEvent(
            AppleRuntimeEvent(
              type: "delta",
              requestID: request.requestID,
              modelID: request.modelID,
              delta: delta
            )
          )
        }
        previous = content
        latestUsage = AppleUsage(snapshot.usage)
      }
      try Task.checkCancellation()
    } catch {
      throw mapCoreAIError(error)
    }

    guard !previous.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      throw AppleRuntimeFailure(
        code: "empty_response",
        message: "Core AI ended without producing visible response content",
        retryable: true
      )
    }

    let result = AppleGenerationResult(
      requestID: request.requestID,
      modelID: request.modelID,
      content: previous,
      usage: latestUsage,
      elapsedMilliseconds: milliseconds(from: started, to: clock.now),
      timeToFirstTokenMilliseconds: firstTokenAt.map { milliseconds(from: started, to: $0) }
    )
    onEvent(
      AppleRuntimeEvent(
        type: "completed",
        requestID: request.requestID,
        modelID: request.modelID,
        content: result.content,
        usage: result.usage,
        elapsedMilliseconds: result.elapsedMilliseconds,
        timeToFirstTokenMilliseconds: result.timeToFirstTokenMilliseconds
      )
    )
    return result
  }

  public func generateStructured(modelID: String, prompt: String) async throws -> AppleStructuredResult {
    let model = try await loadModel()
    let instructions = "Classify the supplied text. Keep the explanation short. Confidence is 0 through 100."
    let prompt = coreAIInputPrompt(
      modelIsReasoning: model.capabilities.contains(.reasoning),
      prompt: prompt
    )
    try await validateContextBudget(
      prompt: prompt,
      instructions: instructions,
      schema: SpikeClassification.generationSchema,
      maximumResponseTokens: 128
    )
    do {
      let response = try await LanguageModelSession(model: model, instructions: instructions)
        .respond(
          to: prompt,
          generating: SpikeClassification.self,
          options: GenerationOptions(temperature: 0, maximumResponseTokens: 128)
        )
      return AppleStructuredResult(
        modelID: modelID,
        label: response.content.label,
        confidence: response.content.confidence,
        explanation: response.content.explanation,
        usage: AppleUsage(response.usage)
      )
    } catch {
      throw mapCoreAIError(error)
    }
  }

  public func exerciseTool(modelID: String, key: String) async throws -> AppleToolResult {
    let model = try await loadModel()
    let recorder = FixtureInvocationRecorder()
    let tool = FixtureLookupTool(recorder: recorder)
    let instructions = "Call mesh_fixture_lookup once. Reply with only its output."
    let prompt = coreAIInputPrompt(
      modelIsReasoning: model.capabilities.contains(.reasoning),
      prompt: "Fixture key: \(key)"
    )
    try await validateContextBudget(
      prompt: prompt,
      instructions: instructions,
      tools: [tool],
      maximumResponseTokens: 128
    )
    do {
      let response = try await LanguageModelSession(
        model: model,
        tools: [tool],
        instructions: instructions
      ).respond(
        to: prompt,
        options: GenerationOptions(
          temperature: 0,
          maximumResponseTokens: 128,
          toolCallingMode: .allowed
        )
      )
      var content = response.content
      var invokedKeys = await recorder.recordedKeys()
      if invokedKeys.isEmpty,
        let fallbackKey = fixtureToolKey(from: response.content)
      {
        content = try await tool.call(arguments: FixtureLookupArguments(key: fallbackKey))
        invokedKeys = await recorder.recordedKeys()
      }
      return AppleToolResult(
        modelID: modelID,
        content: content,
        invokedKeys: invokedKeys,
        usage: AppleUsage(response.usage)
      )
    } catch {
      throw mapCoreAIError(error)
    }
  }

  private var versionedModelID: String? {
    guard let modelID, let modelVersion else { return nil }
    return "\(modelID)@\(modelVersion)"
  }

  private func loadModel() async throws -> CoreAILanguageModel {
    if let model { return model }
    if let loadTask { return try await loadTask.value }
    let modelRoot = try await resolvedModelRoot()
    let task = Task { try await CoreAILanguageModel(resourcesAt: modelRoot) }
    loadTask = task
    do {
      let loaded = try await task.value
      model = loaded
      loadTask = nil
      return loaded
    } catch {
      loadTask = nil
      throw mapCoreAIError(error)
    }
  }

  private func resolvedModelRoot() async throws -> URL {
    if let modelRoot { return modelRoot }
    if let artifactRoot { return artifactRoot }
    if let artifactTask { return try await artifactTask.value }
    guard let artifactReference else {
      throw AppleRuntimeFailure(
        code: "coreai_model_not_configured",
        message: "Configure MESH_APPLE_COREAI_MODEL_ROOT or MESH_APPLE_COREAI_MODEL_REF",
        retryable: false
      )
    }
    let task = Task {
      try await CoreAIArtifactCache(
        reference: artifactReference,
        cacheDirectory: artifactCacheDirectory
      ).materialize()
    }
    artifactTask = task
    do {
      let root = try await task.value
      artifactRoot = root
      artifactTask = nil
      return root
    } catch {
      artifactTask = nil
      throw mapCoreAIError(error)
    }
  }

  private static func packageRoot() -> URL {
    URL(fileURLWithPath: CommandLine.arguments[0])
      .deletingLastPathComponent()
      .deletingLastPathComponent()
  }

  private static func packageArtifactConfiguration() -> CoreAIArtifactConfiguration? {
    let url = packageRoot().appendingPathComponent("Resources/coreai-model.json")
    guard let data = try? Data(contentsOf: url) else { return nil }
    return try? JSONDecoder().decode(CoreAIArtifactConfiguration.self, from: data)
  }

  private func validateContextBudget(
    prompt: String,
    instructions: String?,
    tools: [any Tool] = [],
    schema: GenerationSchema? = nil,
    maximumResponseTokens: Int?
  ) async throws {
    var inputTokens = estimatedTokenCount(prompt)
    if let instructions, !instructions.isEmpty { inputTokens += estimatedTokenCount(instructions) }
    if !tools.isEmpty { inputTokens += estimatedTokenCount(tools.map(\.description).joined(separator: " ")) }
    if schema != nil { inputTokens += 128 }
    try validateAppleContextBudget(
      contextSize: contextSize,
      inputTokens: inputTokens,
      maximumResponseTokens: maximumResponseTokens
    )
  }
}

func coreAIInputPrompt(modelIsReasoning: Bool, prompt: String) -> String {
  guard modelIsReasoning else { return prompt }
  let normalized = prompt.lowercased()
  guard !normalized.contains("/no_think"), !normalized.contains("/think") else {
    return prompt
  }
  return "/no_think\n\(prompt)"
}

func fixtureToolKey(from content: String) -> String? {
  guard let start = content.firstIndex(of: "{"),
    let end = content.lastIndex(of: "}"),
    start <= end
  else { return nil }
  let object = String(content[start...end])
  guard let data = object.data(using: .utf8),
    let value = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
    value["name"] as? String == "mesh_fixture_lookup",
    let arguments = value["arguments"] as? [String: Any]
  else { return nil }
  return arguments["key"] as? String
}

private func estimatedTokenCount(_ text: String) -> Int {
  max(1, Int(ceil(Double(text.utf8.count) / 4.0)))
}

private func capabilities(for model: CoreAILanguageModel) -> [String] {
  var values: [String] = []
  if model.capabilities.contains(.guidedGeneration) { values.append("guided_generation") }
  if model.capabilities.contains(.toolCalling) { values.append("tool_calling") }
  if model.capabilities.contains(.reasoning) { values.append("reasoning") }
  return values.sorted()
}

private func milliseconds(from start: ContinuousClock.Instant, to end: ContinuousClock.Instant) -> Int {
  let duration = start.duration(to: end)
  return Int(duration.components.seconds * 1_000)
    + Int(duration.components.attoseconds / 1_000_000_000_000_000)
}

private func mapCoreAIError(_ error: any Error) -> AppleRuntimeFailure {
  if let failure = error as? AppleRuntimeFailure { return failure }
  if error is CancellationError {
    return AppleRuntimeFailure(code: "cancelled", message: "Generation was cancelled", retryable: false)
  }
  if let error = error as? LanguageModelError {
    return AppleRuntimeFailure(
      code: String(describing: error), message: String(describing: error), retryable: false)
  }
  return AppleRuntimeFailure(
    code: "coreai_model_error", message: String(describing: error), retryable: false)
}
