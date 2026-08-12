import Foundation
import FoundationModels

@Generable
public struct SpikeClassification: Sendable {
  public var label: String
  public var confidence: Int
  public var explanation: String
}

@Generable
public struct FixtureLookupArguments: Sendable {
  public var key: String
}

public actor FixtureInvocationRecorder {
  private var keys: [String] = []

  public init() {}

  public func record(key: String) {
    keys.append(key)
  }

  public func recordedKeys() -> [String] {
    keys
  }
}

public struct FixtureLookupTool: Tool {
  public let name = "mesh_fixture_lookup"
  public let description = "Returns a deterministic fixture value for a key."
  private let recorder: FixtureInvocationRecorder

  public init(recorder: FixtureInvocationRecorder) {
    self.recorder = recorder
  }

  public func call(arguments: FixtureLookupArguments) async throws -> String {
    await recorder.record(key: arguments.key)
    return "mesh-fixture-value-for-\(arguments.key)"
  }
}

public actor SystemModelProvider {
  private let model: SystemLanguageModel
  private var preparedSession: LanguageModelSession?

  public init() {
    model = .default
  }

  public func status(load: AppleProviderLoad) -> AppleModelStatus? {
    let availability = availabilityFields(model.availability)
    guard let modelVersion = AppleRuntimeIdentifiers.systemModelVersion,
      let versionedModelID = AppleRuntimeIdentifiers.versionedSystemModelID
    else {
      return nil
    }
    var capabilities: [String] = []
    if model.capabilities.contains(.guidedGeneration) {
      capabilities.append("guided_generation")
    }
    if model.capabilities.contains(.toolCalling) {
      capabilities.append("tool_calling")
    }
    if model.capabilities.contains(.vision) {
      capabilities.append("vision")
    }
    if model.capabilities.contains(.reasoning) {
      capabilities.append("reasoning")
    }
    return AppleModelStatus(
      modelID: AppleRuntimeIdentifiers.systemModelID,
      providerKind: "system",
      availability: availability.state,
      unavailableReason: availability.reason,
      contextSize: model.contextSize,
      supportedLanguages: model.supportedLanguages
        .map(languageIdentifier)
        .sorted(),
      variant: "system-default-\(modelVersion)",
      modelVersion: modelVersion,
      versionSource: AppleRuntimeIdentifiers.systemModelVersionSource,
      versionedModelID: versionedModelID,
      capabilities: capabilities.sorted(),
      load: load
    )
  }

  public func prewarm(promptPrefix: String? = nil) throws {
    try requireAvailable()
    let session = LanguageModelSession(model: model)
    session.prewarm(promptPrefix: promptPrefix.map(Prompt.init))
    preparedSession = session
  }

  public func generate(
    request: AppleGenerationRequest,
    onEvent: @Sendable (AppleRuntimeEvent) -> Void
  ) async throws -> AppleGenerationResult {
    try requireAvailable()
    try await validateContextBudget(
      prompt: request.prompt,
      instructions: request.instructions,
      maximumResponseTokens: request.maximumResponseTokens
    )
    let session: LanguageModelSession
    if let preparedSession, request.instructions == nil {
      session = preparedSession
    } else {
      session = LanguageModelSession(
        model: model,
        instructions: request.instructions
      )
      session.prewarm(promptPrefix: Prompt(request.prompt))
    }
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
      let stream = session.streamResponse(to: request.prompt, options: options)
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
      // Foundation Models can finish a cancelled stream without throwing
      // when cancellation arrives before its first snapshot. Preserve the
      // provider contract by checking again before emitting completion.
      try Task.checkCancellation()
    } catch {
      throw mapProviderError(error)
    }

    let elapsed = milliseconds(from: started, to: clock.now)
    let firstToken = firstTokenAt.map { milliseconds(from: started, to: $0) }
    let result = AppleGenerationResult(
      requestID: request.requestID,
      modelID: request.modelID,
      content: previous,
      usage: latestUsage,
      elapsedMilliseconds: elapsed,
      timeToFirstTokenMilliseconds: firstToken
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

  public func generateStructured(prompt: String) async throws -> AppleStructuredResult {
    try requireAvailable()
    let instructions =
      "Classify the supplied text. Keep the explanation short. Confidence is 0 through 100."
    try await validateContextBudget(
      prompt: prompt,
      instructions: instructions,
      schema: SpikeClassification.generationSchema,
      maximumResponseTokens: 128
    )
    let session = LanguageModelSession(
      model: model,
      instructions: instructions
    )
    do {
      let response = try await session.respond(
        to: prompt,
        generating: SpikeClassification.self,
        options: GenerationOptions(temperature: 0, maximumResponseTokens: 128)
      )
      return AppleStructuredResult(
        modelID: AppleRuntimeIdentifiers.systemModelID,
        label: response.content.label,
        confidence: response.content.confidence,
        explanation: response.content.explanation,
        usage: AppleUsage(response.usage)
      )
    } catch {
      throw mapProviderError(error)
    }
  }

  public func exerciseTool(key: String) async throws -> AppleToolResult {
    try requireAvailable()
    let recorder = FixtureInvocationRecorder()
    let tool = FixtureLookupTool(recorder: recorder)
    let instructions = "Call mesh_fixture_lookup once. Reply with only its output."
    let prompt = "Fixture key: \(key)"
    try await validateContextBudget(
      prompt: prompt,
      instructions: instructions,
      tools: [tool],
      maximumResponseTokens: 32
    )
    let session = LanguageModelSession(
      model: model,
      tools: [tool],
      instructions: instructions
    )
    do {
      let response = try await session.respond(
        to: prompt,
        options: GenerationOptions(
          temperature: 0,
          maximumResponseTokens: 32,
          toolCallingMode: .allowed
        )
      )
      return AppleToolResult(
        modelID: AppleRuntimeIdentifiers.systemModelID,
        content: response.content,
        invokedKeys: await recorder.recordedKeys(),
        usage: AppleUsage(response.usage)
      )
    } catch {
      throw mapProviderError(error)
    }
  }

  private func requireAvailable() throws {
    guard case .available = model.availability else {
      let fields = availabilityFields(model.availability)
      throw AppleRuntimeFailure(
        code: fields.reason ?? "model_unavailable",
        message: "SystemLanguageModel is unavailable: \(fields.reason ?? "unknown")",
        retryable: fields.reason == "model_not_ready"
      )
    }
  }

  private func validateContextBudget(
    prompt: String,
    instructions: String?,
    tools: [any Tool] = [],
    schema: GenerationSchema? = nil,
    maximumResponseTokens: Int?
  ) async throws {
    var inputTokens = try await model.tokenCount(for: Prompt(prompt))
    if let instructions, !instructions.isEmpty {
      inputTokens += try await model.tokenCount(for: Instructions(instructions))
    }
    if !tools.isEmpty {
      inputTokens += try await model.tokenCount(for: tools)
    }
    if let schema {
      inputTokens += try await model.tokenCount(for: schema)
    }

    try validateAppleContextBudget(
      contextSize: model.contextSize,
      inputTokens: inputTokens,
      maximumResponseTokens: maximumResponseTokens
    )
  }
}

func validateAppleContextBudget(
  contextSize: Int,
  inputTokens: Int,
  maximumResponseTokens: Int?
) throws {
  let outputTokens = maximumResponseTokens ?? 0
  guard outputTokens >= 0 else {
    throw AppleRuntimeFailure(
      code: "invalid_maximum_response_tokens",
      message: "maximum response tokens cannot be negative",
      retryable: false
    )
  }
  guard inputTokens < contextSize, inputTokens + outputTokens <= contextSize else {
    throw AppleRuntimeFailure(
      code: "context_exceeded",
      message:
        "Apple system model context limit is \(contextSize) tokens; "
        + "request input uses \(inputTokens) and reserves \(outputTokens) response tokens",
      retryable: false
    )
  }
}

extension AppleUsage {
  fileprivate static let zero = AppleUsage(
    inputTokens: 0,
    cachedInputTokens: 0,
    outputTokens: 0,
    reasoningTokens: 0
  )

  fileprivate init(_ usage: LanguageModelSession.Usage) {
    self.init(
      inputTokens: usage.input.totalTokenCount,
      cachedInputTokens: usage.input.cachedTokenCount,
      outputTokens: usage.output.totalTokenCount,
      reasoningTokens: usage.output.reasoningTokenCount
    )
  }
}

private func availabilityFields(
  _ availability: SystemLanguageModel.Availability
) -> (state: String, reason: String?) {
  switch availability {
  case .available:
    return ("available", nil)
  case .unavailable(.deviceNotEligible):
    return ("unavailable", "device_not_eligible")
  case .unavailable(.appleIntelligenceNotEnabled):
    return ("unavailable", "apple_intelligence_not_enabled")
  case .unavailable(.modelNotReady):
    return ("unavailable", "model_not_ready")
  @unknown default:
    return ("unavailable", "unknown")
  }
}

private func languageIdentifier(_ language: Locale.Language) -> String {
  let code = language.languageCode?.identifier ?? "und"
  guard let region = language.region?.identifier else {
    return code
  }
  return "\(code)-\(region)"
}

private func milliseconds(
  from start: ContinuousClock.Instant,
  to end: ContinuousClock.Instant
) -> Int {
  let duration = start.duration(to: end)
  return Int(duration.components.seconds * 1_000)
    + Int(duration.components.attoseconds / 1_000_000_000_000_000)
}

private func mapProviderError(_ error: any Error) -> AppleRuntimeFailure {
  if let failure = error as? AppleRuntimeFailure {
    return failure
  }
  if error is CancellationError {
    return AppleRuntimeFailure(
      code: "cancelled",
      message: "Generation was cancelled",
      retryable: false
    )
  }
  if let error = error as? LanguageModelError {
    switch error {
    case .contextSizeExceeded:
      return failure("context_exceeded", error, retryable: false)
    case .rateLimited:
      return failure("rate_limited", error, retryable: true)
    case .guardrailViolation:
      return failure("guardrail_violation", error, retryable: false)
    case .refusal:
      return failure("refusal", error, retryable: false)
    case .unsupportedCapability:
      return failure("unsupported_capability", error, retryable: false)
    case .unsupportedTranscriptContent:
      return failure("unsupported_transcript", error, retryable: false)
    case .unsupportedGenerationGuide:
      return failure("unsupported_generation_guide", error, retryable: false)
    case .unsupportedLanguageOrLocale:
      return failure("unsupported_language", error, retryable: false)
    case .timeout:
      return failure("timeout", error, retryable: true)
    @unknown default:
      return failure("language_model_error", error, retryable: false)
    }
  }
  if let error = error as? LanguageModelSession.Error {
    switch error {
    case .concurrentRequests:
      return failure("provider_busy", error, retryable: true)
    case .transcriptMutationWhileResponding:
      return failure("transcript_mutation", error, retryable: false)
    @unknown default:
      return failure("session_error", error, retryable: false)
    }
  }
  if let error = error as? SystemLanguageModel.Error {
    return failure("assets_unavailable", error, retryable: true)
  }
  return failure("provider_error", error, retryable: false)
}

private func failure(
  _ code: String,
  _ error: any Error,
  retryable: Bool
) -> AppleRuntimeFailure {
  let localized = (error as? LocalizedError)?.errorDescription
  let reflected = String(reflecting: error)
  return AppleRuntimeFailure(
    code: code,
    message: localized.map { "\($0) [\(reflected)]" } ?? reflected,
    retryable: retryable
  )
}
