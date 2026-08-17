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

  public func status() -> AppleModelStatus {
    let availability = availabilityFields(model.availability)
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
      // Xcode 27 beta's public arm64 module does not currently expose a
      // stable system-model build identifier. Keep this explicit rather
      // than deriving a false identity from the OS version.
      variant: "system-default-unversioned",
      capabilities: capabilities.sorted()
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
    let session = LanguageModelSession(
      model: model,
      instructions:
        "Classify the supplied text. Keep the explanation short. Confidence is 0 through 100."
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
    let session = LanguageModelSession(
      model: model,
      tools: [tool],
      instructions: "Call mesh_fixture_lookup once. Reply with only its output."
    )
    do {
      let response = try await session.respond(
        to: "Fixture key: \(key)",
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
