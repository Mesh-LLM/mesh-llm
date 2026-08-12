import Foundation

public enum AppleRuntimeIdentifiers {
  public static let runtimeID = "apple/runtime"
  public static let protocolVersion = "0.1"
  public static let systemModelID = "apple/system"
  public static let systemModelVersionSource = "apple_os_release_band"

  public static var systemModelVersion: String? {
    documentedSystemModelVersion(for: ProcessInfo.processInfo.operatingSystemVersion)
  }

  public static var versionedSystemModelID: String? {
    systemModelVersion.map { "\(systemModelID)@\($0)" }
  }

  public static func isSystemModelID(_ modelID: String) -> Bool {
    guard let versionedSystemModelID else { return false }
    return modelID == systemModelID || modelID == versionedSystemModelID
  }

  public static func documentedSystemModelVersion(
    for operatingSystemVersion: OperatingSystemVersion
  ) -> String? {
    switch (operatingSystemVersion.majorVersion, operatingSystemVersion.minorVersion) {
    case (26, 0...3):
      return "26.0"
    case (26, 4...):
      return "26.4"
    case (27, _):
      return "27.0"
    default:
      return nil
    }
  }
}

public struct AppleRuntimeStatus: Codable, Equatable, Sendable {
  public let runtimeID: String
  public let protocolVersion: String
  public let operatingSystem: String
  public let models: [AppleModelStatus]

  public init(
    runtimeID: String,
    protocolVersion: String,
    operatingSystem: String,
    models: [AppleModelStatus]
  ) {
    self.runtimeID = runtimeID
    self.protocolVersion = protocolVersion
    self.operatingSystem = operatingSystem
    self.models = models
  }
}

public struct AppleModelStatus: Codable, Equatable, Sendable {
  public let modelID: String
  public let providerKind: String
  public let availability: String
  public let unavailableReason: String?
  public let contextSize: Int
  public let supportedLanguages: [String]
  public let variant: String
  public let modelVersion: String
  public let versionSource: String
  public let versionedModelID: String
  public let capabilities: [String]

  public init(
    modelID: String,
    providerKind: String,
    availability: String,
    unavailableReason: String?,
    contextSize: Int,
    supportedLanguages: [String],
    variant: String,
    modelVersion: String,
    versionSource: String,
    versionedModelID: String,
    capabilities: [String]
  ) {
    self.modelID = modelID
    self.providerKind = providerKind
    self.availability = availability
    self.unavailableReason = unavailableReason
    self.contextSize = contextSize
    self.supportedLanguages = supportedLanguages
    self.variant = variant
    self.modelVersion = modelVersion
    self.versionSource = versionSource
    self.versionedModelID = versionedModelID
    self.capabilities = capabilities
  }
}

public struct AppleGenerationRequest: Codable, Equatable, Sendable {
  public let requestID: String
  public let modelID: String
  public let prompt: String
  public let instructions: String?
  public let maximumResponseTokens: Int?
  public let temperature: Double?

  public init(
    requestID: String = UUID().uuidString,
    modelID: String = AppleRuntimeIdentifiers.systemModelID,
    prompt: String,
    instructions: String? = nil,
    maximumResponseTokens: Int? = nil,
    temperature: Double? = nil
  ) {
    self.requestID = requestID
    self.modelID = modelID
    self.prompt = prompt
    self.instructions = instructions
    self.maximumResponseTokens = maximumResponseTokens
    self.temperature = temperature
  }
}

public struct AppleUsage: Codable, Equatable, Sendable {
  public let inputTokens: Int
  public let cachedInputTokens: Int
  public let outputTokens: Int
  public let reasoningTokens: Int

  public init(
    inputTokens: Int,
    cachedInputTokens: Int,
    outputTokens: Int,
    reasoningTokens: Int
  ) {
    self.inputTokens = inputTokens
    self.cachedInputTokens = cachedInputTokens
    self.outputTokens = outputTokens
    self.reasoningTokens = reasoningTokens
  }
}

public struct AppleGenerationResult: Codable, Equatable, Sendable {
  public let requestID: String
  public let modelID: String
  public let content: String
  public let usage: AppleUsage
  public let elapsedMilliseconds: Int
  public let timeToFirstTokenMilliseconds: Int?

  public init(
    requestID: String,
    modelID: String,
    content: String,
    usage: AppleUsage,
    elapsedMilliseconds: Int,
    timeToFirstTokenMilliseconds: Int?
  ) {
    self.requestID = requestID
    self.modelID = modelID
    self.content = content
    self.usage = usage
    self.elapsedMilliseconds = elapsedMilliseconds
    self.timeToFirstTokenMilliseconds = timeToFirstTokenMilliseconds
  }
}

public struct AppleStructuredResult: Codable, Equatable, Sendable {
  public let modelID: String
  public let label: String
  public let confidence: Int
  public let explanation: String
  public let usage: AppleUsage

  public init(
    modelID: String,
    label: String,
    confidence: Int,
    explanation: String,
    usage: AppleUsage
  ) {
    self.modelID = modelID
    self.label = label
    self.confidence = confidence
    self.explanation = explanation
    self.usage = usage
  }
}

public struct AppleToolResult: Codable, Equatable, Sendable {
  public let modelID: String
  public let content: String
  public let invokedKeys: [String]
  public let usage: AppleUsage

  public init(
    modelID: String,
    content: String,
    invokedKeys: [String],
    usage: AppleUsage
  ) {
    self.modelID = modelID
    self.content = content
    self.invokedKeys = invokedKeys
    self.usage = usage
  }
}

public struct AppleRuntimeEvent: Codable, Equatable, Sendable {
  public let type: String
  public let requestID: String?
  public let modelID: String?
  public let delta: String?
  public let content: String?
  public let usage: AppleUsage?
  public let elapsedMilliseconds: Int?
  public let timeToFirstTokenMilliseconds: Int?
  public let error: AppleRuntimeFailure?

  public init(
    type: String,
    requestID: String? = nil,
    modelID: String? = nil,
    delta: String? = nil,
    content: String? = nil,
    usage: AppleUsage? = nil,
    elapsedMilliseconds: Int? = nil,
    timeToFirstTokenMilliseconds: Int? = nil,
    error: AppleRuntimeFailure? = nil
  ) {
    self.type = type
    self.requestID = requestID
    self.modelID = modelID
    self.delta = delta
    self.content = content
    self.usage = usage
    self.elapsedMilliseconds = elapsedMilliseconds
    self.timeToFirstTokenMilliseconds = timeToFirstTokenMilliseconds
    self.error = error
  }
}

public struct AppleRuntimeFailure: Error, Codable, Equatable, Sendable {
  public let code: String
  public let message: String
  public let retryable: Bool

  public init(code: String, message: String, retryable: Bool) {
    self.code = code
    self.message = message
    self.retryable = retryable
  }
}

public func incrementalDelta(previous: String, snapshot: String) -> String {
  guard snapshot.hasPrefix(previous) else {
    return snapshot
  }
  return String(snapshot.dropFirst(previous.count))
}
