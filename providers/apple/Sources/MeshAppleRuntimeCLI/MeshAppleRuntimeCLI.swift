import Darwin
import Foundation
import MeshAppleRuntime

@main
struct MeshAppleRuntimeCLI {
  static func main() async {
    do {
      try await run(arguments: Array(CommandLine.arguments.dropFirst()))
    } catch let failure as AppleRuntimeFailure {
      emit(
        AppleRuntimeEvent(
          type: failure.code == "cancelled" ? "cancelled" : "error",
          error: failure
        )
      )
      exit(failure.code == "cancelled" ? 0 : 1)
    } catch {
      emit(
        AppleRuntimeEvent(
          type: "error",
          error: AppleRuntimeFailure(
            code: "cli_error",
            message: String(describing: error),
            retryable: false
          )
        )
      )
      exit(1)
    }
  }

  private static func run(arguments: [String]) async throws {
    let command = arguments.first ?? "help"
    let options = try ParsedOptions(Array(arguments.dropFirst()))
    let runtime = AppleRuntime()

    switch command {
    case "status":
      emit(await runtime.status())
    case "prewarm":
      try await runtime.prewarm(
        modelID: options.modelID,
        promptPrefix: options.value("prefix")
      )
      emit(["type": "prewarmed", "modelID": options.modelID])
    case "generate":
      try await runGenerate(runtime: runtime, options: options)
    case "structured":
      let prompt =
        options.value("prompt")
        ?? "Classify this message: MeshLLM routes a request to an available Apple silicon node."
      emit(try await runtime.generateStructured(modelID: options.modelID, prompt: prompt))
    case "tool":
      emit(
        try await runtime.exerciseTool(
          modelID: options.modelID,
          key: options.value("key") ?? "milestone-zero"
        )
      )
    case "cancel":
      try await runCancellation(runtime: runtime, options: options)
    case "serve":
      try ParentWatchdog.install(parentPID: try options.intValue("parent-pid"))
      try await runServer(runtime: runtime, options: options)
    case "help", "--help", "-h":
      printHelp()
    default:
      throw AppleRuntimeFailure(
        code: "invalid_command",
        message: "Unknown command: \(command)",
        retryable: false
      )
    }
  }

  private static func runGenerate(
    runtime: AppleRuntime,
    options: ParsedOptions
  ) async throws {
    try ParentWatchdog.install(parentPID: try options.intValue("parent-pid"))
    guard let prompt = options.value("prompt") else {
      throw AppleRuntimeFailure(
        code: "invalid_request",
        message: "generate requires --prompt",
        retryable: false
      )
    }
    let request = AppleGenerationRequest(
      requestID: options.value("request-id") ?? UUID().uuidString,
      modelID: options.modelID,
      prompt: prompt,
      instructions: options.value("instructions"),
      maximumResponseTokens: try options.intValue("max-tokens"),
      temperature: try options.doubleValue("temperature")
    )
    _ = try await runtime.generate(request: request, onEvent: emit)
  }

  private static func runCancellation(
    runtime: AppleRuntime,
    options: ParsedOptions
  ) async throws {
    let cancelAfter = try options.intValue("after-ms") ?? 50
    let request = AppleGenerationRequest(
      modelID: options.modelID,
      prompt: options.value("prompt")
        ?? "Write a detailed multi-section history of distributed computing.",
      maximumResponseTokens: try options.intValue("max-tokens") ?? 512
    )
    let task = Task {
      try await runtime.generate(request: request, onEvent: emit)
    }
    try await Task.sleep(for: .milliseconds(cancelAfter))
    task.cancel()
    do {
      _ = try await task.value
      throw AppleRuntimeFailure(
        code: "cancellation_not_observed",
        message: "Generation completed before cancellation was observed",
        retryable: false
      )
    } catch let failure as AppleRuntimeFailure where failure.code == "cancelled" {
      emit(
        AppleRuntimeEvent(
          type: "cancelled",
          requestID: request.requestID,
          modelID: request.modelID,
          error: failure
        )
      )
    } catch is CancellationError {
      emit(
        AppleRuntimeEvent(
          type: "cancelled",
          requestID: request.requestID,
          modelID: request.modelID,
          error: AppleRuntimeFailure(
            code: "cancelled",
            message: "Generation was cancelled",
            retryable: false
          )
        )
      )
    }
  }

  private static func runServer(
    runtime: AppleRuntime,
    options: ParsedOptions
  ) async throws {
    let portValue = try options.intValue("port") ?? 11_435
    guard portValue >= 0, portValue <= Int(UInt16.max) else {
      throw AppleRuntimeFailure(
        code: "invalid_port",
        message: "--port must be between 0 and 65535",
        retryable: false
      )
    }
    let server = try LoopbackHTTPServer(
      runtime: runtime,
      port: UInt16(portValue)
    ) { port in
      emit(["type": "ready", "host": "127.0.0.1", "port": String(port)])
    }
    try await server.run()
  }

  private static func emit<T: Encodable>(_ value: T) {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    guard let data = try? encoder.encode(value),
      let line = String(data: data, encoding: .utf8)
    else {
      return
    }
    print(line)
    fflush(stdout)
  }

  private static func printHelp() {
    print(
      """
      mesh-apple-runtime commands:
        status
        prewarm [--model MODEL] [--prefix TEXT]
        generate --prompt TEXT [--model MODEL] [--instructions TEXT] [--max-tokens N] [--temperature N] [--parent-pid N]
        structured [--model MODEL] [--prompt TEXT]
        tool [--model MODEL] [--key TEXT]
        cancel [--model MODEL] [--after-ms N] [--max-tokens N] [--prompt TEXT]
        serve [--port N] [--parent-pid N]
      """
    )
  }
}

private struct ParsedOptions {
  private let values: [String: String]

  var modelID: String {
    value("model") ?? AppleRuntimeIdentifiers.systemModelID
  }

  init(_ arguments: [String]) throws {
    var values: [String: String] = [:]
    var index = 0
    while index < arguments.count {
      let key = arguments[index]
      guard key.hasPrefix("--"), index + 1 < arguments.count else {
        throw AppleRuntimeFailure(
          code: "invalid_arguments",
          message: "Expected --name value near '\(key)'",
          retryable: false
        )
      }
      values[String(key.dropFirst(2))] = arguments[index + 1]
      index += 2
    }
    self.values = values
  }

  func value(_ key: String) -> String? {
    values[key]
  }

  func intValue(_ key: String) throws -> Int? {
    guard let value = values[key] else { return nil }
    guard let parsed = Int(value) else {
      throw invalidNumber(key: key, value: value)
    }
    return parsed
  }

  func doubleValue(_ key: String) throws -> Double? {
    guard let value = values[key] else { return nil }
    guard let parsed = Double(value) else {
      throw invalidNumber(key: key, value: value)
    }
    return parsed
  }

  private func invalidNumber(key: String, value: String) -> AppleRuntimeFailure {
    AppleRuntimeFailure(
      code: "invalid_arguments",
      message: "--\(key) expects a number, got '\(value)'",
      retryable: false
    )
  }
}
