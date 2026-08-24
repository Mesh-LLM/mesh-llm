import Foundation
@preconcurrency import Network

public final class LoopbackHTTPServer: @unchecked Sendable {
  private let runtime: AppleRuntime
  private let listener: NWListener
  private let queue = DispatchQueue(label: "mesh.apple.runtime.http")
  private let readyHandler: @Sendable (UInt16) -> Void

  public init(
    runtime: AppleRuntime,
    port: UInt16,
    onReady: @escaping @Sendable (UInt16) -> Void = { _ in }
  ) throws {
    guard let networkPort = NWEndpoint.Port(rawValue: port) else {
      throw AppleRuntimeFailure(
        code: "invalid_port",
        message: "Invalid REST port: \(port)",
        retryable: false
      )
    }
    self.runtime = runtime
    let parameters = NWParameters.tcp
    parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: networkPort)
    listener = try NWListener(using: parameters)
    readyHandler = onReady
  }

  public func run() async throws {
    try await withTaskCancellationHandler {
      try await withCheckedThrowingContinuation { continuation in
        let state = ListenerContinuation(continuation)
        listener.stateUpdateHandler = { [weak self, readyHandler] newState in
          switch newState {
          case .ready:
            readyHandler(self?.listener.port?.rawValue ?? 0)
          case .failed(let error):
            state.resume(throwing: error)
          case .cancelled:
            state.resume()
          default:
            break
          }
        }
        listener.newConnectionHandler = { [weak self] connection in
          self?.accept(connection)
        }
        listener.start(queue: queue)
      }
    } onCancel: {
      listener.cancel()
    }
  }

  private func accept(_ connection: NWConnection) {
    let reader = HTTPRequestReader(
      connection: connection,
      completion: { [weak self] request in
        guard let self else { return }
        let requestTask = Task {
          await self.handle(request, connection: connection)
        }
        self.monitorDisconnect(connection, task: requestTask)
      },
      failure: { [weak self] error in
        let failure = (error as? HTTPFailure)
          ?? HTTPFailure(status: 400, code: "invalid_request", message: String(describing: error))
        self?.sendError(failure, over: connection)
      }
    )
    connection.stateUpdateHandler = { state in
      if case .failed = state {
        connection.cancel()
      }
    }
    connection.start(queue: queue)
    reader.receive()
  }

  private func monitorDisconnect(
    _ connection: NWConnection,
    task: Task<Void, Never>
  ) {
    connection.receive(minimumIncompleteLength: 1, maximumLength: 1) {
      [weak self] _, _, isComplete, error in
      if isComplete || error != nil {
        task.cancel()
        return
      }
      self?.monitorDisconnect(connection, task: task)
    }
  }

  private func handle(_ request: HTTPRequest, connection: NWConnection) async {
    do {
      switch (request.method, request.path) {
      case ("GET", "/health"):
        try sendJSON(
          ["status": "ok", "runtime": AppleRuntimeIdentifiers.runtimeID],
          over: connection
        )
      case ("GET", "/v1/models"):
        try await sendModels(over: connection)
      case ("POST", "/v1/chat/completions"):
        try await handleChat(request.body, connection: connection)
      default:
        throw HTTPFailure(status: 404, code: "not_found", message: "Route not found")
      }
    } catch let failure as HTTPFailure {
      sendError(failure, over: connection)
    } catch let failure as AppleRuntimeFailure {
      sendError(
        HTTPFailure(status: 400, code: failure.code, message: failure.message),
        over: connection
      )
    } catch {
      sendError(
        HTTPFailure(status: 500, code: "internal_error", message: String(describing: error)),
        over: connection
      )
    }
  }

  private func sendModels(over connection: NWConnection) async throws {
    let status = await runtime.status()
    let models = status.models.flatMap { model -> [[String: Any]] in
      var alias = modelObject(model, id: model.modelID)
      alias["resolved_model"] = model.versionedModelID
      var versioned = modelObject(model, id: model.versionedModelID)
      versioned["alias_of"] = model.modelID
      return [alias, versioned]
    }
    try sendJSON(["object": "list", "data": models], over: connection)
  }

  private func modelObject(_ model: AppleModelStatus, id: String) -> [String: Any] {
    var object = [
        "id": model.modelID,
        "object": "model",
        "owned_by": "apple",
        "availability": model.availability,
        "context_length": model.contextSize,
        "capabilities": model.capabilities,
        "variant": model.variant,
      ] as [String: Any]
    object["id"] = id
    object["model_version"] = model.modelVersion
    object["version_source"] = model.versionSource
    object["versioned_model_id"] = model.versionedModelID
    return object
  }

  private func handleChat(_ body: Data, connection: NWConnection) async throws {
    let request: OpenAIChatRequest
    do {
      request = try JSONDecoder().decode(OpenAIChatRequest.self, from: body)
    } catch {
      throw HTTPFailure(status: 400, code: "invalid_json", message: String(describing: error))
    }
    guard AppleRuntimeIdentifiers.isSystemModelID(request.model) else {
      throw HTTPFailure(
        status: 404,
        code: "model_not_found",
        message: "Apple runtime does not provide model '\(request.model)'"
      )
    }
    if let tools = request.tools, !tools.isEmpty {
      try await handleFixtureTool(request, connection: connection)
      return
    }
    let prompt = try request.prompt()
    let generation = AppleGenerationRequest(
      modelID: request.model,
      prompt: prompt,
      instructions: request.instructions,
      maximumResponseTokens: request.responseTokenLimit,
      temperature: request.temperature
    )
    if request.stream == true {
      try await stream(generation, connection: connection)
      return
    }
    let result = try await runtime.generate(request: generation) { _ in }
    try sendJSON(chatResponse(result), over: connection)
  }

  private func handleFixtureTool(
    _ request: OpenAIChatRequest,
    connection: NWConnection
  ) async throws {
    guard request.stream != true else {
      throw HTTPFailure(
        status: 400,
        code: "unsupported_streaming_tool_demo",
        message: "The experimental REST fixture tool currently requires stream=false"
      )
    }
    let names = request.tools?.map(\.function.name) ?? []
    guard names == ["mesh_fixture_lookup"] else {
      throw HTTPFailure(
        status: 400,
        code: "unsupported_tool",
        message: "The experimental REST surface only supports mesh_fixture_lookup"
      )
    }
    let key = request.fixtureKey()
    let result = try await runtime.exerciseTool(modelID: request.model, key: key)
    let usage = usageObject(result.usage)
    try sendJSON(
      [
        "id": "chatcmpl-\(UUID().uuidString)",
        "object": "chat.completion",
        "created": Int(Date().timeIntervalSince1970),
        "model": result.modelID,
        "choices": [
          [
            "index": 0,
            "message": ["role": "assistant", "content": result.content],
            "finish_reason": "stop",
          ]
        ],
        "usage": usage,
        "mesh_tool_executions": result.invokedKeys.map { invokedKey in
          [
            "name": "mesh_fixture_lookup",
            "arguments": ["key": invokedKey],
            "result": "mesh-fixture-value-for-\(invokedKey)",
          ]
        },
      ],
      over: connection
    )
  }

  private func stream(
    _ request: AppleGenerationRequest,
    connection: NWConnection
  ) async throws {
    sendHeaders(
      status: 200,
      contentType: "text/event-stream",
      contentLength: nil,
      extra: ["Cache-Control": "no-cache", "Connection": "close"],
      over: connection
    )
    let completionID = "chatcmpl-\(UUID().uuidString)"
    do {
      _ = try await runtime.generate(request: request) { event in
        guard event.type == "delta", let delta = event.delta else { return }
        let payload: [String: Any] = [
          "id": completionID,
          "object": "chat.completion.chunk",
          "created": Int(Date().timeIntervalSince1970),
          "model": request.modelID,
          "choices": [
            [
              "index": 0,
              "delta": ["content": delta],
              "finish_reason": NSNull(),
            ]
          ],
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: payload),
          let json = String(data: data, encoding: .utf8)
        else { return }
        connection.send(content: Data("data: \(json)\n\n".utf8), completion: .idempotent)
      }
    } catch {
      let failure = (error as? AppleRuntimeFailure)
        ?? AppleRuntimeFailure(code: "internal_error", message: String(describing: error), retryable: false)
      let payload: [String: Any] = [
        "error": ["code": failure.code, "message": failure.message, "retryable": failure.retryable]
      ]
      if let data = try? JSONSerialization.data(withJSONObject: payload),
        let json = String(data: data, encoding: .utf8) {
        connection.send(content: Data("data: \(json)\n\n".utf8), completion: .idempotent)
      }
      connection.send(
        content: Data("data: [DONE]\n\n".utf8),
        contentContext: .finalMessage, isComplete: true,
        completion: .contentProcessed { _ in connection.cancel() })
      return
    }
    let finalPayload: [String: Any] = [
      "id": completionID,
      "object": "chat.completion.chunk",
      "created": Int(Date().timeIntervalSince1970),
      "model": request.modelID,
      "choices": [
        [
          "index": 0,
          "delta": [:],
          "finish_reason": "stop",
        ]
      ],
    ]
    let finalData = try JSONSerialization.data(withJSONObject: finalPayload)
    guard let finalJSON = String(data: finalData, encoding: .utf8) else { return }
    let trailer = Data("data: \(finalJSON)\n\ndata: [DONE]\n\n".utf8)
    connection.send(
      content: trailer, contentContext: .finalMessage, isComplete: true,
      completion: .contentProcessed { _ in connection.cancel() })
  }

  private func chatResponse(_ result: AppleGenerationResult) -> [String: Any] {
    [
      "id": "chatcmpl-\(result.requestID)",
      "object": "chat.completion",
      "created": Int(Date().timeIntervalSince1970),
      "model": result.modelID,
      "choices": [
        [
          "index": 0,
          "message": ["role": "assistant", "content": result.content],
          "finish_reason": "stop",
        ]
      ],
      "usage": usageObject(result.usage),
      "mesh_timing": [
        "elapsed_ms": result.elapsedMilliseconds,
        "time_to_first_token_ms": result.timeToFirstTokenMilliseconds.map { $0 as Any }
          ?? NSNull(),
      ],
    ]
  }

  private func usageObject(_ usage: AppleUsage) -> [String: Any] {
    [
      "prompt_tokens": usage.inputTokens,
      "completion_tokens": usage.outputTokens,
      "total_tokens": usage.inputTokens + usage.outputTokens,
      "prompt_tokens_details": ["cached_tokens": usage.cachedInputTokens],
      "completion_tokens_details": ["reasoning_tokens": usage.reasoningTokens],
    ]
  }

  private func sendJSON(_ object: Any, over connection: NWConnection) throws {
    let body = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    sendHeaders(
      status: 200,
      contentType: "application/json",
      contentLength: body.count,
      over: connection
    )
    connection.send(
      content: body, contentContext: .finalMessage, isComplete: true,
      completion: .contentProcessed { _ in connection.cancel() })
  }

  private func sendError(_ failure: HTTPFailure, over connection: NWConnection) {
    let object = [
      "error": [
        "message": failure.message,
        "type": "apple_runtime_error",
        "code": failure.code,
      ]
    ]
    guard let body = try? JSONSerialization.data(withJSONObject: object) else {
      connection.cancel()
      return
    }
    sendHeaders(
      status: failure.status,
      contentType: "application/json",
      contentLength: body.count,
      over: connection
    )
    connection.send(
      content: body, contentContext: .finalMessage, isComplete: true,
      completion: .contentProcessed { _ in connection.cancel() })
  }

  private func sendHeaders(
    status: Int,
    contentType: String,
    contentLength: Int?,
    extra: [String: String] = [:],
    over connection: NWConnection
  ) {
    let reason = status == 200 ? "OK" : "Error"
    var headers = [
      "HTTP/1.1 \(status) \(reason)",
      "Content-Type: \(contentType)",
    ]
    if let contentLength {
      headers.append("Content-Length: \(contentLength)")
    }
    for (name, value) in extra.sorted(by: { $0.key < $1.key }) {
      headers.append("\(name): \(value)")
    }
    headers.append("")
    headers.append("")
    connection.send(content: Data(headers.joined(separator: "\r\n").utf8), completion: .idempotent)
  }
}

private final class ListenerContinuation: @unchecked Sendable {
  private let lock = NSLock()
  private var continuation: CheckedContinuation<Void, any Error>?

  init(_ continuation: CheckedContinuation<Void, any Error>) {
    self.continuation = continuation
  }

  func resume() {
    take()?.resume()
  }

  func resume(throwing error: any Error) {
    take()?.resume(throwing: error)
  }

  private func take() -> CheckedContinuation<Void, any Error>? {
    lock.lock()
    defer { lock.unlock() }
    let value = continuation
    continuation = nil
    return value
  }
}

private final class HTTPRequestReader: @unchecked Sendable {
  private let connection: NWConnection
  private let completion: @Sendable (HTTPRequest) -> Void
  private let failure: @Sendable (any Error) -> Void
  private var buffer = Data()

  init(
    connection: NWConnection,
    completion: @escaping @Sendable (HTTPRequest) -> Void,
    failure: @escaping @Sendable (any Error) -> Void
  ) {
    self.connection = connection
    self.completion = completion
    self.failure = failure
  }

  func receive() {
    connection.receive(minimumIncompleteLength: 1, maximumLength: 1_048_576) {
      data, _, isComplete, error in
      if let data {
        self.buffer.append(data)
      }
      do {
        if let request = try HTTPRequest.parse(self.buffer) {
          self.completion(request)
          return
        }
      } catch {
        self.failure(error)
        return
      }
      if isComplete || error != nil {
        self.connection.cancel()
        return
      }
      if self.buffer.count > HTTPRequest.maximumHeaderBytes {
        self.failure(
          HTTPFailure(status: 400, code: "headers_too_large", message: "Request headers exceed limit")
        )
        return
      }
      self.receive()
    }
  }
}

private struct HTTPRequest: Sendable {
  fileprivate static let maximumHeaderBytes = 65_536
  private static let maximumRequestBodyBytes = 8 * 1_048_576
  let method: String
  let path: String
  let body: Data

  static func parse(_ data: Data) throws -> HTTPRequest? {
    let separator = Data("\r\n\r\n".utf8)
    guard let headerRange = data.range(of: separator) else { return nil }
    let headerData = data[..<headerRange.lowerBound]
    guard let header = String(data: headerData, encoding: .utf8) else {
      throw HTTPFailure(status: 400, code: "invalid_headers", message: "Headers are not UTF-8")
    }
    let lines = header.components(separatedBy: "\r\n")
    let requestLine = lines.first?.split(separator: " ") ?? []
    guard requestLine.count >= 2 else {
      throw HTTPFailure(status: 400, code: "invalid_request", message: "Invalid request line")
    }
    let contentLength =
      lines.dropFirst().compactMap { line -> Int? in
        let parts = line.split(separator: ":", maxSplits: 1)
        guard parts.count == 2,
          parts[0].trimmingCharacters(in: .whitespaces).lowercased() == "content-length"
        else { return nil }
        return Int(parts[1].trimmingCharacters(in: .whitespaces))
      }.first ?? 0
    guard contentLength >= 0, contentLength <= Self.maximumRequestBodyBytes else {
      throw HTTPFailure(
        status: 400,
        code: "invalid_content_length",
        message: "Content-Length is invalid or too large"
      )
    }
    let bodyStart = headerRange.upperBound
    guard data.count >= bodyStart + contentLength else { return nil }
    return HTTPRequest(
      method: String(requestLine[0]),
      path: String(requestLine[1]).components(separatedBy: "?").first ?? String(requestLine[1]),
      body: data.subdata(in: bodyStart..<(bodyStart + contentLength))
    )
  }
}

private struct OpenAIChatRequest: Decodable, Sendable {
  struct Message: Decodable, Sendable {
    let role: String
    let content: String?
  }

  struct ToolDefinition: Decodable, Sendable {
    struct Function: Decodable, Sendable {
      let name: String
    }

    let type: String
    let function: Function
  }

  let model: String
  let messages: [Message]
  let stream: Bool?
  let temperature: Double?
  let maxTokens: Int?
  let maxCompletionTokens: Int?
  let tools: [ToolDefinition]?

  enum CodingKeys: String, CodingKey {
    case model, messages, stream, temperature, tools
    case maxTokens = "max_tokens"
    case maxCompletionTokens = "max_completion_tokens"
  }

  var responseTokenLimit: Int? {
    maxCompletionTokens ?? maxTokens
  }

  var instructions: String? {
    let values = messages.filter { $0.role == "system" }.compactMap(\.content)
    return values.isEmpty ? nil : values.joined(separator: "\n")
  }

  func prompt() throws -> String {
    let values = messages.filter { $0.role != "system" }.compactMap { message in
      message.content.map { "\(message.role): \($0)" }
    }
    guard !values.isEmpty else {
      throw HTTPFailure(status: 400, code: "missing_prompt", message: "No text messages supplied")
    }
    return values.joined(separator: "\n")
  }

  func fixtureKey() -> String {
    let content = messages.last(where: { $0.role == "user" })?.content ?? ""
    guard let range = content.range(of: "key:", options: .caseInsensitive) else {
      return "rest-demo"
    }
    let suffix = content[range.upperBound...].trimmingCharacters(in: .whitespacesAndNewlines)
    return suffix.split(whereSeparator: \.isWhitespace).first.map(String.init) ?? "rest-demo"
  }
}

private struct HTTPFailure: Error {
  let status: Int
  let code: String
  let message: String
}
