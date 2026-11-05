import Foundation
import XCTest
@testable import MeshLLM
#if os(macOS)
import CryptoKit
import Darwin
import MachO
#endif

/// Opt in only with a freshly built native framework and isolated launch environment.
/// The enclosing test process must also have an operator-owned wall-clock deadline.
final class NativeLiveTranscriptionTests: XCTestCase {
    func testNativeSpeechDiscoveryValidationAndCancellation() async throws {
        guard ProcessInfo.processInfo.environment["MESH_SDK_LIVE_SPEECH"] == "1" else {
            throw XCTSkip("Set MESH_SDK_LIVE_SPEECH=1 to exercise the genuine native speech SDK")
        }
        #if canImport(MeshLLMFFI) && os(macOS)
        let inputs = try NativeSpeechInputs(testExecutable: Bundle(for: Self.self).executableURL)
        try await NativeSpeechFixture(inputs: inputs).run()
        #else
        throw NativeSpeechFailure("Opted-in speech requires the native macOS framework")
        #endif
    }
}

private struct NativeSpeechFailure: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

#if canImport(MeshLLMFFI) && os(macOS)
private func requireSpeech(_ condition: Bool, _ message: String) throws {
    if !condition { throw NativeSpeechFailure(message) }
}

private struct NativeSpeechInputs: Sendable {
    let model: URL
    let pcm: Data
    let evidence: URL
    let ports: [Int]
    let artifact: URL
    let artifactHash: String
    let testExecutable: URL
    let testHash: String
    let runtimeManifest: URL
    let runtimeManifestHash: String
    let runtimeHashes: Set<String>

    init(testExecutable: URL?) throws {
        let env = ProcessInfo.processInfo.environment
        func value(_ key: String) throws -> String {
            guard let value = env[key], !value.isEmpty else { throw NativeSpeechFailure("Missing \(key)") }
            return value
        }
        func path(_ key: String) throws -> URL {
            let raw = try value(key)
            try requireSpeech(raw.hasPrefix("/"), "\(key) must be absolute")
            // Foundation normalizes existing /private/tmp paths differently
            // from missing descendants. Resolve the existing ancestor first.
            var ancestor = URL(fileURLWithPath: raw).standardizedFileURL
            var missing: [String] = []
            while !FileManager.default.fileExists(atPath: ancestor.path) {
                let link = try? FileManager.default.destinationOfSymbolicLink(atPath: ancestor.path)
                try requireSpeech(link == nil, "\(key) contains an unresolved symbolic link")
                let parent = ancestor.deletingLastPathComponent()
                try requireSpeech(parent.path != ancestor.path, "\(key) has no existing ancestor")
                missing.append(ancestor.lastPathComponent)
                ancestor = parent
            }
            var resolved = ancestor.resolvingSymlinksInPath()
            for component in missing.reversed() { resolved.appendPathComponent(component) }
            return resolved
        }
        let source = try path("MESH_SPEECH_INPUTS")
        model = source.appendingPathComponent("ggml-tiny.bin")
        pcm = try Data(contentsOf: source.appendingPathComponent("jfk.s16le"))
        try requireSpeech(pcm.count == 352_000, "Expected exactly 11 seconds of PCM16LE")
        try requireSpeech(Self.hash(pcm) == "a29462b8ebd467318000e683b9117ade46230d3255ed2024e7db894abd9b38c9", "PCM digest mismatch")
        try requireSpeech(Self.hash(try Data(contentsOf: model)) == "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21", "Tiny model digest mismatch")
        artifact = try path("MESH_SDK_NATIVE_LIBRARY")
        artifactHash = Self.hash(try Data(contentsOf: artifact))
        try requireSpeech(artifactHash == value("MESH_SDK_NATIVE_SHA256"), "Native archive digest mismatch")
        guard let testExecutable else { throw NativeSpeechFailure("Cannot identify the loaded XCTest bundle") }
        self.testExecutable = testExecutable.resolvingSymlinksInPath()
        testHash = Self.hash(try Data(contentsOf: self.testExecutable))
        try requireSpeech(self.testExecutable == path("MESH_SDK_TEST_ARTIFACT"), "Loaded XCTest artifact differs from the admitted artifact")
        try requireSpeech(testHash == value("MESH_SDK_TEST_SHA256"), "Loaded XCTest digest mismatch")
        ports = try ["MESH_SDK_API_PORT", "MESH_SDK_CONSOLE_PORT", "MESH_SDK_BIND_PORT"].map {
            guard let port = Int(try value($0)), (1024...65535).contains(port) else {
                throw NativeSpeechFailure("Invalid \($0)")
            }
            return port
        }
        try requireSpeech(Set(ports).count == 3, "Fixture ports must be distinct")
        let state = try path("MESH_SDK_STATE_ROOT")
        try requireSpeech(state.path != NSHomeDirectory() && state.path != "/", "State root must be isolated")
        for key in ["MESH_LLM_CONFIG", "MESH_LLM_RUNTIME_ROOT", "MESH_LLM_DATA_DIR", "MESH_LLM_PLUGIN_DIR",
                    "MESH_LLM_NODE_KEY_PATH", "MESH_LLM_NATIVE_RUNTIME_CACHE_DIR", "XDG_CACHE_HOME",
                    "XDG_CONFIG_HOME", "XDG_RUNTIME_DIR", "HF_HUB_CACHE", "HF_XET_CACHE"] {
            let destination = try path(key)
            try requireSpeech(destination.path.hasPrefix(state.path + "/"), "\(key) must stay under the isolated state root")
            if FileManager.default.fileExists(atPath: destination.path) {
                let children = try FileManager.default.contentsOfDirectory(atPath: destination.path)
                try requireSpeech(children.isEmpty, "\(key) must be fresh")
            }
        }
        let runtimeRoot = try path("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR")
        runtimeManifest = try path("MESH_SDK_RUNTIME_MANIFEST")
        try requireSpeech(runtimeManifest.path.hasPrefix(runtimeRoot.path + "/"), "Runtime manifest must belong to the selected bundle")
        let runtimeBytes = try Data(contentsOf: runtimeManifest)
        runtimeManifestHash = Self.hash(runtimeBytes)
        try requireSpeech(runtimeManifestHash == value("MESH_SDK_RUNTIME_MANIFEST_SHA256"), "Runtime manifest digest mismatch")
        guard let manifest = try JSONSerialization.jsonObject(with: runtimeBytes) as? [String: Any],
              let runtime = manifest["runtime"] as? [String: Any], let files = runtime["files"] as? [String: String], !files.isEmpty else {
            throw NativeSpeechFailure("Runtime manifest must declare hashed runtime files")
        }
        runtimeHashes = Set(files.values)
        evidence = try path("MESH_SPEECH_EVIDENCE")
        if FileManager.default.fileExists(atPath: evidence.path) {
            try requireSpeech(FileManager.default.contentsOfDirectory(atPath: evidence.path).isEmpty, "Evidence directory must be fresh")
        }
        try FileManager.default.createDirectory(at: evidence, withIntermediateDirectories: true)
    }

    static func hash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }

    func write(_ name: String, _ value: [String: Any]) throws {
        let bytes = try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys])
        try bytes.write(to: evidence.appendingPathComponent(name), options: .atomic)
    }

    var config: [String: ModelJSONValue] {
        ["mode": .string("serve"),
         "http": .object(["api_port": .number(Double(ports[0])), "console_port": .number(Double(ports[1])), "console_ui": .bool(false)]),
         "serving": .object(["models": .array([.string(model.path)])]),
         "storage": .object(["isolated_config": .bool(true)]),
         "network": .object(["auto_join": .bool(false), "publish": .bool(false), "enumerate_host": .bool(true),
                             "bind_ip": .string("127.0.0.1"), "bind_port": .number(Double(ports[2])),
                             "disable_iroh_relays": .bool(true), "discovery_mode": .string("mdns")]),
         "startup_timeout": .object(["secs": .number(60), "nanos": .number(0)])]
    }
}

/// Unstructured calls let a deadline report failure without waiting for an uncancellable
/// FFI read. Cleanup closes the owned sessions and then requires every call to settle.
private actor NativeSpeechCalls {
    var active: Set<UUID> = []
    func begin(_ id: UUID) { active.insert(id) }
    func end(_ id: UUID) { active.remove(id) }
    var isEmpty: Bool { active.isEmpty }
}

private actor NativeSpeechResult<Value: Sendable> {
    var value: Result<Value, Error>?
    func finish(_ result: Result<Value, Error>) { value = result }
}

private actor NativeSpeechEvents {
    var values: [LiveTranscriptionEvent] = []
    var waiting = false
    var finished = false
    let file: URL
    let start = ContinuousClock.now
    init(file: URL) { self.file = file }
    func pending() { waiting = true }
    func end() { waiting = false; finished = true }
    var isPending: Bool { waiting && !finished && values.isEmpty }
    func record(_ event: LiveTranscriptionEvent, source: String) throws {
        values.append(event)
        let projection: [String: Any] = ["type": event.type, "item_id": event.itemId.map { $0 as Any } ?? NSNull(),
            "text": event.text.map { $0 as Any } ?? NSNull(), "transcript": event.transcript.map { $0 as Any } ?? NSNull(),
            "message": event.message.map { $0 as Any } ?? NSNull(),
            "audio_start_ms": event.audioStartMs.map { $0 as Any } ?? NSNull(),
            "audio_end_ms": event.audioEndMs.map { $0 as Any } ?? NSNull()]
        var data = try JSONSerialization.data(withJSONObject: ["elapsed": String(describing: start.duration(to: .now)), "source": source, "event": projection], options: [.sortedKeys])
        data.append(10)
        if !FileManager.default.fileExists(atPath: file.path) { try Data().write(to: file) }
        let handle = try FileHandle(forWritingTo: file)
        defer { try? handle.close() }
        try handle.seekToEnd(); try handle.write(contentsOf: data)
    }
}

private actor NativeSpeechSessions {
    var values: [LiveTranscription] = []
    func add(_ session: LiveTranscription) { values.append(session) }
}

private final class NativeSpeechFixture: @unchecked Sendable {
    let inputs: NativeSpeechInputs
    let calls = NativeSpeechCalls()
    let sessions = NativeSpeechSessions()
    init(inputs: NativeSpeechInputs) { self.inputs = inputs }

    func within<Value: Sendable>(_ seconds: Double, _ label: String,
                                _ operation: @escaping @Sendable () async throws -> Value) async throws -> Value {
        let result = NativeSpeechResult<Value>()
        let id = UUID()
        await calls.begin(id)
        let task = Task {
            do { await result.finish(.success(try await operation())) }
            catch { await result.finish(.failure(error)) }
            await calls.end(id)
        }
        let deadline = ContinuousClock.now.advanced(by: .milliseconds(Int64(seconds * 1000)))
        while ContinuousClock.now < deadline {
            if let value = await result.value { return try value.get() }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        task.cancel()
        throw NativeSpeechFailure("\(label) timed out; fallback cleanup is not a pass")
    }

    func run() async throws {
        try assertPortsFree()
        try inputs.write("identity.json", ["pid": ProcessInfo.processInfo.processIdentifier,
            "version": currentMeshVersion(), "native_archive": inputs.artifact.path, "native_archive_sha256": inputs.artifactHash,
            "test_executable": inputs.testExecutable.path, "test_sha256": inputs.testHash,
            "static_link_provenance": "Requires the separate exact-source producer/link receipt", "ports": inputs.ports])
        let node = try Node.fromConfig(inputs.config)
        var failure: Error?
        do {
            try await within(75, "node start") { try await node.start() }
            let running = try await within(5, "node status") { try await node.status().running }
            try requireSpeech(running, "Started node is not running")
            let model = try await within(45, "native discovery") { [self] in try await discover(node) }
            try recordLoadedRuntime()
            try await invalidBounds(node, model: model)
            try await transcribe(node, model: model)
            try await cancelCollector(node, model: model)
        } catch { failure = error }
        do { try await cleanup(node) } catch { failure = failure ?? error }
        try inputs.write("outcome.json", ["passed": failure == nil, "cleanup_completed": await calls.isEmpty])
        if let failure { throw failure }
    }

    func discover(_ node: Node) async throws -> String {
        let deadline = ContinuousClock.now.advanced(by: .seconds(45))
        while ContinuousClock.now < deadline {
            let page = try await within(10, "native model list") {
                try await node.models.list(MeshModelQuery(workloadClass: "speech_recognition", available: true))
            }
            if let model = page.data.first(where: Self.ready) {
                let exact = try await within(10, "native model get") { try await node.models.get(model.id) }
                try requireSpeech(exact.id == model.id && Self.ready(exact), "Native get must preserve the ready speech identity")
                try requireSpeech(exact.operations == model.operations && exact.workloadClass == model.workloadClass, "Native list/get capability mismatch")
                try inputs.write("native-discovery.json", ["id": exact.id, "kind": exact.kind,
                    "workload_class": exact.workloadClass ?? "", "availability": exact.availability,
                    "operations": exact.operations, "local_available_runtime_ready": true])
                return exact.id
            }
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        throw NativeSpeechFailure("Native catalog never advertised a ready local speech model")
    }

    static func ready(_ model: MeshModel) -> Bool {
        model.kind == "model" && model.workloadClass == "speech_recognition" && model.availability == "available"
            && model.operations.contains("live_transcription")
            && model.providers.contains { $0.local && $0.available && $0.runtime?.ready == true }
    }

    func recordLoadedRuntime() throws {
        let images = (0..<_dyld_image_count()).compactMap { index -> URL? in
            guard let name = _dyld_get_image_name(index) else { return nil }
            let path = URL(fileURLWithPath: String(cString: name)).resolvingSymlinksInPath()
            return ["libggml", "libllama", "libmtmd"].contains { path.lastPathComponent.hasPrefix($0) } ? path : nil
        }
        try requireSpeech(images.contains { $0.lastPathComponent.hasPrefix("libllama.") }, "No actual mapped llama runtime found")
        let mapped = try Set(images).sorted { $0.path < $1.path }.map { image -> [String: String] in
            let hash = NativeSpeechInputs.hash(try Data(contentsOf: image))
            try requireSpeech(inputs.runtimeHashes.contains(hash), "Mapped runtime image is not admitted by the manifest")
            return ["path": image.path, "sha256": hash]
        }
        try inputs.write("model-loaded-identity.json", ["runtime_manifest": inputs.runtimeManifest.path,
            "runtime_manifest_sha256": inputs.runtimeManifestHash, "mapped_runtime_images": mapped,
            "loaded_test_executable": inputs.testExecutable.path, "test_sha256": inputs.testHash])
    }

    func options(_ model: String, step: UInt32 = 400) -> LiveTranscriptionOptions {
        LiveTranscriptionOptions(model: model, language: "en", sampleRate: 16000, turnDetection: nil, stepMs: step, tentative: true)
    }

    func start(_ node: Node, model: String, step: UInt32 = 400) async throws -> LiveTranscription {
        try await within(35, "session start") { [self] in
            let session = try await node.startLiveTranscription(options(model, step: step))
            await sessions.add(session)
            return session
        }
    }

    func invalidBounds(_ node: Node, model: String) async throws {
        for step: UInt32 in [199, 5001] {
            var rejection: Error?
            do { _ = try await start(node, model: model, step: step) } catch { rejection = error }
            guard let rejection else { throw NativeSpeechFailure("Invalid step \(step) created a session") }
            let message = String(describing: rejection)
            try requireSpeech(message.contains("200") && message.contains("5000"), "Invalid step must report the supported range")
            try inputs.write("invalid-\(step).json", ["step_ms": step, "error_type": String(reflecting: type(of: rejection)), "message": message])
        }
    }

    func transcribe(_ node: Node, model: String) async throws {
        let session = try await start(node, model: model)
        let events = NativeSpeechEvents(file: inputs.evidence.appendingPathComponent("speech.jsonl"))
        let readerID = UUID()
        await calls.begin(readerID)
        let reader = Task {
            do {
                while let event = try await session.nextEvent() { try await events.record(event, source: "reader") }
                await events.end()
                await calls.end(readerID)
            } catch { await calls.end(readerID); throw error }
        }
        do {
            for offset in stride(from: 0, to: inputs.pcm.count, by: 3200) {
                let samples = stride(from: offset, to: offset + 3200, by: 2).map {
                    Int16(bitPattern: UInt16(inputs.pcm[$0]) | UInt16(inputs.pcm[$0 + 1]) << 8)
                }
                try await within(10, "audio send") { try await session.sendAudio(samples) }
                try await Task.sleep(nanoseconds: 100_000_000)
            }
            let tail = try await within(20, "speech stop") { try await session.stop() }
            try await within(5, "reader EOF") { try await reader.value }
            for event in tail { try await events.record(event, source: "stop") }
            try inputs.write("sent-audio.json", ["samples": 176000, "sends": 110, "sample_rate": 16000,
                "step_ms": 400, "tentative": true, "pcm_sha256": NativeSpeechInputs.hash(inputs.pcm)])
            try assertTranscript(await events.values)
            var closed = false
            do { _ = try await within(2, "repeat stop") { try await session.stop() } }
            catch { closed = String(describing: error).lowercased().contains("closed") }
            try requireSpeech(closed, "Repeated stop must report a closed session")
            let eof = try await within(2, "ended read") { try await session.nextEvent() }
            try requireSpeech(eof == nil, "Stopped reader must be at EOF")
        } catch {
            try? await within(3, "failed speech close") { await session.close() }
            _ = try? await within(3, "failed reader cleanup") { try await reader.value }
            throw error
        }
        try await within(3, "close speech") { await session.close(); await session.close() }
    }

    func assertTranscript(_ events: [LiveTranscriptionEvent]) throws {
        let completed = events.filter { $0.type == "completed" }
        let text = completed.first?.transcript ?? ""
        try inputs.write("transcript.json", ["transcript": text])
        try requireSpeech(completed.count == 1, "Expected exactly one completed utterance")
        try requireSpeech(events.filter { $0.type == "delta" }.compactMap(\.text).joined() == text, "Delta/completed mismatch")
        try requireSpeech(events.contains { $0.type == "tentative" && !($0.text ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }, "Missing tentative text")
        try requireSpeech(!events.contains { $0.type == "error" }, "Native speech emitted an error")
        try requireSpeech((events.firstIndex { $0.type == "completed" } ?? -1) > (events.lastIndex { $0.type == "delta" } ?? -1), "Completed must follow the final delta")
        let words = text.lowercased().split { !$0.isLetter && !$0.isNumber }.joined(separator: " ")
        try requireSpeech(words.range(of: "fellow americans.*ask not.*can do for you.*ask what you can do for your country", options: .regularExpression) != nil, "JFK clauses missing or out of order")
        for phrase in ["fellow americans", "can do for you", "what you can do", "do for your country"] {
            let regex = try NSRegularExpression(pattern: "\\b\(phrase)\\b")
            try requireSpeech(regex.numberOfMatches(in: words, range: NSRange(words.startIndex..., in: words)) == 1, "Expected one occurrence of \(phrase)")
        }
    }

    func cancelCollector(_ node: Node, model: String) async throws {
        let session = try await start(node, model: model)
        let observation = NativeSpeechEvents(file: inputs.evidence.appendingPathComponent("cancellation-events.jsonl"))
        let stream = session.events()
        let collectorID = UUID()
        await calls.begin(collectorID)
        let collector = Task {
            await observation.pending()
            do { for try await event in stream { try await observation.record(event, source: "collector") } }
            catch { await observation.end(); await calls.end(collectorID); throw error }
            await observation.end()
            await calls.end(collectorID)
        }
        do {
            try await within(2, "collector entered") {
                while !(await observation.waiting) { try await Task.sleep(nanoseconds: 10_000_000) }
            }
            try await Task.sleep(nanoseconds: 200_000_000)
            let pending = await observation.isPending
            try requireSpeech(pending, "Empty-session collector must be started and pending before cancellation")
            collector.cancel()
            try await within(3, "cancelled collector termination") {
                do { try await collector.value } catch is CancellationError { }
            }
            try requireSpeech(await observation.values.isEmpty, "Empty session emitted an event during cancellation")
            let eof = try await within(3, "cancellation native EOF") { try await session.nextEvent() }
            try requireSpeech(eof == nil, "Cancellation must close the native reader before explicit cleanup")
            try inputs.write("cancellation.json", ["collector_started": true, "pending_after_ms": 200,
                "collector_terminated": true, "native_eof_before_explicit_close": true])
        } catch {
            collector.cancel()
            try? await within(3, "fallback cancellation close") { await session.close() }
            _ = try? await within(3, "fallback collector join") { try await collector.value }
            throw error
        }
        try await within(3, "repeat cancellation close") { await session.close(); await session.close() }
        let eof = try await within(2, "repeat cancelled read") { try await session.nextEvent() }
        try requireSpeech(eof == nil, "Cancelled reader must stay at EOF")
    }

    func cleanup(_ node: Node) async throws {
        var failure: Error?
        for session in await sessions.values {
            do { try await within(5, "cleanup session") { await session.close() } } catch { failure = failure ?? error }
        }
        do { try await within(30, "node stop") { try await node.stop() } } catch { failure = failure ?? error }
        do {
            let running = try await within(5, "stopped status") { try await node.status().running }
            try requireSpeech(!running, "Node remains running after stop")
            let deadline = ContinuousClock.now.advanced(by: .seconds(5))
            while !(await calls.isEmpty) && ContinuousClock.now < deadline {
                try await Task.sleep(nanoseconds: 10_000_000)
            }
            try requireSpeech(await calls.isEmpty, "Native calls remain outstanding after cleanup")
            try assertPortsFree()
        } catch { failure = failure ?? error }
        try inputs.write("cleanup.json", ["passed": failure == nil, "owned_ports": inputs.ports])
        if let failure { throw failure }
    }

    func assertPortsFree() throws {
        for (index, port) in inputs.ports.enumerated() {
            let descriptor = socket(AF_INET, index == 2 ? SOCK_DGRAM : SOCK_STREAM, 0)
            try requireSpeech(descriptor >= 0, "Cannot create port-check socket")
            defer { Darwin.close(descriptor) }
            var address = sockaddr_in()
            address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
            address.sin_family = sa_family_t(AF_INET)
            address.sin_port = UInt16(port).bigEndian
            address.sin_addr.s_addr = inet_addr("127.0.0.1")
            if index != 2 {
                var reuse: Int32 = 1
                setsockopt(descriptor, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))
            }
            let bound = withUnsafePointer(to: &address) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.bind(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
            }
            try requireSpeech(bound == 0 && (index == 2 || listen(descriptor, 1) == 0), "Owned port \(port) is not free")
        }
    }
}
#endif
