import Foundation
import MeshLLM

enum ExampleError: Error {
    case noModels
    case noTokenDelta
    case didNotComplete
    case chatFailed(String)
    case servingDidNotLoad(String)
}

let args = Array(CommandLine.arguments.dropFirst())
let inviteTokenArg = args.first { !$0.hasPrefix("--") }
let token = inviteTokenArg
    ?? ProcessInfo.processInfo.environment["MESH_SDK_INVITE_TOKEN"]
    ?? "local-mac-example"
let localModelRef = ProcessInfo.processInfo.environment["MESH_SDK_MODEL_REF"]
    ?? (inviteTokenArg == nil && ProcessInfo.processInfo.environment["MESH_SDK_INVITE_TOKEN"] == nil
        ? "Qwen2.5-3B-Instruct-Q4_K_M"
        : nil)
let remoteModelId = ProcessInfo.processInfo.environment["MESH_SDK_MODEL_ID"]
let cacheDir = ProcessInfo.processInfo.environment["MESH_SDK_CACHE_DIR"]
let runtimeDir = ProcessInfo.processInfo.environment["MESH_SDK_RUNTIME_DIR"]
let skipDownload = ProcessInfo.processInfo.environment["MESH_SDK_SKIP_DOWNLOAD"] == "1"
let prompt = ProcessInfo.processInfo.environment["MESH_SDK_PROMPT"]
    ?? "Say hello from this Mac in exactly seven words."

if localModelRef != nil {
    do {
        let runtime = try await NativeRuntime.resolve(
            NativeRuntimeResolveOptions(
                artifactDirectory: ProcessInfo.processInfo.environment["MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR"]
                    .map(URL.init(fileURLWithPath:)),
                allowDownload: ProcessInfo.processInfo.environment["MESH_SDK_RUNTIME_ALLOW_DOWNLOAD"] == "1"
            )
        )
        print("[runtime] artifact=\(runtime.nativeRuntimeId) path=\(runtime.path)")
    } catch {
        FileHandle.standardError.write(Data("[error] \(error)\n".utf8))
        exit(1)
    }
}

// Generate an ephemeral owner keypair for the example. In a real app this
// must be persisted across launches.
let ownerKeypairHex = generateOwnerKeypairHex()

Task {
    do {
        let node = try Node(
            inviteToken: InviteToken(token),
            ownerKeypairBytesHex: ownerKeypairHex,
            cacheDir: cacheDir,
            runtimeDir: runtimeDir
        )
        let recommended = try await node.models.recommended()
        let serving = try await node.serving.status()
        print("[node] recommended_models=\(recommended.count) serving_enabled=\(serving.enabled)")

        try await node.start()
        print("[connected]")

        let selectedModel: String
        let loadedInstanceId: String?
        if let localModelRef {
            if !skipDownload {
                let downloaded = try await node.models.download(localModelRef)
                print("[download] model_ref=\(downloaded.modelRef) files=\(downloaded.paths.count)")
            }

            let loaded = try await node.serving.load(
                localModelRef,
                options: LoadModelOptions(devicePolicy: .auto)
            )
            guard loaded.state == .ready else {
                throw ExampleError.servingDidNotLoad(loaded.error ?? "\(loaded.state)")
            }
            print("[serving] model=\(loaded.modelId) instance=\(loaded.instanceId ?? "-")")
            selectedModel = loaded.modelId
            loadedInstanceId = loaded.instanceId
        } else {
            loadedInstanceId = nil
            let models = try await waitForModels(node)
            print("[models] N=\(models.count)")
            guard !models.isEmpty else {
                throw ExampleError.noModels
            }
            selectedModel = remoteModelId ?? models[0].id
        }

        let request = ChatRequest(
            model: selectedModel,
            messages: [ChatMessage(role: "user", content: prompt)]
        )

        let startTime = Date()
        var firstToken = true
        var sawToken = false
        var completed = false
        chatLoop: for try await event in node.inference.chatStream(request) {
            switch event {
            case .tokenDelta(_, let delta):
                if firstToken {
                    let ms = Int(Date().timeIntervalSince(startTime) * 1000)
                    print("[chat] first_token_ms=\(ms)")
                    firstToken = false
                }
                sawToken = true
                print(delta, terminator: "")
            case .completed:
                completed = true
                print("\n[chat] done")
                break chatLoop
            case .failed(_, let error):
                throw ExampleError.chatFailed(error)
            default:
                break
            }
        }

        guard sawToken else {
            throw ExampleError.noTokenDelta
        }
        guard completed else {
            throw ExampleError.didNotComplete
        }

        if let instanceId = loadedInstanceId {
            try await node.serving.unloadInstance(
                instanceId,
                options: UnloadModelOptions(drainTimeoutMs: 1_000, force: false)
            )
            print("[serving] unloaded instance=\(instanceId)")
        }

        try await node.stop()
        print("[disconnect] ok")
    } catch {
        FileHandle.standardError.write(Data("[error] \(error)\n".utf8))
        exit(1)
    }
    exit(0)
}

RunLoop.main.run()

func waitForModels(_ node: Node) async throws -> [Model] {
    let deadline = Date().addingTimeInterval(30)
    while Date() < deadline {
        let models = try await node.inference.listModels()
        if !models.isEmpty {
            return models
        }
        try await Task.sleep(for: .milliseconds(250))
    }
    return []
}
