import Foundation
import MeshLLM

@main
struct MeshExampleApp {
    static func main() async throws {
        let token = ProcessInfo.processInfo.environment["MESH_SDK_INVITE_TOKEN"]
        let node = try Node(
            mode: .client,
            joinTokens: token.map { [$0] } ?? [],
            autoJoin: token == nil,
            ownerKeyPath: ProcessInfo.processInfo.environment["MESH_SDK_OWNER_KEY_PATH"]
        )
        try await node.start()
        do {
            let models = try await node.inference.listModels()
            print("[models] \(models.count)")
            if let model = models.first {
                let response = try await node.inference.chatCompletions([
                    "model": model.id,
                    "messages": [["role": "user", "content": "hello"]],
                ])
                print(response.body)
            }
        } catch {
            try? await node.stop()
            throw error
        }
        try await node.stop()
    }
}
