#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DOC="$ROOT/mesh/docs/SDK.md"
SWIFT_NODE="$ROOT/mesh/sdk/swift/Sources/MeshLLM/Node.swift"
SWIFT_STREAM="$ROOT/mesh/sdk/swift/Sources/MeshLLM/EventStream.swift"
KOTLIN_NODE="$ROOT/mesh/sdk/kotlin/src/main/kotlin/ai/meshllm/Node.kt"
NODE_SDK="$ROOT/mesh/sdk/node/index.js"
NODE_INFERENCE="$ROOT/mesh/sdk/node/inference.js"
NODE_NATIVE="$ROOT/mesh/crates/mesh-llm-nodejs/src/lib.rs"
NODE_TYPES="$ROOT/mesh/sdk/node/index.d.ts"
automation=(cargo xtool)
if [[ -n "${MESH_LLM_AUTOMATION_BIN:-}" ]]; then automation=("$MESH_LLM_AUTOMATION_BIN"); fi
PYTHON_SOURCE="$(cd "$ROOT" && "${automation[@]}" automation smoke-observation sdk-source --kind root)"
PYTHON_SDK="$PYTHON_SOURCE/sdk/src/meshllm/client.py"
PYTHON_TYPES="$PYTHON_SOURCE/sdk/src/meshllm/types.py"

missing=0

require() {
    local file="$1"
    local pattern="$2"
    local label="$3"
    if ! grep -Fq "$pattern" "$file"; then
        echo "missing SDK contract item: $label" >&2
        echo "  file: $file" >&2
        echo "  pattern: $pattern" >&2
        missing=1
    fi
}

required_doc_terms=(
    "Mesh LLM SDK Usage Guide"
    "client-only, serve-only, or combined mode"
    "SDK breaking change"
    "owner keystore path"
    "Native Runtime Artifacts"
    "Node.js"
)

for term in "${required_doc_terms[@]}"; do
    require "$DOC" "$term" "docs: $term"
done

swift_patterns=(
    "public typealias MeshError = FfiError"
    "public enum NodeMode"
    "public final class Node"
    "ownerKeyPath: String? = nil"
    "public func start() async throws"
    "public func stop() async throws"
    "public func status() async throws -> NodeStatus"
    "public func joinToken(_ token: String) async throws"
    "public let inference: Inference"
    "public func listModels() async throws"
    "public func request(path: String, body: [String: Any]) async throws"
    "public func chatCompletions(_ body: [String: Any]) async throws"
    "public func responses(_ body: [String: Any]) async throws"
    "public func stream(path: String, body: [String: Any])"
    "public func streamChatCompletions(_ body: [String: Any])"
    "public func streamResponses(_ body: [String: Any])"
)

for pattern in "${swift_patterns[@]}"; do
    require "$SWIFT_NODE" "$pattern" "swift: $pattern"
done
require "$SWIFT_STREAM" "class OpenAIStreamBridge" "swift: stream bridge"

kotlin_patterns=(
    "enum class NodeMode"
    "class Node internal constructor"
    "ownerKeyPath: String? = null"
    "suspend fun start()"
    "suspend fun stop()"
    "suspend fun status()"
    "suspend fun joinToken(token: String)"
    "val inference = Inference(handle)"
    "suspend fun listModels()"
    "suspend fun request(path: String, body: JsonObject)"
    "suspend fun chatCompletions(body: JsonObject)"
    "suspend fun responses(body: JsonObject)"
    "fun stream(path: String, bodyJson: String)"
    "fun streamChatCompletions(bodyJson: String)"
    "fun streamResponses(bodyJson: String)"
)

for pattern in "${kotlin_patterns[@]}"; do
    require "$KOTLIN_NODE" "$pattern" "kotlin: $pattern"
done

node_patterns=(
    "class Node"
    "static create(options = {})"
    "options.mode || 'client'"
    "options.ownerKeyPath || null"
    "this.inference = new Inference(handle)"
    "joinToken(token)"
)

for pattern in "${node_patterns[@]}"; do
    require "$NODE_SDK" "$pattern" "node: $pattern"
done

node_inference_patterns=(
    "class OpenAIRequestError"
    "async listModels()"
    "async request(path, body"
    "async chatCompletions(body)"
    "async responses(body)"
    "async *stream(path, body)"
    "streamChatCompletions(body)"
    "streamResponses(body)"
)

for pattern in "${node_inference_patterns[@]}"; do
    require "$NODE_INFERENCE" "$pattern" "node inference: $pattern"
done

require "$NODE_NATIVE" 'js_name = "openaiRequestJson"' "node native: buffered OpenAI request"
require "$NODE_NATIVE" 'js_name = "openaiStream"' "node native: OpenAI stream"

node_type_patterns=(
    "export declare class Node"
    "export type NodeMode = 'client' | 'serve' | 'combined'"
    "ownerKeyPath?: string"
    "readonly inference: Inference"
    "joinToken(token: string)"
    "export type InstalledNativeRuntime"
    "export type NativeRuntimeDownloadProgress"
    "export type NativeRuntimeInstallOptions"
    "export type NativeRuntimeResolveOptions"
    "export declare function installNativeRuntime"
    "export declare function resolveNativeRuntime"
    "export declare class Inference"
    "responses(body: OpenAIRequestBody)"
    "streamChatCompletions(body: OpenAIRequestBody)"
    "streamResponses(body: OpenAIRequestBody)"
)

for pattern in "${node_type_patterns[@]}"; do
    require "$NODE_TYPES" "$pattern" "node types: $pattern"
done

python_patterns=(
    "class Node"
    "class Inference"
    "owner_key_path: str | None = None"
    "async def start(self)"
    "async def stop(self)"
    "async def status(self)"
    "async def join_token(self, token: str)"
    "async def list_models(self)"
    "async def chat_completions(self"
    "async def responses(self"
    "async def request("
    "async def stream("
    "async def stream_chat_completions("
    "async def stream_responses("
)

for pattern in "${python_patterns[@]}"; do
    require "$PYTHON_SDK" "$pattern" "python: $pattern"
done

python_type_patterns=(
    "class MeshError"
    "class OpenAIRequestError"
    "class OpenAIResponse"
    "class OpenAIStreamStarted"
    "class OpenAIStreamChunk"
)

for pattern in "${python_type_patterns[@]}"; do
    require "$PYTHON_TYPES" "$pattern" "python types: $pattern"
done

if [[ "$missing" != "0" ]]; then
    exit 1
fi

echo "SDK contract check passed"
