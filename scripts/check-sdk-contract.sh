#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DOC="$ROOT/docs/SDK.md"
SWIFT_NODE="$ROOT/sdk/swift/Sources/MeshLLM/Node.swift"
SWIFT_STREAM="$ROOT/sdk/swift/Sources/MeshLLM/EventStream.swift"
KOTLIN_NODE="$ROOT/sdk/kotlin/src/main/kotlin/ai/meshllm/Node.kt"
NODE_SDK="$ROOT/sdk/node/index.js"
NODE_INFERENCE="$ROOT/sdk/node/inference.js"
NODE_NATIVE="$ROOT/crates/mesh-llm-nodejs/src/lib.rs"
NODE_TYPES="$ROOT/sdk/node/index.d.ts"
PYTHON_SDK="$ROOT/sdk/python/src/meshllm/client.py"
PYTHON_TYPES="$ROOT/sdk/python/src/meshllm/types.py"

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
    "MeshLLM SDK Usage Guide"
    "Language SDKs"
    "Native runtime artifacts"
    "Node Lifecycle"
    "Native Runtime Artifacts"
    "Node"
    "Client"
    "list mesh models"
    "download the model"
    "load the model through serving"
    "Serving unsupported"
    "Node.js"
)

for term in "${required_doc_terms[@]}"; do
    require "$DOC" "$term" "docs: $term"
done

swift_patterns=(
    "public typealias MeshError = FfiError"
    "public final class Client"
    "public func start()"
    "public func stop()"
    "public func reconnect()"
    "public func status() async"
    "public func listModels()"
    "public func chat(_ request: ChatRequest)"
    "public func responses(_ request: ResponsesRequest)"
    "public func request(path: String, body: [String: Any])"
    "public func responses(_ body: [String: Any])"
    "public func stream(path: String, body: [String: Any])"
    "public func streamChatCompletions(_ body: [String: Any])"
    "public func streamResponses(_ body: [String: Any])"
    "public func cancel(_ requestId: RequestId)"
    "public func recommended()"
    "public func search(_ query: ModelSearchQuery)"
    "public func show(_ modelRef: String)"
    "public func installed()"
    "public func cacheStatus()"
    "public func download(_ modelRef: String)"
    "public func delete(_ modelRef: String"
    "public func cleanup(_ policy: CleanupPolicy)"
    "public func pruneDerivedCache(_ policy: PrunePolicy)"
    "public func servedModels()"
    "public func load(_ modelRef: String"
    "public func unload(_ target: UnloadTarget"
    "public func unloadModel(_ modelId: String"
    "public func unloadInstance(_ instanceId: String"
    "public func setDevicePolicy(_ policy: DevicePolicy)"
    "public static func packaged"
)

for pattern in "${swift_patterns[@]}"; do
    require "$SWIFT_NODE" "$pattern" "swift: $pattern"
done
require "$SWIFT_STREAM" "func chatStream(_ request: ChatRequest)" "swift: chatStream"
require "$SWIFT_STREAM" "func responsesStream(_ request: ResponsesRequest)" "swift: responsesStream"

kotlin_patterns=(
    "typealias MeshException = uniffi.mesh_ffi.FfiException"
    "class Client"
    "suspend fun start()"
    "suspend fun stop()"
    "suspend fun reconnect()"
    "suspend fun status()"
    "suspend fun listModels()"
    "fun chat(request: ChatRequest"
    "fun responses(request: ResponsesRequest"
    "suspend fun request(path: String, body: JsonObject)"
    "suspend fun responses(body: JsonObject)"
    "fun stream(path: String, body: JsonObject)"
    "fun streamChatCompletions(body: JsonObject)"
    "fun streamResponses(body: JsonObject)"
    "fun cancel(requestId: RequestId)"
    "fun chatFlow(request: ChatRequest)"
    "fun responsesFlow(request: ResponsesRequest)"
    "suspend fun recommended()"
    "suspend fun search(query: ModelSearchQuery)"
    "suspend fun show(modelRef: String)"
    "suspend fun installed()"
    "suspend fun cacheStatus()"
    "suspend fun download(modelRef: String)"
    "suspend fun delete(modelRef: String"
    "suspend fun cleanup(policy: CleanupPolicy)"
    "suspend fun pruneDerivedCache(policy: PrunePolicy)"
    "suspend fun servedModels()"
    "suspend fun load(modelRef: String"
    "suspend fun unload(target: UnloadTarget"
    "suspend fun unloadModel(modelId: String"
    "suspend fun unloadInstance(instanceId: String"
    "suspend fun setDevicePolicy(policy: DevicePolicy)"
    "object ConsoleAssets"
    "fun packagedOptions("
)

for pattern in "${kotlin_patterns[@]}"; do
    require "$KOTLIN_NODE" "$pattern" "kotlin: $pattern"
done

node_patterns=(
    "class Client"
    "class Node"
    "static create(options)"
    "recommended()"
    "search(query)"
    "show(modelRef)"
    "installed()"
    "download(modelRef)"
    "load(modelRef"
    "unload(target"
    "unloadModel(modelId"
    "unloadInstance(instanceId"
)

for pattern in "${node_patterns[@]}"; do
    require "$NODE_SDK" "$pattern" "node: $pattern"
done

node_inference_patterns=(
    "class OpenAIRequestError"
    "async listModels()"
    "async chat(request"
    "async responsesText(request"
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
    "export declare class Client"
    "export declare class Node"
    "export type InstalledNativeRuntime"
    "export type NativeRuntimeDownloadProgress"
    "export type NativeRuntimeInstallOptions"
    "export type NativeRuntimeResolveOptions"
    "export declare function installNativeRuntime"
    "export declare function resolveNativeRuntime"
    "export declare function defaultConsoleAssetDir"
    "export declare class Inference"
    "responses(body: OpenAIRequestBody)"
    "streamChatCompletions(body: OpenAIRequestBody)"
    "streamResponses(body: OpenAIRequestBody)"
    "servingEnabled?: boolean"
    "load(modelRef: string"
)

for pattern in "${node_type_patterns[@]}"; do
    require "$NODE_TYPES" "$pattern" "node types: $pattern"
done

python_patterns=(
    "class Client"
    "class Node"
    "class Inference"
    "async def start(self)"
    "async def stop(self)"
    "async def reconnect(self)"
    "async def status(self)"
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
