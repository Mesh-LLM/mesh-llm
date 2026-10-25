export type CapabilityLevel =
  | { type: 'None' }
  | { type: 'Likely' }
  | { type: 'Supported' }

export type ServingModelState =
  | { type: 'Loading' }
  | { type: 'Ready' }
  | { type: 'Failed' }
  | { type: 'Unloading' }
  | { type: 'Stopped' }
  | { type: 'Unknown'; value: string }

export type ModelCapabilities = {
  multimodal: boolean
  vision: CapabilityLevel
  audio: CapabilityLevel
  reasoning: CapabilityLevel
  toolUse: CapabilityLevel
  moe: boolean
}

export type Model = {
  id: string
  name: string
}

export type ModelSummary = Model & {
  sizeLabel?: string | null
  description?: string | null
  capabilities: ModelCapabilities
}

export type ModelDetails = ModelSummary & {
  modelRef: string
  downloadRef: string
  path?: string | null
  sizeBytes?: number | null
  draft?: string | null
  installed: boolean
}

export type InstalledModel = {
  modelRef: string
  path: string
  sizeBytes?: number | null
  capabilities: ModelCapabilities
}

export type ChatMessage = {
  role: string
  content: string
}

export type ChatRequest = {
  model: string
  messages: ChatMessage[]
}

export type ResponsesRequest = {
  model: string
  input: string
}

export type InferenceResult = {
  requestId: string
  content: string
  events: unknown[]
}

export type OpenAIRequestBody = Record<string, unknown>

export type OpenAIResponse = {
  statusCode: number
  contentType?: string | null
  body: string
  json(): unknown
}

export type OpenAIStreamStarted = {
  type: 'started'
  requestId: string
  statusCode: number
  contentType?: string | null
}

export type OpenAIStreamSSE = {
  type: 'sse'
  requestId: string
  event?: string | null
  data: string
  raw: string
  done: boolean
  json(): unknown | null
}

export type OpenAIStreamEvent = OpenAIStreamStarted | OpenAIStreamSSE

export declare class OpenAIRequestError extends Error {
  readonly statusCode: number | null
  readonly body: string | null
}

export declare class Inference {
  listModels(): Promise<Model[]>
  request(path: string, body: OpenAIRequestBody, options?: { raiseForStatus?: boolean }): Promise<OpenAIResponse>
  chatCompletions(body: OpenAIRequestBody): Promise<unknown>
  responses(body: OpenAIRequestBody): Promise<unknown>
  stream(path: string, body: OpenAIRequestBody): AsyncGenerator<OpenAIStreamEvent, void, unknown>
  streamChatCompletions(body: OpenAIRequestBody): AsyncGenerator<OpenAIStreamEvent, void, unknown>
  streamResponses(body: OpenAIRequestBody): AsyncGenerator<OpenAIStreamEvent, void, unknown>
  cancel(requestId: string): Promise<void>
}

export type DevicePolicy = 'auto' | 'cpu' | 'gpu' | { gpu: string[] }

export type LoadModelOptions = {
  devicePolicy?: DevicePolicy
}

export type UnloadModelOptions = {
  drainTimeoutMs?: number
  force?: boolean
}

export type ServedModel = {
  modelRef: string
  modelId: string
  instanceId?: string | null
  state: ServingModelState
  backend?: string | null
  capabilities: ModelCapabilities
  contextLength?: number | null
  error?: string | null
}

export type ServingStatus = {
  enabled: boolean
  models: ServedModel[]
}

export type InstalledNativeRuntime = {
  meshVersion: string
  nativeRuntimeId: string
  flavor: string
  path: string
  skippyAbiVersion?: string | null
}

export type NativeRuntimeDownloadProgress = {
  nativeRuntimeId: string
  url: string
  downloadedBytes: number
  totalBytes?: number | null
  finished: boolean
}

export type NativeRuntimeInstallOptions = {
  meshVersion?: string
  skippyAbiVersion?: string
  selection?: string
  manifestPath?: string
  manifestUrl?: string
  bundleDirs?: string[]
  cacheDir?: string
  verificationPolicy?: 'require_checksum' | 'require_checksum_and_signature'
  allowDownload?: boolean
  onProgress?: (event: NativeRuntimeDownloadProgress) => void
}

export type NativeRuntimeResolveOptions = NativeRuntimeInstallOptions & {
  artifactDir?: string
  searchDirs?: string[]
}

export type NativeRuntimeInstallOutcome = {
  status: 'already_installed' | 'installed'
  runtime: InstalledNativeRuntime
  selectedNativeRuntimeId: string
  selectedSource: 'installed' | 'bundle' | 'download' | 'missing'
}

export type NativeRuntimePruneOptions = {
  cacheDir?: string
  activeMeshVersion?: string
  mode?: 'keep_active_and_previous' | 'active_only'
}

export type NativeRuntimePruneResult = {
  removedDirs: string[]
}

export type NodeMode = 'client' | 'serve' | 'combined'

export type NodeOptions = {
  mode?: NodeMode
  joinTokens?: string[]
  models?: string[]
  autoJoin?: boolean
  ownerKeyPath?: string
  apiPort?: number
  consolePort?: number
}

export declare class Node {
  static create(options?: NodeOptions): Node
  readonly inference: Inference
  start(): Promise<void>
  stop(): Promise<void>
  status(): Promise<{
    running: boolean
    mode: NodeMode
    apiBaseUrl?: string
    consoleUrl?: string
    payload?: unknown
  }>
  joinToken(token: string): Promise<void>
}

export declare function currentMeshVersion(): string
export declare function currentSkippyAbiVersion(): string
export declare function installNativeRuntime(options?: NativeRuntimeInstallOptions): Promise<NativeRuntimeInstallOutcome>
export declare function installedNativeRuntimes(options?: { cacheDir?: string }): Promise<InstalledNativeRuntime[]>
export declare function removeNativeRuntime(options: {
  cacheDir?: string
  meshVersion: string
  nativeRuntimeId: string
}): boolean
export declare function pruneNativeRuntimes(options?: NativeRuntimePruneOptions): NativeRuntimePruneResult
export declare function resolveNativeRuntime(config?: NativeRuntimeResolveOptions): Promise<InstalledNativeRuntime>
export declare function validateNativeRuntime(artifactDir: string, options?: NativeRuntimeInstallOptions): Promise<InstalledNativeRuntime>
