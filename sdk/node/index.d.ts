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

export type NativeRuntimeArtifact = {
  artifactId: string
  artifactDir: string
  manifest: string
  library: string
  metadata: Record<string, unknown>
}

export type NativeRuntimeConfig = {
  artifactDir?: string
  searchDirs?: string[]
}

export type NodeOptions = {
  ownerKeypairHex: string
  inviteToken: string
  cacheDir?: string
  runtimeDir?: string
  servingEnabled?: boolean
}

export type ConsoleOptions = {
  assetDir?: string
  port?: number
  listenAll?: boolean
}

export declare class Console {
  readonly url: string
  stop(): Promise<void>
}

export type ClientOptions = {
  ownerKeypairHex: string
  inviteToken: string
}

export declare class Client {
  static create(options: ClientOptions): Client
  readonly inference: {
    listModels(): Promise<Model[]>
    chat(request: ChatRequest, options?: { timeoutMs?: number }): Promise<InferenceResult>
    responses(request: ResponsesRequest, options?: { timeoutMs?: number }): Promise<InferenceResult>
    cancel(requestId: string): Promise<void>
  }
  start(): Promise<void>
  stop(): Promise<void>
  reconnect(): Promise<void>
  status(): Promise<{ connected: boolean; peerCount: number }>
}

export declare class Node {
  static create(options: NodeOptions): Node
  readonly inference: {
    listModels(): Promise<Model[]>
    chat(request: ChatRequest, options?: { timeoutMs?: number }): Promise<InferenceResult>
    responses(request: ResponsesRequest, options?: { timeoutMs?: number }): Promise<InferenceResult>
    cancel(requestId: string): Promise<void>
  }
  readonly models: {
    recommended(): Promise<ModelSummary[]>
    search(query: { query: string; limit?: number }): Promise<ModelSummary[]>
    show(modelRef: string): Promise<ModelDetails>
    installed(): Promise<InstalledModel[]>
    download(modelRef: string): Promise<{ modelRef: string; paths: string[]; primaryPath?: string | null }>
  }
  readonly serving: {
    status(): Promise<ServingStatus>
    load(modelRef: string, options?: LoadModelOptions): Promise<ServedModel>
    unload(target: { modelId: string } | { instanceId: string }, options?: UnloadModelOptions): Promise<void>
    unloadModel(modelId: string, options?: UnloadModelOptions): Promise<void>
    unloadInstance(instanceId: string, options?: UnloadModelOptions): Promise<void>
  }
  start(): Promise<void>
  stop(): Promise<void>
  reconnect(): Promise<void>
  status(): Promise<{ connected: boolean; peerCount: number }>
  startConsole(options?: ConsoleOptions): Promise<Console>
}

export declare function generateOwnerKeypairHex(): string
export declare function resolveNativeRuntime(config?: NativeRuntimeConfig): NativeRuntimeArtifact
export declare function validateNativeRuntime(artifactDir: string): NativeRuntimeArtifact
