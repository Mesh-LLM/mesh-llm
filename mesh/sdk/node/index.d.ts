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
  documentOcr(file: Uint8Array, options: DocumentOcrOptions, requestOptions?: { raiseForStatus?: boolean }): Promise<OpenAIResponse>
  stream(path: string, body: OpenAIRequestBody): AsyncGenerator<OpenAIStreamEvent, void, unknown>
  streamChatCompletions(body: OpenAIRequestBody): AsyncGenerator<OpenAIStreamEvent, void, unknown>
  streamResponses(body: OpenAIRequestBody): AsyncGenerator<OpenAIStreamEvent, void, unknown>
  cancel(requestId: string): Promise<void>
}

export type DocumentOcrOptions = {
  model: string
  filename?: string
  contentType?: string
  responseFormat?: 'json' | 'text' | 'markdown'
  maxPages?: number
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

/** Known values are listed for completion; newer peers may advertise other workloads. */
export type ModelWorkloadClass =
  | 'causal_generation' | 'embedding' | 'rerank' | 'encoder_decoder'
  | 'speech_synthesis' | 'decision' | 'speech_recognition' | 'document_ocr'
  | 'unknown' | (string & {})

export type MeshModelQuery = {
  workloadClass?: ModelWorkloadClass
  kind?: 'llm' | 'model' | 'virtual' | 'unknown'
  available?: boolean
  limit?: number
  cursor?: string
}

export type MeshModelMetadata = {
  workload_class?: ModelWorkloadClass
  architecture?: string
  parameter_size?: string
  parameter_count_b?: number
  quant?: string
  native_context_length?: number
  tokenizer?: string
  layer_count?: number
  embedding_size?: number
  head_count?: number
  kv_head_count?: number
  expert_count?: number
  active_expert_count?: number
}

export type MeshModelRuntime = {
  model_name: string
  identity_hash?: string
  context_length?: number
  ready: boolean
}

export type MeshModelProvider = {
  provider_id: string
  local: boolean
  endpoint_id?: string
  plugin?: string
  workload_class?: ModelWorkloadClass
  available: boolean
  operations: string[]
  capabilities?: Record<string, unknown>
  metadata?: MeshModelMetadata
  runtime?: MeshModelRuntime
  thinking?: unknown
  requires?: unknown
}

/** Catalog results retain the Mesh HTTP API's snake_case field names. */
export type MeshModel = {
  id: string
  display_name: string
  kind: 'llm' | 'model' | 'virtual' | 'unknown' | (string & {})
  workload_class?: ModelWorkloadClass
  availability: 'available' | 'unavailable' | 'unknown' | (string & {})
  operations: string[]
  metadata?: MeshModelMetadata
  providers: MeshModelProvider[]
  virtual_model?: unknown
}

export type MeshModelPage = {
  data: MeshModel[]
  next_cursor: string | null
}

export declare class Models {
  /** List all advertised workloads in this node's mesh; filters combine with AND. */
  list(query?: MeshModelQuery): Promise<MeshModelPage>
  /** Look up an exact public serving ID, including profile and quantization. */
  get(id: string): Promise<MeshModel>
}

export type NodeOptions = {
  mode?: NodeMode
  joinTokens?: string[]
  models?: string[]
  autoJoin?: boolean
  ownerKeyPath?: string
  apiPort?: number
  consolePort?: number
}

export type MeshMembershipConfig = {
  name: string
  joinTokens?: string[]
  trustedOwners?: string[]
  apiPort: number
  nodeKeyPath?: string | null
  enumerateHost?: boolean
}

export type MeshMembershipStatus = {
  name: string
  apiBaseUrl: string
  inviteToken: string
  endpointId: string
  peers: number
  joinError: string | null
}

/** Exact snake_case EmbeddedMeshNodeConfig schema; native defaults/validation apply. */
export type EmbeddedMeshNodeConfig = {
  mode?: 'client' | 'serve_only' | 'serve'
  http?: { api_port?: number; console_port?: number; console_ui?: boolean }
  serving?: { models?: string[]; max_vram_gb?: number | null }
  network?: {
    join_tokens?: string[]
    auto_join?: boolean
    discovery_mode?: 'nostr' | 'mdns'
    publish?: boolean
    mesh_name?: string | null
    region?: string | null
    node_name?: string | null
    iroh_relays?: string[]
    iroh_relay_auth?: Record<string, string>
    disable_iroh_relays?: boolean
    nostr_relays?: string[]
    bind_ip?: string | null
    bind_port?: number | null
    listen_all?: boolean
    enumerate_host?: boolean
  }
  admission?: {
    owner_key?: string | null
    owner_required?: boolean
    node_label?: string | null
    trust_policy?: 'off' | 'prefer_owned' | 'require_owned' | 'allowlist' | null
    trusted_owners?: string[]
    mesh_requirements?: {
      min_node_version?: string | null
      max_node_version?: string | null
      min_protocol_version?: number | null
      max_protocol_version?: number | null
      require_release_attestation?: boolean
      release_signer_keys?: string[]
    }
  }
  storage?: { config_path?: string | null; isolated_config?: boolean }
  log_format?: 'pretty' | 'json'
  startup_timeout?: { secs: number; nanos: number }
}

export declare class Node {
  static create(options?: NodeOptions): Node
  static fromConfig(config: EmbeddedMeshNodeConfig): Node
  readonly inference: Inference
  readonly models: Models
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
  startLiveTranscription(options?: LiveTranscriptionOptions): Promise<LiveTranscription>
  addMembership(config: MeshMembershipConfig): Promise<MeshMembershipStatus>
  removeMembership(name: string): Promise<boolean>
  memberships(): Promise<MeshMembershipStatus[]>
}

export type LiveTranscriptionOptions = {
  model?: string
  language?: string
  prompt?: string
  sampleRate?: number
  /** null uses explicit commit calls; omission uses server-side turn detection. */
  turnDetection?: { threshold?: number; prefixPaddingMs?: number; silenceDurationMs?: number } | null
}

export type LiveTranscriptionEvent =
  | { type: 'speech_started'; item_id: string; audio_start_ms: number }
  | { type: 'speech_stopped'; item_id: string; audio_end_ms: number }
  | { type: 'delta'; item_id: string; text: string }
  | { type: 'committed'; item_id: string }
  | { type: 'completed'; item_id: string; transcript: string }
  | { type: 'error'; message: string }

export declare class LiveTranscription implements AsyncIterable<LiveTranscriptionEvent> {
  sendAudio(samples: number[] | Int16Array): Promise<void>
  sendAudioF32(samples: number[] | Float32Array): Promise<void>
  commit(): Promise<void>
  nextEvent(): Promise<LiveTranscriptionEvent | null>
  stop(): Promise<LiveTranscriptionEvent[]>
  close(): Promise<void>
  events(): AsyncIterableIterator<LiveTranscriptionEvent>
  [Symbol.asyncIterator](): AsyncIterableIterator<LiveTranscriptionEvent>
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
