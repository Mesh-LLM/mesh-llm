import type { StatusPayload, MeshModelRaw, PeerInfo, GpuInfo } from '@/lib/api/types'
import { gpuAllocatableVramGB, gpuRatedVramGB, gpuReservedVramGB, gpuSystemReportedVramGB } from '@/lib/vram'
import { CONFIGURATION_HARNESS } from '@/features/app-tabs/data'
import type {
  ConfigurationDefaultsValues,
  ConfigurationHarnessData,
  ConfigurationModelPlacementPaths,
  ConfigurationSettingsHarnessData,
  ConfigAssign,
  ConfigAssignModelConfig,
  ConfigModel,
  ConfigNode
} from '@/features/app-tabs/types'
import {
  combineSettingsHarnessData,
  createConfigurationAttestationSettingsFromSchema,
  createConfigurationAuditSettingsFromSchema,
  createConfigurationIntegrationsFromSchema,
  createConfigurationMeshLLMSettingsFromSchema,
  createConfigurationModelSettingsFromSchema,
  createConfigurationNetworkSettingsFromSchema,
  createConfigurationRuntimeSettingsFromSchema,
  modelPlacementOptionsFromSchema,
  modelPlacementPathsFromSchema,
  overlayDefaultsValues
} from './config-adapter-schema'
import { DEFAULT_MODEL_PLACEMENT_PATHS } from './config-adapter-schema'
import type {
  RuntimeConfigControlStatePayload,
  RuntimeControlMeshConfig,
  RuntimeControlModelConfigEntry,
  RuntimeConfigSchemaReference
} from './config-adapter-types'
import { modelEntryPathSegments, readPath } from './config-adapter-paths'

function mapNodeState(state: string | undefined): 'online' | 'degraded' | 'offline' {
  if (state === 'client') return 'offline'
  if (state === 'loading') return 'degraded'
  if (state === 'standby' || state === 'serving') return 'online'
  return 'offline'
}

function finiteNumber(value: unknown, fallback = 0): number {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return fallback
}

function optionalPositiveNumber(value: unknown): number | undefined {
  const parsed = finiteNumber(value)
  return parsed > 0 ? parsed : undefined
}

function optionalNonEmptyString(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value.trim() : undefined
}

function parameterCountBFromText(text: string): number {
  const multiplied = [...text.matchAll(/(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*([bm])\b/gi)]
    .map((match) => {
      const left = Number(match[1])
      const right = Number(match[2])
      const unit = match[3]?.toLowerCase()
      if (!Number.isFinite(left) || !Number.isFinite(right)) return 0
      return unit === 'm' ? (left * right) / 1000 : left * right
    })
    .filter((value) => value > 0)
  const simple = [...text.matchAll(/(\d+(?:\.\d+)?)\s*([bm])\b/gi)]
    .map((match) => {
      const value = Number(match[1])
      const unit = match[2]?.toLowerCase()
      if (!Number.isFinite(value)) return 0
      return unit === 'm' ? value / 1000 : value
    })
    .filter((value) => value > 0)
  return Math.max(0, ...multiplied, ...simple)
}

function quantFromText(text: string): string | undefined {
  const stem = text.replace(/\.gguf$/i, '')
  const match = /(?:^|[-./:])((?:UD-)?Q\d[^-./:]*|IQ\d[^-./:]*|BF16|F16|F32)$/i.exec(stem)
  return match?.[1]
}

function familyFromModel(model: MeshModelRaw): string {
  if (model.family) return model.family
  const source = model.source_ref ?? model.name
  return source.split('/')[0] ?? 'unknown'
}

function resolvePeerId(peer: PeerInfo, fallbackIndex: number): string {
  return peer.node_id ?? peer.id ?? peer.hostname ?? `peer-${fallbackIndex}`
}

function adaptGpuToConfigGpu(gpu: GpuInfo, fallbackIndex: number) {
  const systemTotalGB = gpuSystemReportedVramGB(gpu) ?? 0
  const totalGB = gpuRatedVramGB(gpu) ?? systemTotalGB
  const reservedGB = gpuReservedVramGB(gpu)

  return {
    idx: finiteNumber(gpu.idx, fallbackIndex),
    name: gpu.name,
    totalGB,
    systemTotalGB,
    reservedGB: reservedGB > 0 ? reservedGB : undefined,
    allocatableGB: gpuAllocatableVramGB(gpu) ?? undefined
  }
}

function adaptLocalStatusToConfigNode(payload: StatusPayload): ConfigNode {
  return {
    id: payload.node_id,
    hostname: payload.hostname ?? payload.my_hostname ?? payload.node_id,
    region: payload.region ?? 'local',
    status: mapNodeState(payload.node_state),
    cpu: 'Local runtime',
    ramGB: 0,
    gpus: payload.gpus.map(adaptGpuToConfigGpu),
    placement: 'separate',
    memoryTopology: payload.my_is_soc ? 'unified' : 'discrete'
  }
}

function adaptPeerToConfigNode(peer: PeerInfo, fallbackIndex: number): ConfigNode {
  const id = resolvePeerId(peer, fallbackIndex)

  return {
    id,
    hostname: peer.hostname ?? id,
    region: peer.region ?? 'unknown',
    status: mapNodeState(peer.node_state ?? peer.state ?? peer.role?.toLowerCase()),
    cpu: peer.hardware_label ?? 'Unknown CPU',
    ramGB: 0,
    gpus: peer.gpus?.map(adaptGpuToConfigGpu) ?? [],
    placement: 'separate'
  }
}

function adaptModelToConfigModel(model: MeshModelRaw): ConfigModel {
  const quant = model.quantization ?? quantFromText(model.name) ?? quantFromText(model.source_file ?? '') ?? 'unknown'
  const sizeGB = finiteNumber(model.size_gb)
  const contextLength = finiteNumber(model.context_length)
  const paramsB = finiteNumber(model.params_b, parameterCountBFromText(`${model.name} ${model.display_name ?? ''}`))
  return {
    id: model.name,
    name: model.name,
    family: familyFromModel(model),
    paramsB,
    paramsLabel: paramsB > 0 ? `${paramsB}B` : undefined,
    quant,
    sizeGB,
    diskGB: finiteNumber(model.disk_gb, sizeGB),
    ctxMaxK: contextLength > 0 ? Math.round(contextLength / 1000) : 0,
    layers: optionalPositiveNumber(model.layer_count),
    heads: optionalPositiveNumber(model.head_count),
    embed: optionalPositiveNumber(model.embedding_size),
    tokenizer: optionalNonEmptyString(model.tokenizer),
    moe: model.capabilities?.moe ?? model.moe ?? false,
    vision: model.capabilities?.vision ?? model.vision ?? model.tags?.includes('vision') ?? false,
    tags: model.tags ?? []
  }
}

export function modelEntries(config: RuntimeControlMeshConfig | undefined): RuntimeControlModelConfigEntry[] {
  return Array.isArray(config?.models) ? config.models : []
}

export function modelNameFromEntry(
  entry: RuntimeControlModelConfigEntry,
  placementPaths: ConfigurationModelPlacementPaths
): string | undefined {
  const configured = readPath(entry, modelEntryPathSegments(placementPaths.model))
  const value = typeof configured === 'string' ? configured : entry.model
  return typeof value === 'string' && value.trim() ? value : undefined
}

function ctxFromModelEntry(
  entry: RuntimeControlModelConfigEntry,
  placementPaths: ConfigurationModelPlacementPaths
): number {
  const configured = readPath(entry, modelEntryPathSegments(placementPaths.ctxSize))
  const nested = readPath(entry, ['model_fit', 'ctx_size'])
  return Math.max(512, finiteNumber(configured ?? nested ?? entry.ctx_size, 4096))
}

function containerIdxFromModelEntry(
  entry: RuntimeControlModelConfigEntry,
  placementPaths: ConfigurationModelPlacementPaths
): number {
  const rawDevice =
    readPath(entry, modelEntryPathSegments(placementPaths.device)) ?? readPath(entry, ['hardware', 'device'])
  if (typeof rawDevice === 'string') {
    const match = rawDevice.match(/(\d+)$/)
    if (match) return finiteNumber(match[1])
  }
  const legacyGpu = entry['gpu_id']
  if (typeof legacyGpu === 'string') {
    const match = legacyGpu.match(/(\d+)$/)
    if (match) return finiteNumber(match[1])
  }
  return 0
}

function stringModelEntryValue(entry: RuntimeControlModelConfigEntry, path: string): string | undefined {
  const value = readPath(entry, modelEntryPathSegments(path))
  return typeof value === 'string' && value.trim() ? value : undefined
}

function numberModelEntryValue(entry: RuntimeControlModelConfigEntry, path: string): number | undefined {
  const value = readPath(entry, modelEntryPathSegments(path))
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function modelConfigFromEntry(
  entry: RuntimeControlModelConfigEntry,
  placementPaths: ConfigurationModelPlacementPaths
): ConfigAssignModelConfig | undefined {
  const config: ConfigAssignModelConfig = {}
  const slots = numberModelEntryValue(entry, 'models.<model-ref>.throughput.parallel')
  if (slots !== undefined) config.slots = Math.max(1, slots)
  const batch = numberModelEntryValue(entry, 'models.<model-ref>.model_fit.batch')
  const ubatch = numberModelEntryValue(entry, 'models.<model-ref>.model_fit.ubatch')
  if (batch === 512 && ubatch === 128) config.batchProfile = 'balanced'
  if (batch === 1024 && ubatch === 256) config.batchProfile = 'throughput'
  if (batch === 256 && ubatch === 64) config.batchProfile = 'saver'

  const splitMode = stringModelEntryValue(entry, 'models.<model-ref>.hardware.split_mode')
  if (splitMode === 'layer' || splitMode === 'row') config.splitMode = splitMode

  config.tensorSplit = stringModelEntryValue(entry, 'models.<model-ref>.hardware.tensor_split')
  config.mmproj = stringModelEntryValue(entry, placementPaths.mmproj ?? DEFAULT_MODEL_PLACEMENT_PATHS.mmproj!)
  config.draftModelPath = stringModelEntryValue(entry, 'models.<model-ref>.speculative.draft_model')

  const flashAttention = stringModelEntryValue(
    entry,
    placementPaths.flashAttention ?? DEFAULT_MODEL_PLACEMENT_PATHS.flashAttention!
  )
  if (flashAttention === 'enabled' || flashAttention === 'disabled') config.flashAttention = flashAttention

  config.cacheTypeK = stringModelEntryValue(
    entry,
    placementPaths.cacheTypeK ?? DEFAULT_MODEL_PLACEMENT_PATHS.cacheTypeK!
  )
  config.cacheTypeV = stringModelEntryValue(
    entry,
    placementPaths.cacheTypeV ?? DEFAULT_MODEL_PLACEMENT_PATHS.cacheTypeV!
  )

  const kvCachePolicy = stringModelEntryValue(
    entry,
    placementPaths.kvCachePolicy ?? DEFAULT_MODEL_PLACEMENT_PATHS.kvCachePolicy!
  )
  if (kvCachePolicy === 'quality' || kvCachePolicy === 'balanced' || kvCachePolicy === 'saver') {
    config.kvCachePolicy = kvCachePolicy
  }

  return Object.keys(config).length ? config : undefined
}

function modelAssignmentsFromMeshConfig(
  config: RuntimeControlMeshConfig | undefined,
  localNodeId: string,
  placementPaths: ConfigurationModelPlacementPaths
): ConfigAssign[] {
  return modelEntries(config)
    .map<ConfigAssign | null>((entry, index) => {
      const model = modelNameFromEntry(entry, placementPaths)
      if (!model) return null
      return {
        id: `configured-model-${index}`,
        modelId: model,
        nodeId: localNodeId,
        containerIdx: containerIdxFromModelEntry(entry, placementPaths),
        ctx: ctxFromModelEntry(entry, placementPaths),
        config: modelConfigFromEntry(entry, placementPaths)
      }
    })
    .filter((assign): assign is ConfigAssign => assign !== null)
}

function placeholderModelFromEntry(
  entry: RuntimeControlModelConfigEntry,
  placementPaths: ConfigurationModelPlacementPaths
): ConfigModel | undefined {
  const model = modelNameFromEntry(entry, placementPaths)
  if (!model) return undefined
  return {
    id: model,
    name: model,
    family: model.split('/')[0] ?? 'configured',
    paramsB: 0,
    quant: 'configured',
    sizeGB: 0,
    diskGB: 0,
    ctxMaxK: Math.max(1, Math.round(ctxFromModelEntry(entry, placementPaths) / 1000)),
    moe: false,
    vision: false,
    tags: ['Configured']
  }
}

function mergeCatalogWithConfiguredModels(
  catalog: ConfigModel[],
  config: RuntimeControlMeshConfig | undefined,
  placementPaths: ConfigurationModelPlacementPaths
): ConfigModel[] {
  const existingIds = new Set(catalog.map((model) => model.id))
  const configuredModels = modelEntries(config)
    .map((entry) => placeholderModelFromEntry(entry, placementPaths))
    .filter((model): model is ConfigModel => {
      if (model === undefined || existingIds.has(model.id)) return false
      existingIds.add(model.id)
      return true
    })
  return [...catalog, ...configuredModels]
}

export function adaptStatusToConfiguration(
  payload: StatusPayload,
  models: MeshModelRaw[],
  defaultsValues?: ConfigurationDefaultsValues,
  schema?: RuntimeConfigSchemaReference,
  config?: RuntimeControlMeshConfig,
  controlState?: RuntimeConfigControlStatePayload
): ConfigurationHarnessData {
  const nodes: ConfigNode[] = [adaptLocalStatusToConfigNode(payload), ...payload.peers.map(adaptPeerToConfigNode)]
  const localNodeId = nodes[0]?.id ?? payload.node_id
  const modelPlacementPaths = modelPlacementPathsFromSchema(schema)
  const modelPlacementOptions = modelPlacementOptionsFromSchema(schema)
  const catalog: ConfigModel[] = mergeCatalogWithConfiguredModels(
    models.map(adaptModelToConfigModel),
    config,
    modelPlacementPaths
  )
  const meshllmSettings = createConfigurationMeshLLMSettingsFromSchema(schema, controlState)
  const runtimeSettings = createConfigurationRuntimeSettingsFromSchema(schema, controlState)
  const modelSettings = createConfigurationModelSettingsFromSchema(schema, controlState)
  const network = createConfigurationNetworkSettingsFromSchema(schema, controlState)
  const attestation = createConfigurationAttestationSettingsFromSchema(schema, controlState)
  const auditSettings = createConfigurationAuditSettingsFromSchema(schema, controlState)
  const schemaIntegrations = createConfigurationIntegrationsFromSchema(schema, controlState)
  const overlay = (settings: ConfigurationSettingsHarnessData) =>
    defaultsValues ? overlayDefaultsValues(settings, defaultsValues) : settings
  const plugins =
    schemaIntegrations && defaultsValues
      ? overlayDefaultsValues(schemaIntegrations, defaultsValues)
      : schemaIntegrations
  const legacyDefaults = overlay(
    combineSettingsHarnessData(meshllmSettings, runtimeSettings, modelSettings, network, attestation, auditSettings)
  )

  return {
    ...CONFIGURATION_HARNESS,
    nodes,
    catalog,
    defaults: legacyDefaults,
    meshllm: overlay(meshllmSettings),
    runtimeSettings: overlay(runtimeSettings),
    modelSettings: overlay(modelSettings),
    network: overlay(network),
    attestation: overlay(attestation),
    audit: overlay(auditSettings),
    plugins,
    integrations: plugins,
    validationWarnings: undefined,
    attestationStatus: {
      owner: payload.owner,
      release_attestation: payload.release_attestation
    },
    modelPlacementPaths,
    modelPlacementOptions,
    modelConfigEntries: modelEntries(config),
    assigns: modelAssignmentsFromMeshConfig(config, localNodeId, modelPlacementPaths),
    preferredAssignId: undefined
  }
}
