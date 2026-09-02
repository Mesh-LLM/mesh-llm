import type {
  ConfigurationDefaultsSetting,
  ConfigurationDefaultsValues,
  ConfigurationModelPlacementPaths,
  ConfigurationSettingValueSchema,
  ConfigAssignModelConfig,
  ConfigNode
} from '@/features/app-tabs/types'
import {
  evaluateSettingControlState,
  getSettingBaselineValue,
  getSettingDisabledReason,
  getSettingWriteDisposition
} from '@/features/configuration/lib/settings-utils'
import { objectArrayItemSchema, parseSchemaObjectArrayValue } from '@/features/configuration/lib/schema-object-array'
import {
  createConfigurationAuditSettingsFromSchema,
  createConfigurationAttestationSettingsFromSchema,
  createConfigurationIntegrationsFromSchema,
  createConfigurationMeshLLMSettingsFromSchema,
  createConfigurationModelSettingsFromSchema,
  createConfigurationNetworkSettingsFromSchema,
  createConfigurationRuntimeSettingsFromSchema,
  modelPlacementPathsFromSchema,
  pluginInstanceByName,
  DEFAULT_MODEL_PLACEMENT_PATHS
} from './config-adapter-schema'
import { modelEntries, modelNameFromEntry } from './config-adapter-status'
import {
  deletePath,
  modelEntryPathSegments,
  readPath,
  resolveConfigSettingPath,
  writePath
} from './config-adapter-paths'
import type {
  RuntimeConfigControlStatePayload,
  RuntimeConfigPluginInstance,
  RuntimeConfigSchemaReference,
  RuntimeControlApplyInput,
  RuntimeControlDiagnostic,
  RuntimeControlMeshConfig,
  RuntimeControlModelConfigEntry,
  RuntimeControlPluginConfigEntry
} from './config-adapter-types'

type MergeConfigurationIntoMeshConfigOptions = {
  includeModelAssignments?: boolean
  controlState?: RuntimeConfigControlStatePayload
}

class RuntimeControlSaveBlockedError extends Error {
  readonly diagnostics: readonly RuntimeControlDiagnostic[]

  constructor(diagnostics: readonly RuntimeControlDiagnostic[]) {
    super(diagnostics[0]?.message ?? 'Configuration save was blocked.')
    this.name = 'RuntimeControlSaveBlockedError'
    this.diagnostics = diagnostics
  }
}

function hasSchemaKind(
  schema: ConfigurationSettingValueSchema,
  kind: ConfigurationSettingValueSchema['kind']
): boolean {
  if (schema.kind === kind) return true
  if (schema.kind === 'one_of') return schema.variants.some((variant) => hasSchemaKind(variant, kind))
  return false
}

function normalizedChoiceValue(value: string) {
  if (value === 'true') return 'on'
  if (value === 'false') return 'off'
  return value
}

function serializeDefaultSettingValue(setting: ConfigurationDefaultsSetting, value: unknown): string | undefined {
  if (value == null) return undefined
  if (typeof value === 'boolean' && setting.control.kind === 'choice') {
    const optionValues = new Set(setting.control.options.map((option) => option.value))
    if (optionValues.has('on') && optionValues.has('off')) return value ? 'on' : 'off'
  }
  if (typeof value === 'string' && setting.control.kind === 'choice') return normalizedChoiceValue(value)
  if (Array.isArray(value) && setting.control.kind === 'text') {
    return objectArrayItemSchema(setting.valueSchema) ? JSON.stringify(value) : value.join(',')
  }
  if (value && typeof value === 'object' && setting.control.kind === 'text') {
    if (
      setting.canonicalPath === 'telemetry.headers' &&
      !Array.isArray(value) &&
      Object.keys(value as Record<string, unknown>).length === 0
    )
      return ''
    return JSON.stringify(value)
  }
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  if (setting.control.kind === 'choice') return String(value)
  return undefined
}

function parseArrayItemValue(schema: ConfigurationSettingValueSchema, value: string): unknown {
  if (schema.kind === 'integer') {
    const parsed = Number(value)
    if (Number.isInteger(parsed)) return parsed
  }
  if (schema.kind === 'float') {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  if (schema.kind === 'boolean') {
    if (value === 'true' || value === 'on') return true
    if (value === 'false' || value === 'off') return false
  }
  return value
}

function parseDefaultSettingValue(setting: ConfigurationDefaultsSetting, value: string): unknown {
  if (setting.control.kind === 'choice') {
    const optionValues = new Set(setting.control.options.map((option) => option.value))
    if (optionValues.has('on') && optionValues.has('off')) {
      if (value === 'on') return true
      if (value === 'off') return false
    }
    if (optionValues.has('on') && optionValues.has('off') && !optionValues.has('auto')) {
      return value === 'on'
    }
    if (hasSchemaKind(setting.valueSchema ?? { kind: 'string' }, 'integer')) {
      const parsed = Number(value)
      if (Number.isInteger(parsed)) return parsed
    }
  }
  if (setting.control.kind === 'range') {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : value
  }
  if (setting.control.kind === 'text') {
    if (setting.valueSchema?.kind === 'object') {
      try {
        return JSON.parse(value)
      } catch {
        return value
      }
    }
    if (setting.valueSchema?.kind === 'array') {
      const arraySchema = setting.valueSchema
      if (objectArrayItemSchema(arraySchema)) return parseSchemaObjectArrayValue(value) ?? value
      return value
        .split(',')
        .map((item) => item.trim())
        .filter(Boolean)
        .map((item) => parseArrayItemValue(arraySchema.items, item))
    }
    if (hasSchemaKind(setting.valueSchema ?? { kind: 'string' }, 'integer')) {
      const parsed = Number(value)
      if (Number.isInteger(parsed)) return parsed
    }
    if (hasSchemaKind(setting.valueSchema ?? { kind: 'string' }, 'float')) {
      const parsed = Number(value)
      if (Number.isFinite(parsed)) return parsed
    }
  }
  return value
}

function cloneMeshConfig(config: RuntimeControlMeshConfig): RuntimeControlMeshConfig {
  return JSON.parse(JSON.stringify(config)) as RuntimeControlMeshConfig
}

function pluginEntries(config: RuntimeControlMeshConfig): RuntimeControlPluginConfigEntry[] {
  return Array.isArray(config.plugin) ? config.plugin : []
}

function pluginEntryByName(
  config: RuntimeControlMeshConfig,
  pluginName: string
): RuntimeControlPluginConfigEntry | undefined {
  return pluginEntries(config).find((entry) => entry.name === pluginName)
}

function parsePluginCanonicalPath(
  setting: Pick<ConfigurationDefaultsSetting, 'canonicalPath' | 'tomlSection' | 'tomlKey'>
) {
  const canonicalPath = setting.canonicalPath
  const sectionSegments = setting.tomlSection?.split('.').filter(Boolean) ?? []
  if (canonicalPath?.startsWith('plugin.') && sectionSegments[0] === 'plugin' && setting.tomlKey) {
    const hasSettingsSection = sectionSegments.at(-1) === 'settings'
    const pluginName = sectionSegments.slice(1, hasSettingsSection ? -1 : undefined).join('.')
    if (!pluginName) return undefined

    return {
      pluginName,
      path: hasSettingsSection ? ['settings', setting.tomlKey] : [setting.tomlKey]
    }
  }

  const match = canonicalPath?.match(/^plugin\.([^.]+)\.(.+)$/)
  if (!match) return undefined

  return { pluginName: match[1], path: match[2].split('.').filter(Boolean) }
}

function readPluginConfigPath(
  config: RuntimeControlMeshConfig,
  setting: Pick<ConfigurationDefaultsSetting, 'canonicalPath' | 'tomlSection' | 'tomlKey'>
): unknown {
  const parsed = parsePluginCanonicalPath(setting)
  if (!parsed) return undefined

  const plugin = pluginEntryByName(config, parsed.pluginName)
  if (!plugin) return undefined

  return readPath(plugin, parsed.path)
}

function ensurePluginEntry(
  config: RuntimeControlMeshConfig,
  pluginName: string,
  instance?: RuntimeConfigPluginInstance
): RuntimeControlPluginConfigEntry {
  const existing = pluginEntryByName(config, pluginName)
  if (existing) return existing

  const nextEntry: RuntimeControlPluginConfigEntry = {
    name: pluginName,
    ...(instance?.enabled === false ? { enabled: false } : {})
  }
  config.plugin = [...pluginEntries(config), nextEntry]
  return nextEntry
}

function shouldPreserveDisabledPluginBaseline(
  setting: ConfigurationDefaultsSetting,
  parsed: { pluginName: string; path: string[] },
  nextValue: string | undefined,
  instance: RuntimeConfigPluginInstance | undefined
): boolean {
  return (
    instance?.enabled === false &&
    parsed.path.length === 1 &&
    parsed.path[0] === 'enabled' &&
    nextValue === getSettingBaselineValue(setting)
  )
}

function mergeConfigurationPluginSettingsIntoMeshConfig(
  config: RuntimeControlMeshConfig,
  values: ConfigurationDefaultsValues,
  schema?: RuntimeConfigSchemaReference,
  controlState?: RuntimeConfigControlStatePayload
): RuntimeControlDiagnostic[] {
  const integrations = createConfigurationIntegrationsFromSchema(schema, controlState)
  if (!integrations) return []
  const instances = schema ? pluginInstanceByName(schema) : new Map<string, RuntimeConfigPluginInstance>()
  const diagnostics: RuntimeControlDiagnostic[] = []

  for (const setting of integrations.settings) {
    const parsed = parsePluginCanonicalPath(setting)
    if (!parsed) continue

    const instance = instances.get(parsed.pluginName)
    const nextValue = values[setting.id]
    const currentPlugin = pluginEntryByName(config, parsed.pluginName)
    const currentValue = serializeDefaultSettingValue(setting, readPluginConfigPath(config, setting))
    const writeDisposition = settingWriteDisposition(setting, integrations.settings, values)

    if (writeDisposition === 'reject') {
      if (shouldPreserveRejectedDisabledWrite(currentValue, nextValue)) continue
      diagnostics.push(disabledWriteRejectedDiagnostic(setting, integrations.settings, values))
      continue
    }

    if (
      currentPlugin &&
      writeDisposition === 'omit' &&
      !shouldPreserveDisabledPluginBaseline(setting, parsed, nextValue, instance)
    ) {
      deletePath(currentPlugin, parsed.path)
    }

    if (writeDisposition !== 'write' || nextValue == null) continue

    const plugin = ensurePluginEntry(config, parsed.pluginName, instance)
    writePath(plugin, parsed.path, parseDefaultSettingValue(setting, nextValue))
  }

  if (Array.isArray(config.plugin)) {
    config.plugin = config.plugin.filter((entry) => {
      const keys = Object.keys(entry).filter((key) => key !== 'name')
      return keys.length > 0
    })
    if (config.plugin.length === 0) delete config.plugin
  }

  return diagnostics
}

function builtInSettingsFromSchema(
  schema?: RuntimeConfigSchemaReference,
  controlState?: RuntimeConfigControlStatePayload
): ConfigurationDefaultsSetting[] {
  const byId = new Map<string, ConfigurationDefaultsSetting>()
  for (const group of [
    createConfigurationMeshLLMSettingsFromSchema(schema, controlState),
    createConfigurationRuntimeSettingsFromSchema(schema, controlState),
    createConfigurationModelSettingsFromSchema(schema, controlState),
    createConfigurationNetworkSettingsFromSchema(schema, controlState),
    createConfigurationAttestationSettingsFromSchema(schema, controlState),
    createConfigurationAuditSettingsFromSchema(schema, controlState)
  ]) {
    for (const setting of group.settings) byId.set(setting.id, setting)
  }
  return Array.from(byId.values())
}

export function createConfigurationDefaultsValuesFromMeshConfig(
  config: RuntimeControlMeshConfig,
  schema?: RuntimeConfigSchemaReference,
  controlState?: RuntimeConfigControlStatePayload
): ConfigurationDefaultsValues {
  const values: ConfigurationDefaultsValues = {}

  for (const setting of builtInSettingsFromSchema(schema, controlState)) {
    const source = setting.canonicalPath?.startsWith('defaults.') ? config.defaults : config
    if (!source) continue
    const value = readPath(source, resolveConfigSettingPath(setting))
    const serialized = serializeDefaultSettingValue(setting, value)
    if (serialized !== undefined) values[setting.id] = serialized
  }

  const integrations = createConfigurationIntegrationsFromSchema(schema, controlState)
  if (integrations) {
    for (const setting of integrations.settings) {
      const value = readPluginConfigPath(config, setting)
      const serialized = serializeDefaultSettingValue(setting, value)
      if (serialized !== undefined) values[setting.id] = serialized
    }
  }

  return values
}

function settingWriteDisposition(
  setting: ConfigurationDefaultsSetting,
  settings: readonly ConfigurationDefaultsSetting[],
  values: ConfigurationDefaultsValues
) {
  const nextValue = values[setting.id]
  if (nextValue == null) return 'preserve' as const

  const evaluation = evaluateSettingControlState(setting, settings, values)
  if (!evaluation.enabled) {
    switch (evaluation.write_policy) {
      case 'preserve_existing':
        return 'preserve' as const
      case 'omit_when_disabled':
        return 'omit' as const
      case 'reject_when_disabled':
        return 'reject' as const
    }
  }

  return getSettingWriteDisposition(setting, settings, values)
}

function diagnosticPathForSetting(setting: ConfigurationDefaultsSetting): string {
  return setting.canonicalPath ?? resolveConfigSettingPath(setting).join('.')
}

function disabledWriteRejectedDiagnostic(
  setting: ConfigurationDefaultsSetting,
  settings: readonly ConfigurationDefaultsSetting[],
  values: ConfigurationDefaultsValues
): RuntimeControlDiagnostic {
  const canonicalPath = diagnosticPathForSetting(setting)
  const reason = getSettingDisabledReason(setting, settings, values) ?? 'This setting cannot be written while disabled.'

  return {
    code: 'disabled_write_rejected',
    severity: 'error',
    source: 'ui',
    path: canonicalPath,
    canonical_path: canonicalPath,
    message: `${canonicalPath}: ${reason}`,
    help: 'Remove the pending value or re-enable the setting before saving.'
  }
}

function shouldPreserveRejectedDisabledWrite(currentValue: string | undefined, nextValue: string | undefined): boolean {
  return nextValue == null || currentValue === nextValue
}

function currentBuiltInSettingValue(
  config: RuntimeControlMeshConfig,
  setting: ConfigurationDefaultsSetting
): string | undefined {
  const source = setting.canonicalPath?.startsWith('defaults.') ? config.defaults : config
  if (!source) return undefined
  return serializeDefaultSettingValue(setting, readPath(source, resolveConfigSettingPath(setting)))
}

function mergeBuiltInSettingsIntoMeshConfig(
  config: RuntimeControlMeshConfig,
  values: ConfigurationDefaultsValues,
  schema?: RuntimeConfigSchemaReference,
  controlState?: RuntimeConfigControlStatePayload
): RuntimeControlDiagnostic[] {
  const settings = builtInSettingsFromSchema(schema, controlState)
  const defaults =
    config.defaults && typeof config.defaults === 'object' && !Array.isArray(config.defaults)
      ? { ...config.defaults }
      : {}
  const diagnostics: RuntimeControlDiagnostic[] = []

  for (const setting of settings) {
    const path = resolveConfigSettingPath(setting)
    const target = setting.canonicalPath?.startsWith('defaults.') ? defaults : config
    const nextValue = values[setting.id]
    const currentValue = currentBuiltInSettingValue(config, setting)
    const writeDisposition = settingWriteDisposition(setting, settings, values)

    if (writeDisposition === 'reject') {
      if (shouldPreserveRejectedDisabledWrite(currentValue, nextValue)) continue
      diagnostics.push(disabledWriteRejectedDiagnostic(setting, settings, values))
      continue
    }

    if (writeDisposition === 'omit') {
      deletePath(target, path)
      continue
    }

    if (writeDisposition !== 'write') continue

    deletePath(target, path)
    writePath(target, path, parseDefaultSettingValue(setting, nextValue))
  }

  if (Object.keys(defaults).length === 0) {
    delete config.defaults
  } else {
    config.defaults = defaults
  }

  return diagnostics
}

function writeModelEntryPath(entry: RuntimeControlModelConfigEntry, path: string, value: unknown) {
  const segments = modelEntryPathSegments(path)
  if (segments.length === 0) return
  writePath(entry, segments, value)
}

function deleteModelEntryPath(entry: RuntimeControlModelConfigEntry, path: string) {
  const segments = modelEntryPathSegments(path)
  if (segments.length === 0) return
  deletePath(entry, segments)
}

function cloneModelEntry(entry: RuntimeControlModelConfigEntry): RuntimeControlModelConfigEntry {
  return JSON.parse(JSON.stringify(entry)) as RuntimeControlModelConfigEntry
}

function existingModelEntriesByName(
  config: RuntimeControlMeshConfig,
  placementPaths: ConfigurationModelPlacementPaths
): Map<string, RuntimeControlModelConfigEntry[]> {
  const entriesByName = new Map<string, RuntimeControlModelConfigEntry[]>()
  for (const entry of modelEntries(config)) {
    const modelName = modelNameFromEntry(entry, placementPaths)
    if (!modelName) continue

    const entries = entriesByName.get(modelName) ?? []
    entries.push(entry)
    entriesByName.set(modelName, entries)
  }
  return entriesByName
}

function consumeExistingModelEntry(
  entriesByName: Map<string, RuntimeControlModelConfigEntry[]>,
  modelName: string
): RuntimeControlModelConfigEntry {
  const entries = entriesByName.get(modelName)
  const existing = entries?.shift()
  return existing ? cloneModelEntry(existing) : {}
}

function isUnifiedMemoryConfigNode(node: ConfigNode | undefined): boolean {
  return Boolean(
    node &&
    (node.memoryTopology === 'unified' ||
      node.region.toLowerCase() === 'unified' ||
      node.gpus.some((gpu) => gpu.name.toLowerCase().includes('unified memory')))
  )
}

function mergeModelAssignmentsIntoMeshConfig(
  config: RuntimeControlMeshConfig,
  input: RuntimeControlApplyInput,
  schema?: RuntimeConfigSchemaReference
) {
  const localNode = input.nodes[0]
  if (!localNode) return

  const placementPaths = input.modelPlacementPaths ?? modelPlacementPathsFromSchema(schema)
  const emitsDevice = localNode.placement === 'separate' && !isUnifiedMemoryConfigNode(localNode)
  const localAssigns = input.assigns.filter((assign) => assign.nodeId === localNode.id)

  if (localAssigns.length === 0) {
    delete config.models
    return
  }

  const entriesByName = existingModelEntriesByName(config, placementPaths)

  config.models = localAssigns.map((assign) => {
    const model = input.catalog.find((item) => item.id === assign.modelId)
    const modelName = model?.name ?? assign.modelId
    const entry = consumeExistingModelEntry(entriesByName, modelName)
    writeModelEntryPath(entry, placementPaths.model, modelName)
    writeModelEntryPath(entry, placementPaths.ctxSize, assign.ctx)
    deleteModelEntryPath(entry, placementPaths.device)
    deleteModelEntryPath(entry, placementPaths.gpuLayers)
    if (emitsDevice) {
      writeModelEntryPath(entry, placementPaths.device, `cuda:${assign.containerIdx}`)
      writeModelEntryPath(entry, placementPaths.gpuLayers, -1)
    }
    writeSelectedModelConfig(entry, assign.config, placementPaths)
    return entry
  })
}

function writeOptionalModelEntryPath(
  entry: RuntimeControlModelConfigEntry,
  path: string,
  value: string | number | undefined
) {
  if (value === undefined || value === '') {
    deleteModelEntryPath(entry, path)
    return
  }
  writeModelEntryPath(entry, path, value)
}

function batchProfileValues(profile: ConfigAssignModelConfig['batchProfile']) {
  switch (profile) {
    case 'balanced':
      return { batch: 512, ubatch: 128 }
    case 'throughput':
      return { batch: 1024, ubatch: 256 }
    case 'saver':
      return { batch: 256, ubatch: 64 }
    default:
      return undefined
  }
}

function writeSelectedModelConfig(
  entry: RuntimeControlModelConfigEntry,
  config: ConfigAssignModelConfig | undefined,
  placementPaths: ConfigurationModelPlacementPaths
) {
  if (!config) return

  writeOptionalModelEntryPath(entry, 'models.<model-ref>.throughput.parallel', config?.slots)
  const batchProfile = batchProfileValues(config?.batchProfile)
  writeOptionalModelEntryPath(entry, 'models.<model-ref>.model_fit.batch', batchProfile?.batch)
  writeOptionalModelEntryPath(entry, 'models.<model-ref>.model_fit.ubatch', batchProfile?.ubatch)
  writeOptionalModelEntryPath(entry, 'models.<model-ref>.hardware.split_mode', config?.splitMode)
  writeOptionalModelEntryPath(entry, 'models.<model-ref>.hardware.tensor_split', config?.tensorSplit?.trim())
  writeOptionalModelEntryPath(
    entry,
    placementPaths.mmproj ?? DEFAULT_MODEL_PLACEMENT_PATHS.mmproj!,
    config?.mmproj?.trim()
  )
  writeOptionalModelEntryPath(entry, 'models.<model-ref>.speculative.draft_model', config?.draftModelPath?.trim())
  writeOptionalModelEntryPath(
    entry,
    placementPaths.flashAttention ?? DEFAULT_MODEL_PLACEMENT_PATHS.flashAttention!,
    config?.flashAttention
  )
  writeOptionalModelEntryPath(
    entry,
    placementPaths.cacheTypeK ?? DEFAULT_MODEL_PLACEMENT_PATHS.cacheTypeK!,
    config?.cacheTypeK
  )
  writeOptionalModelEntryPath(
    entry,
    placementPaths.cacheTypeV ?? DEFAULT_MODEL_PLACEMENT_PATHS.cacheTypeV!,
    config?.cacheTypeV
  )
  writeOptionalModelEntryPath(
    entry,
    placementPaths.kvCachePolicy ?? DEFAULT_MODEL_PLACEMENT_PATHS.kvCachePolicy!,
    config?.kvCachePolicy
  )
}

export function mergeConfigurationIntoMeshConfig(
  config: RuntimeControlMeshConfig,
  input: RuntimeControlApplyInput,
  schema?: RuntimeConfigSchemaReference,
  options: MergeConfigurationIntoMeshConfigOptions = {}
): RuntimeControlMeshConfig {
  const nextConfig = cloneMeshConfig(config)
  const diagnostics = [
    ...mergeBuiltInSettingsIntoMeshConfig(nextConfig, input.values, schema, options.controlState),
    ...mergeConfigurationPluginSettingsIntoMeshConfig(nextConfig, input.values, schema, options.controlState)
  ]
  if (diagnostics.length > 0) throw new RuntimeControlSaveBlockedError(diagnostics)
  if (options.includeModelAssignments) mergeModelAssignmentsIntoMeshConfig(nextConfig, input, schema)
  return nextConfig
}

export function mergeConfigurationDefaultsIntoMeshConfig(
  config: RuntimeControlMeshConfig,
  defaultsValues: ConfigurationDefaultsValues,
  schema?: RuntimeConfigSchemaReference,
  controlState?: RuntimeConfigControlStatePayload
): RuntimeControlMeshConfig {
  return mergeConfigurationIntoMeshConfig(
    config,
    { values: defaultsValues, nodes: [], assigns: [], catalog: [] },
    schema,
    { includeModelAssignments: false, controlState }
  )
}
