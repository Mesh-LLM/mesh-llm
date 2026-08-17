import { CFG_CATALOG } from '@/features/app-tabs/data'
import { isUnifiedMemoryNode } from '@/features/configuration/lib/config-math'
import {
  evaluateSettingControlState,
  getSettingBaselineValue,
  getSettingValue
} from '@/features/configuration/lib/settings-utils'
import type {
  ConfigAssign,
  ConfigAssignModelConfig,
  ConfigModel,
  ConfigNode,
  ConfigurationDefaultsHarnessData,
  ConfigurationDefaultsSetting,
  ConfigurationSettingValueSchema,
  ConfigurationDefaultsValues,
  ConfigurationModelPlacementPaths,
  ConfigurationTomlSectionId
} from '@/features/app-tabs/types'

type BuildTomlOptions = {
  defaults?: ConfigurationDefaultsHarnessData
  defaultsValues?: ConfigurationDefaultsValues
  modelPlacementPaths?: ConfigurationModelPlacementPaths
  modelConfigEntries?: readonly Record<string, unknown>[]
}

type DefaultTomlPlacement = { sectionPath: string | null; key: string }
type SectionSettingLine = { setting: ConfigurationDefaultsSetting; key: string }

const DEFAULT_MODEL_PLACEMENT_PATHS: ConfigurationModelPlacementPaths = {
  model: 'models.<model-ref>.model',
  ctxSize: 'models.<model-ref>.model_fit.ctx_size',
  device: 'models.<model-ref>.hardware.device',
  gpuLayers: 'models.<model-ref>.hardware.gpu_layers',
  cacheTypeK: 'models.<model-ref>.model_fit.cache_type_k',
  cacheTypeV: 'models.<model-ref>.model_fit.cache_type_v',
  kvCachePolicy: 'models.<model-ref>.model_fit.kv_cache_policy'
}

const defaultSectionOrder: readonly ConfigurationTomlSectionId[] = [
  'gpu',
  'telemetry',
  'telemetry.metrics',
  'runtime',
  'owner_control',
  'mesh_requirements',
  'defaults',
  'defaults.model_fit',
  'defaults.hardware',
  'defaults.throughput',
  'defaults.skippy',
  'defaults.speculative',
  'defaults.request_defaults',
  'defaults.multimodal',
  'defaults.advanced.server'
]
function tomlString(value: string): string {
  return JSON.stringify(value)
}

function tomlInlineKey(key: string): string {
  return /^[A-Za-z0-9_-]+$/.test(key) ? key : tomlString(key)
}

function tomlScalar(value: string): string {
  return /^-?\d+(\.\d+)?$/.test(value) ? value : tomlString(value)
}

function tomlInlineValue(value: unknown): string {
  if (typeof value === 'string') return tomlString(value)
  if (typeof value === 'number' && Number.isFinite(value)) return String(value)
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (Array.isArray(value)) return `[${value.map(tomlInlineValue).join(', ')}]`
  if (value && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>)
    return `{ ${entries.map(([key, item]) => `${tomlInlineKey(key)} = ${tomlInlineValue(item)}`).join(', ')} }`
  }
  return tomlString(String(value ?? ''))
}

function arrayTomlScalar(value: string): string {
  const items = value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)
  return `[${items.map(tomlString).join(', ')}]`
}

function objectTomlScalar(value: string): string {
  try {
    const parsed = JSON.parse(value)
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) return tomlInlineValue(parsed)
  } catch {
    return tomlString(value)
  }
  return tomlString(value)
}

function isTelemetryHeadersSetting(setting: ConfigurationDefaultsSetting): boolean {
  return setting.canonicalPath === 'telemetry.headers'
}

function isEmptyObjectTextValue(value: string): boolean {
  const trimmed = value.trim()
  if (trimmed.length === 0) return true

  try {
    const parsed: unknown = JSON.parse(trimmed)
    return parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed) && Object.keys(parsed).length === 0
  } catch {
    return false
  }
}

export function shouldOmitDefaultSettingValue(setting: ConfigurationDefaultsSetting, value: string): boolean {
  if (setting.control.kind === 'text' && value.trim().length === 0) return true
  return isTelemetryHeadersSetting(setting) && isEmptyObjectTextValue(value)
}

function isBooleanToggleChoice(setting: ConfigurationDefaultsSetting): boolean {
  return (
    setting.control.kind === 'choice' &&
    setting.control.presentation === 'toggle' &&
    setting.control.options.length === 2 &&
    setting.control.options.every((option) => option.value === 'on' || option.value === 'off')
  )
}

const boolOrAutoSettingNames = new Set([
  'context_shift',
  'continuous_batching',
  'cpu_moe',
  'fit_context',
  'kv_offload',
  'kv_unified',
  'mmap',
  'mmproj_offload',
  'prompt_cache',
  'spec_default',
  'warmup'
])

function isBooleanChoiceValue(setting: ConfigurationDefaultsSetting, value: string): boolean {
  if (setting.control.kind !== 'choice') return false
  if (value !== 'on' && value !== 'off') return false
  const optionValues = new Set(setting.control.options.map((option) => option.value))
  if (!optionValues.has('on') || !optionValues.has('off')) return false
  if (hasSchemaKind(setting.valueSchema, 'boolean')) return true
  return 'name' in setting.control && boolOrAutoSettingNames.has(setting.control.name)
}

function hasSchemaKind(
  schema: ConfigurationSettingValueSchema | undefined,
  kind: ConfigurationSettingValueSchema['kind']
): boolean {
  if (!schema) return false
  if (schema.kind === kind) return true
  if (schema.kind === 'one_of') return schema.variants.some((variant) => hasSchemaKind(variant, kind))
  return false
}

function numericSchemaTomlScalar(setting: ConfigurationDefaultsSetting, value: string): string | undefined {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return undefined

  if (hasSchemaKind(setting.valueSchema, 'float')) return String(parsed)

  if (hasSchemaKind(setting.valueSchema, 'integer')) {
    return Number.isInteger(parsed) ? String(parsed) : undefined
  }

  return undefined
}

export function defaultSettingTomlScalar(setting: ConfigurationDefaultsSetting, value: string): string {
  if (isBooleanToggleChoice(setting)) return value === 'on' ? 'true' : 'false'
  if (isBooleanChoiceValue(setting, value)) return value === 'on' ? 'true' : 'false'
  if (
    setting.canonicalPath?.endsWith('.gpu_layers') ||
    ('name' in setting.control && setting.control.name === 'gpu_layers')
  ) {
    const numericValue = numericSchemaTomlScalar(setting, value)
    return numericValue ?? tomlScalar(value)
  }
  if (setting.control.kind === 'text' && setting.valueSchema?.kind === 'array') return arrayTomlScalar(value)
  if (setting.control.kind === 'text' && setting.valueSchema?.kind === 'object') return objectTomlScalar(value)
  if (setting.control.kind === 'text') {
    const numericValue = numericSchemaTomlScalar(setting, value)
    if (numericValue !== undefined) return numericValue
  }
  if (setting.control.kind === 'text') return tomlString(value)
  return tomlScalar(value)
}

const legacySectionPaths: Partial<Record<ConfigurationDefaultsSetting['categoryId'], string>> = {
  advanced: 'defaults.runtime',
  'speculative-decoding': 'defaults.speculative'
}

const defaultFlatAliases: ReadonlyMap<string, DefaultTomlPlacement> = new Map([
  ['defaults.model_fit.ctx_size', { sectionPath: 'defaults', key: 'ctx_size' }],
  ['defaults.model_fit.batch', { sectionPath: 'defaults', key: 'batch' }],
  ['defaults.model_fit.ubatch', { sectionPath: 'defaults', key: 'ubatch' }],
  ['defaults.model_fit.cache_type_k', { sectionPath: 'defaults', key: 'cache_type_k' }],
  ['defaults.model_fit.cache_type_v', { sectionPath: 'defaults', key: 'cache_type_v' }],
  ['defaults.model_fit.flash_attention', { sectionPath: 'defaults', key: 'flash_attention' }],
  ['defaults.hardware.device', { sectionPath: 'defaults', key: 'gpu_id' }],
  ['defaults.throughput.parallel', { sectionPath: 'defaults', key: 'parallel' }],
  ['defaults.multimodal.mmproj', { sectionPath: 'defaults', key: 'mmproj' }]
] satisfies readonly (readonly [string, DefaultTomlPlacement])[])

function defaultSettingTomlKey(setting: ConfigurationDefaultsSetting): string {
  return setting.tomlKey ?? (setting.control.kind === 'metric' ? setting.id : setting.control.name)
}

function defaultFlatAliasForSetting(
  setting: ConfigurationDefaultsSetting,
  sectionPath: string | null,
  key: string
): DefaultTomlPlacement | undefined {
  if (setting.canonicalPath) return defaultFlatAliases.get(setting.canonicalPath)
  if (!sectionPath) return undefined
  return defaultFlatAliases.get(`${sectionPath}.${key}`)
}

function resolveDefaultSettingSectionPath(
  setting: ConfigurationDefaultsSetting,
  categoryById: ReadonlyMap<string, ConfigurationDefaultsHarnessData['categories'][number]>
) {
  if (setting.canonicalPath?.startsWith('defaults.')) {
    const segments = setting.canonicalPath.split('.')
    if (segments[1] === 'advanced' && segments[2] === 'server') return 'defaults.advanced.server'
    if (segments.length > 2) return `defaults.${segments[1]}`
  }

  const explicitSection = setting.tomlSection ?? categoryById.get(setting.categoryId)?.tomlSection
  if (explicitSection) return explicitSection
  if (!setting.canonicalPath) return legacySectionPaths[setting.categoryId] ?? null
  return null
}

export function defaultSettingTomlPlacement(
  setting: ConfigurationDefaultsSetting,
  categoryById: ReadonlyMap<string, ConfigurationDefaultsHarnessData['categories'][number]>
): DefaultTomlPlacement {
  const key = defaultSettingTomlKey(setting)
  const sectionPath = resolveDefaultSettingSectionPath(setting, categoryById)
  return defaultFlatAliasForSetting(setting, sectionPath, key) ?? { sectionPath, key }
}

export function defaultSettingSectionPath(
  setting: ConfigurationDefaultsSetting,
  categoryById: ReadonlyMap<string, ConfigurationDefaultsHarnessData['categories'][number]>
) {
  return defaultSettingTomlPlacement(setting, categoryById).sectionPath
}

export function shouldOmitSettingFromGeneratedToml(
  setting: ConfigurationDefaultsSetting,
  allSettings: readonly ConfigurationDefaultsSetting[],
  values: ConfigurationDefaultsValues
) {
  const evaluation = evaluateSettingControlState(setting, allSettings, values)
  if (evaluation.enabled || evaluation.write_policy !== 'omit_when_disabled') return false

  const pendingValue = values[setting.id]
  if (pendingValue === undefined) return false
  return pendingValue !== setting.control.value
}

function tomlKey(value: string): string {
  return value.replaceAll('-', '_')
}

function pluginPathParts(setting: ConfigurationDefaultsSetting): { pluginName: string; path: string[] } | undefined {
  const canonicalPath = setting.canonicalPath
  const sectionSegments = setting.tomlSection?.split('.').filter(Boolean) ?? []
  if (canonicalPath?.startsWith('plugin.') && sectionSegments[0] === 'plugin') {
    const hasSettingsSection = sectionSegments.at(-1) === 'settings'
    const pluginName = sectionSegments.slice(1, hasSettingsSection ? -1 : undefined).join('.')
    const prefix = hasSettingsSection ? `plugin.${pluginName}.settings.` : `plugin.${pluginName}.`
    if (!pluginName || !canonicalPath.startsWith(prefix)) return undefined

    const suffix = canonicalPath.slice(prefix.length).split('.').filter(Boolean)
    return { pluginName, path: hasSettingsSection ? ['settings', ...suffix] : suffix }
  }

  const match = canonicalPath?.match(/^plugin\.([^.]+)\.(.+)$/)
  if (!match) return undefined
  return {
    pluginName: match[1],
    path: match[2].split('.').filter(Boolean)
  }
}

function pluginTomlKey(setting: ConfigurationDefaultsSetting, path: readonly string[]): string {
  if (setting.tomlKey) return setting.tomlKey
  return path.at(-1) ?? ('name' in setting.control ? setting.control.name : setting.id)
}

function modelPlacementSectionAndKey(path: string): { section?: string; key: string } {
  const suffix = path.replace(/^models\.(?:<model-ref>\.)?/, '')
  const segments = suffix.split('.').filter(Boolean)
  const key = segments.pop() ?? 'model'

  if (segments.length === 0) return { key }

  if (
    segments.join('.') === 'model_fit' &&
    ['ctx_size', 'cache_type_k', 'cache_type_v', 'batch', 'ubatch', 'flash_attention'].includes(key)
  ) {
    return { key }
  }
  if (segments.join('.') === 'throughput' && key === 'parallel') return { key }
  if (segments.join('.') === 'hardware' && key === 'device') return { key: 'gpu_id' }

  return {
    section: segments.length ? `models.${segments.join('.')}` : undefined,
    key
  }
}

function appendModelPlacementLine(
  modelLines: string[],
  sectionLines: Map<string, string[]>,
  section: string | undefined,
  line: string
) {
  if (!section) {
    modelLines.push(line)
    return
  }

  sectionLines.set(section, [...(sectionLines.get(section) ?? []), line])
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function modelEntryPathSegments(path: string): string[] {
  return path
    .replace(/^models\.<model-ref>\.?/, '')
    .split('.')
    .filter(Boolean)
}

function readModelEntryPath(entry: Record<string, unknown>, path: string): unknown {
  let current: unknown = entry
  for (const segment of modelEntryPathSegments(path)) {
    if (!isRecord(current)) return undefined
    current = current[segment]
  }
  return current
}

function configuredModelName(
  entry: Record<string, unknown>,
  placementPaths: ConfigurationModelPlacementPaths
): string | undefined {
  const configured = readModelEntryPath(entry, placementPaths.model)
  const value = typeof configured === 'string' ? configured : entry.model
  return typeof value === 'string' && value.trim() ? value : undefined
}

function modelEntryQueuesByName(
  entries: readonly Record<string, unknown>[],
  placementPaths: ConfigurationModelPlacementPaths
): Map<string, Record<string, unknown>[]> {
  const queues = new Map<string, Record<string, unknown>[]>()
  for (const entry of entries) {
    const modelName = configuredModelName(entry, placementPaths)
    if (!modelName) continue
    queues.set(modelName, [...(queues.get(modelName) ?? []), entry])
  }
  return queues
}

function modelLookupKeys(assign: ConfigAssign, model: ConfigModel | undefined): string[] {
  return Array.from(
    new Set([model?.name, model?.id, assign.modelId].filter((value): value is string => Boolean(value)))
  )
}

function consumeModelConfigEntry(
  queues: Map<string, Record<string, unknown>[]>,
  assign: ConfigAssign,
  model: ConfigModel | undefined
): Record<string, unknown> | undefined {
  for (const key of modelLookupKeys(assign, model)) {
    const queue = queues.get(key)
    if (queue?.length) return queue.shift()
  }
  return undefined
}

function modelFitOverridePaths(placementPaths: ConfigurationModelPlacementPaths): Array<{ key: string; path: string }> {
  return [
    { key: 'cache_type_k', path: placementPaths.cacheTypeK ?? DEFAULT_MODEL_PLACEMENT_PATHS.cacheTypeK! },
    { key: 'cache_type_v', path: placementPaths.cacheTypeV ?? DEFAULT_MODEL_PLACEMENT_PATHS.cacheTypeV! },
    { key: 'kv_cache_policy', path: placementPaths.kvCachePolicy ?? DEFAULT_MODEL_PLACEMENT_PATHS.kvCachePolicy! }
  ]
}

function modelThroughputOverridePaths(): Array<{ key: string; path: string }> {
  return [{ key: 'parallel', path: 'models.<model-ref>.throughput.parallel' }]
}

const BATCH_PROFILE_VALUES: Record<
  NonNullable<ConfigAssignModelConfig['batchProfile']>,
  { batch: number; ubatch: number } | undefined
> = {
  auto: undefined,
  balanced: { batch: 512, ubatch: 128 },
  throughput: { batch: 1024, ubatch: 256 },
  saver: { batch: 256, ubatch: 64 }
}
function readModelFitOverride(entry: Record<string, unknown>, path: string, key: string): unknown {
  const configured = readModelEntryPath(entry, path)
  if (configured !== undefined) return configured

  const modelFit = entry.model_fit as Record<string, unknown> | undefined
  if (isRecord(modelFit) && modelFit[key] !== undefined) return modelFit[key]
  return entry[key]
}

function readModelThroughputOverride(entry: Record<string, unknown>, path: string, key: string): unknown {
  const configured = readModelEntryPath(entry, path)
  if (configured !== undefined) return configured

  const throughput = entry.throughput as Record<string, unknown> | undefined
  if (isRecord(throughput) && throughput[key] !== undefined) return throughput[key]
  return entry[key]
}

function appendPreservedModelFitOverrides(
  entry: Record<string, unknown> | undefined,
  modelLines: string[],
  sectionLines: Map<string, string[]>,
  placementPaths: ConfigurationModelPlacementPaths,
  emittedKeys: Set<string>
) {
  if (!entry) return

  for (const { key: fallbackKey, path } of modelFitOverridePaths(placementPaths)) {
    const value = readModelFitOverride(entry, path, fallbackKey)
    if (value === undefined || value === null) continue

    const { section, key } = modelPlacementSectionAndKey(path)
    if (emittedKeys.has(key)) continue
    emittedKeys.add(key)
    appendModelPlacementLine(modelLines, sectionLines, section, `${tomlKey(key)} = ${tomlInlineValue(value)}`)
  }
}

function appendPreservedModelThroughputOverrides(
  entry: Record<string, unknown> | undefined,
  modelLines: string[],
  sectionLines: Map<string, string[]>,
  emittedKeys: Set<string>
) {
  if (!entry) return

  for (const { key: fallbackKey, path } of modelThroughputOverridePaths()) {
    const value = readModelThroughputOverride(entry, path, fallbackKey)
    if (value === undefined || value === null) continue

    const { section, key } = modelPlacementSectionAndKey(path)
    if (emittedKeys.has(key)) continue
    emittedKeys.add(key)
    appendModelPlacementLine(modelLines, sectionLines, section, `${tomlKey(key)} = ${tomlInlineValue(value)}`)
  }
}

function appendModelConfigLine(
  modelLines: string[],
  sectionLines: Map<string, string[]>,
  path: string,
  value: unknown
) {
  const { section, key } = modelPlacementSectionAndKey(path)
  appendModelPlacementLine(modelLines, sectionLines, section, `${tomlKey(key)} = ${tomlInlineValue(value)}`)
}

export function appendCanonicalModelConfigLine(
  sectionLines: Map<string, string[]>,
  sectionPath: string,
  key: string,
  value: unknown
) {
  const { section, key: normalizedKey } = modelPlacementSectionAndKey(`${sectionPath}.${key}`)
  appendModelPlacementLine([], sectionLines, section, `${tomlKey(normalizedKey)} = ${tomlInlineValue(value)}`)
}

export function appendSelectedModelConfig(
  config: ConfigAssignModelConfig | undefined,
  modelLines: string[],
  sectionLines: Map<string, string[]>,
  emittedKeys: Set<string>
) {
  if (!config) return

  if (config.slots !== undefined) {
    appendModelConfigLine(modelLines, sectionLines, 'models.<model-ref>.throughput.parallel', config.slots)
    emittedKeys.add('parallel')
  }
  const batchProfile = config.batchProfile ? BATCH_PROFILE_VALUES[config.batchProfile] : undefined
  if (batchProfile) {
    appendModelConfigLine(modelLines, sectionLines, 'models.<model-ref>.model_fit.batch', batchProfile.batch)
    emittedKeys.add('batch')
    appendModelConfigLine(modelLines, sectionLines, 'models.<model-ref>.model_fit.ubatch', batchProfile.ubatch)
    emittedKeys.add('ubatch')
  }
  if (config.splitMode) {
    appendModelConfigLine(modelLines, sectionLines, 'models.<model-ref>.hardware.split_mode', config.splitMode)
    emittedKeys.add('split_mode')
  }
  if (config.tensorSplit?.trim()) {
    appendModelConfigLine(
      modelLines,
      sectionLines,
      'models.<model-ref>.hardware.tensor_split',
      config.tensorSplit.trim()
    )
    emittedKeys.add('tensor_split')
  }
  if (config.mmproj?.trim()) {
    appendModelConfigLine(modelLines, sectionLines, 'models.<model-ref>.multimodal.mmproj', config.mmproj.trim())
    emittedKeys.add('mmproj')
  }
  if (config.draftModelPath?.trim()) {
    appendModelConfigLine(
      modelLines,
      sectionLines,
      'models.<model-ref>.speculative.draft_model',
      config.draftModelPath.trim()
    )
    emittedKeys.add('draft_model')
  }
  if (config.flashAttention) {
    appendModelConfigLine(
      modelLines,
      sectionLines,
      'models.<model-ref>.model_fit.flash_attention',
      config.flashAttention
    )
    emittedKeys.add('flash_attention')
  }
  if (config.cacheTypeK) {
    appendModelConfigLine(modelLines, sectionLines, 'models.<model-ref>.model_fit.cache_type_k', config.cacheTypeK)
    emittedKeys.add('cache_type_k')
  }
  if (config.cacheTypeV) {
    appendModelConfigLine(modelLines, sectionLines, 'models.<model-ref>.model_fit.cache_type_v', config.cacheTypeV)
    emittedKeys.add('cache_type_v')
  }
  if (config.kvCachePolicy) {
    appendModelConfigLine(
      modelLines,
      sectionLines,
      'models.<model-ref>.model_fit.kv_cache_policy',
      config.kvCachePolicy
    )
    emittedKeys.add('kv_cache_policy')
  }
}

export function buildTOML(
  nodes: ConfigNode[],
  assigns: ConfigAssign[],
  models: ConfigModel[] = CFG_CATALOG,
  options: BuildTomlOptions = {}
): string {
  const localNode = nodes[0]
  const lines: string[] = [
    '# Mesh LLM generated config preview',
    '# Remote nodes are read-only context and are not serialized from this page.',
    'version = 1',
    ''
  ]

  if (options.defaults) {
    const sectionSettings = new Map<string, SectionSettingLine[]>()
    const pluginSettings = new Map<
      string,
      { host: ConfigurationDefaultsSetting[]; custom: ConfigurationDefaultsSetting[] }
    >()
    const defaultsValues = options.defaultsValues ?? {}
    const categoryById = new Map(options.defaults.categories.map((category) => [category.id, category] as const))

    for (const setting of options.defaults.settings) {
      if (shouldOmitSettingFromGeneratedToml(setting, options.defaults.settings, defaultsValues)) continue

      const value = getSettingValue(setting, defaultsValues)
      if (value === getSettingBaselineValue(setting)) continue
      if (shouldOmitDefaultSettingValue(setting, value)) continue

      const pluginPath = pluginPathParts(setting)
      if (pluginPath) {
        const group = pluginSettings.get(pluginPath.pluginName) ?? { host: [], custom: [] }
        if (pluginPath.path[0] === 'settings') group.custom.push(setting)
        else if (pluginPath.path[0] !== 'name') group.host.push(setting)
        pluginSettings.set(pluginPath.pluginName, group)
        continue
      }

      const placement = defaultSettingTomlPlacement(setting, categoryById)
      const sectionPath = placement.sectionPath
      if (sectionPath) {
        sectionSettings.set(sectionPath, [...(sectionSettings.get(sectionPath) ?? []), { setting, key: placement.key }])
        continue
      }

      lines.push(`${tomlKey(placement.key)} = ${defaultSettingTomlScalar(setting, value)}`)
    }

    const orderedSectionPaths = [
      ...defaultSectionOrder.filter((sectionPath) => sectionSettings.has(sectionPath)),
      ...Array.from(sectionSettings.keys()).filter(
        (sectionPath) => !defaultSectionOrder.includes(sectionPath as ConfigurationTomlSectionId)
      )
    ]

    let emittedDefaultsSectionCount = 0
    for (const sectionPath of orderedSectionPaths) {
      const settings = sectionSettings.get(sectionPath)
      if (!settings) continue

      if (lines[lines.length - 1] !== '') lines.push('')
      if (emittedDefaultsSectionCount > 0) lines.push('')
      lines.push(`[${sectionPath}]`)
      for (const { setting, key } of settings) {
        const value = getSettingValue(setting, defaultsValues)
        lines.push(`${tomlKey(key)} = ${defaultSettingTomlScalar(setting, value)}`)
      }
      emittedDefaultsSectionCount += 1
    }

    for (const [pluginName, settings] of pluginSettings) {
      if (lines[lines.length - 1] !== '') lines.push('')
      lines.push('[[plugin]]', `name = ${tomlString(pluginName)}`)
      for (const setting of settings.host) {
        const pluginPath = pluginPathParts(setting)
        if (!pluginPath) continue
        const value = getSettingValue(setting, defaultsValues)
        lines.push(
          `${tomlInlineKey(pluginTomlKey(setting, pluginPath.path))} = ${defaultSettingTomlScalar(setting, value)}`
        )
      }
      if (settings.custom.length > 0) {
        lines.push('', '[plugin.settings]')
        for (const setting of settings.custom) {
          const pluginPath = pluginPathParts(setting)
          if (!pluginPath) continue
          const value = getSettingValue(setting, defaultsValues)
          lines.push(
            `${tomlInlineKey(pluginTomlKey(setting, pluginPath.path.slice(1)))} = ${defaultSettingTomlScalar(setting, value)}`
          )
        }
      }
    }
    lines.push('')
  }

  if (!localNode) return lines.join('\n').trimEnd()

  const emitsPerGpuDevice = localNode.placement === 'separate' && !isUnifiedMemoryNode(localNode)
  const modelPlacementPaths = options.modelPlacementPaths ?? DEFAULT_MODEL_PLACEMENT_PATHS
  const modelPath = modelPlacementSectionAndKey(modelPlacementPaths.model)
  const ctxPath = modelPlacementSectionAndKey(modelPlacementPaths.ctxSize)
  const devicePath = modelPlacementSectionAndKey(modelPlacementPaths.device)
  const gpuLayersPath = modelPlacementSectionAndKey(modelPlacementPaths.gpuLayers)
  const modelConfigEntryQueues = modelEntryQueuesByName(options.modelConfigEntries ?? [], modelPlacementPaths)
  for (const assign of assigns.filter((item) => item.nodeId === localNode.id)) {
    const model = models.find((catalogModel) => catalogModel.id === assign.modelId)
    const modelConfigEntry = consumeModelConfigEntry(modelConfigEntryQueues, assign, model)
    const sectionLines = new Map<string, string[]>()
    const modelLines: string[] = []

    appendModelPlacementLine(modelLines, sectionLines, ctxPath.section, `${tomlKey(ctxPath.key)} = ${assign.ctx}`)
    const emittedKeys = new Set<string>()
    appendSelectedModelConfig(assign.config, modelLines, sectionLines, emittedKeys)
    appendPreservedModelFitOverrides(modelConfigEntry, modelLines, sectionLines, modelPlacementPaths, emittedKeys)
    appendPreservedModelThroughputOverrides(modelConfigEntry, modelLines, sectionLines, emittedKeys)
    if (emitsPerGpuDevice) {
      appendModelPlacementLine(
        modelLines,
        sectionLines,
        devicePath.section,
        `${tomlKey(devicePath.key)} = ${tomlString(`cuda:${assign.containerIdx}`)}`
      )
      appendModelPlacementLine(modelLines, sectionLines, gpuLayersPath.section, `${tomlKey(gpuLayersPath.key)} = -1`)
    }

    if (lines[lines.length - 1] !== '') lines.push('')
    lines.push('[[models]]')
    appendModelPlacementLine(
      modelLines,
      sectionLines,
      modelPath.section,
      `${tomlKey(modelPath.key)} = ${tomlString(model?.name ?? assign.modelId)}`
    )
    lines.push(...modelLines)

    for (const sectionName of sectionLines.keys()) {
      const values = sectionLines.get(sectionName)
      if (!values?.length) continue
      lines.push('', `[${sectionName}]`, ...values)
    }
  }

  return lines.join('\n').trimEnd()
}
