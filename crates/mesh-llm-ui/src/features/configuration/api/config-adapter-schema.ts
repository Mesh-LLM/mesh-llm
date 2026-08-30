import type {
  ConfigurationDefaultsCategory,
  ConfigurationDefaultsControl,
  ConfigurationDefaultsHarnessData,
  ConfigurationDefaultsSetting,
  ConfigurationIntegrationsHarnessData,
  ConfigurationRuntimeControlStateEntry,
  ConfigurationSettingsHarnessData
} from '@/features/app-tabs/types'
import { CONFIGURATION_HARNESS } from '@/features/app-tabs/data'
import { createSchemaControl } from '@/features/configuration/api/schema-control-factory'
import { createRuntimePolicySettingsFromSchema } from './runtime-settings'
import {
  DEFAULT_CATEGORY_ORDER,
  DEFAULT_SETTING_ORDER,
  sortCategories,
  sortSettings,
  titleCaseIdentifier
} from './schema-setting-order'
import type {
  RuntimeConfigControlStatePayload,
  RuntimeConfigPluginInstance,
  RuntimeConfigSchemaEntry,
  RuntimeConfigSchemaReference
} from './config-adapter-types'
import { combineSettingsHarnessData } from './config-adapter-schema-values'
import { rendererIdForEntry } from './config-adapter-schema-placement'

export {
  DEFAULT_MODEL_PLACEMENT_PATHS,
  modelPlacementOptionsFromSchema,
  modelPlacementPathsFromSchema
} from './config-adapter-schema-placement'
export {
  combineSettingsHarnessData,
  configurationDefaultsSchemaPathEntries,
  overlayDefaultsValues
} from './config-adapter-schema-values'

const CATEGORY_ICON_BY_ID: Record<string, ConfigurationDefaultsSetting['icon']> = {
  meshllm: 'cpu',
  network: 'server',
  attestation: 'shield',
  telemetry: 'gauge',
  'runtime-policy': 'cog',
  runtime: 'cpu',
  memory: 'memory',
  'speculative-decoding': 'brain',
  advanced: 'cog',
  'request-defaults': 'filter',
  'skippy-transport': 'binary',
  multimodal: 'image',
  topology: 'layers',
  'advanced-server': 'server',
  'logs-general': 'layers',
  'logs-retention': 'gauge',
  'logs-buffers': 'server',
  'logs-artifacts': 'folder',
  'logs-webhooks': 'zap',
  'logs-audit': 'shield'
}

const FALLBACK_DEFAULTS_CATEGORY: ConfigurationDefaultsCategory = {
  id: 'advanced',
  label: 'Advanced',
  summary: 'Schema-derived advanced settings.',
  help: 'Additional supported config settings from the exported schema'
}

type SchemaSettingContext = 'settings' | 'integrations'

const DEFAULTS_CATEGORY_FALLBACKS: Record<string, ConfigurationDefaultsCategory> = {
  meshllm: {
    id: 'meshllm',
    label: 'General',
    summary: 'Local node startup and observability settings',
    help: 'Settings owned by the local mesh-llm process',
    tomlSection: 'gpu',
    order: 10
  },
  telemetry: {
    id: 'telemetry',
    label: 'Telemetry',
    summary: 'Opt-in metrics export and queue settings',
    help: 'Telemetry settings written to the local config file',
    tomlSection: 'telemetry',
    order: 20
  },
  'logs-general': {
    id: 'logs-general',
    label: 'General',
    summary: 'Master enable, summary, and export controls',
    help: 'General request-log settings written to the local config file',
    tomlSection: 'logging',
    order: 10
  },
  'logs-retention': {
    id: 'logs-retention',
    label: 'Retention',
    summary: 'How long request logs are kept and when cleanup runs',
    help: 'Retention settings written to the local config file',
    tomlSection: 'logging',
    order: 20
  },
  'logs-buffers': {
    id: 'logs-buffers',
    label: 'Buffers & Replay',
    summary: 'In-memory event buffers and the replay window',
    help: 'Buffer settings written to the local config file',
    tomlSection: 'logging',
    order: 30
  },
  'runtime-policy': {
    id: 'runtime-policy',
    label: 'Runtime Policy',
    summary: 'Runtime reconciliation behavior',
    help: 'Runtime settings applied by the local process on startup',
    tomlSection: 'runtime',
    order: 10
  },
  network: {
    id: 'network',
    label: 'Network',
    summary: 'Owner-control listener and advertised endpoint settings',
    help: 'Network settings used by owner-control on startup',
    tomlSection: 'owner_control',
    order: 10
  },
  attestation: {
    id: 'attestation',
    label: 'Attestation',
    summary: 'Certified-build admission requirements',
    help: 'Creation-time mesh requirement settings',
    tomlSection: 'mesh_requirements',
    order: 10
  },
  'logs-artifacts': {
    id: 'logs-artifacts',
    label: 'Artifacts & Storage',
    summary: 'On-disk artifact capture and byte limits',
    help: 'Artifact settings written to the local config file',
    tomlSection: 'logging',
    order: 40
  },
  'logs-webhooks': {
    id: 'logs-webhooks',
    label: 'Webhooks',
    summary: 'Outbound webhook delivery of log events',
    help: 'Webhook settings written to the local config file',
    tomlSection: 'logging',
    order: 50
  },
  'logs-audit': {
    id: 'logs-audit',
    label: 'Security Audit',
    summary: 'Independent security event log with automatic redaction',
    help: 'Security audit settings written to the local config file',
    tomlSection: 'logging.audit',
    order: 60
  },
  runtime: {
    id: 'runtime',
    label: 'Runtime',
    summary: 'Load-time runtime behavior and concurrency defaults',
    help: 'Runtime defaults inherited by model placements',
    tomlSection: 'defaults.throughput',
    order: 10
  },
  memory: {
    id: 'memory',
    label: 'Memory',
    summary: 'VRAM accounting and KV cache policy',
    help: 'Memory defaults inherited by model placements',
    tomlSection: 'defaults.model_fit',
    order: 20
  },
  'speculative-decoding': {
    id: 'speculative-decoding',
    label: 'Speculative Decoding',
    summary: 'Speculative draft policy defaults',
    help: 'Speculative decoding defaults inherited by model placements',
    tomlSection: 'defaults.speculative',
    order: 30
  },
  'request-defaults': {
    id: 'request-defaults',
    label: 'Request Defaults',
    summary: 'Request-time sampling and reasoning defaults',
    help: 'Request defaults merged into compatible API requests',
    tomlSection: 'defaults.request_defaults',
    order: 40
  },
  'skippy-transport': {
    id: 'skippy-transport',
    label: 'Skippy Transport',
    summary: 'Stage transport, chunking, and lifecycle defaults',
    help: 'Skippy runtime defaults inherited by placements',
    tomlSection: 'defaults.skippy',
    order: 50
  },
  multimodal: {
    id: 'multimodal',
    label: 'Multimodal',
    summary: 'Vision projector and image token defaults',
    help: 'Multimodal defaults inherited by placements',
    tomlSection: 'defaults.multimodal',
    order: 60
  },
  topology: {
    id: 'topology',
    label: 'Topology',
    summary: 'Locked staged topology defaults.',
    help: 'Ordered layer ranges and node selectors for locked staged serving',
    tomlSection: 'defaults.topology',
    order: 70
  },
  'advanced-server': {
    id: 'advanced-server',
    label: 'Advanced Server',
    summary: 'Advanced server defaults and identity overrides',
    help: 'Advanced server defaults inherited by placements',
    tomlSection: 'defaults.advanced.server',
    order: 70
  }
}

type ChoicePresentation = Extract<ConfigurationDefaultsControl, { kind: 'choice' }>['presentation']

function settingIdFromPath(canonicalPath: string) {
  return canonicalPath
}

function lastPathSegment(canonicalPath: string) {
  return canonicalPath.split('.').filter(Boolean).at(-1) ?? canonicalPath
}

function defaultsSectionForPath(canonicalPath: string) {
  const segments = canonicalPath.split('.')
  if (segments[0] !== 'defaults') return undefined
  if (segments[1] === 'advanced' && segments[2] === 'server') return 'defaults.advanced.server'
  return segments.length >= 2 ? `defaults.${segments[1]}` : undefined
}

function configSectionForPath(canonicalPath: string) {
  if (canonicalPath.startsWith('plugin.')) return undefined
  const segments = canonicalPath.split('.').filter(Boolean)
  if (segments.length <= 1) return undefined
  if (segments[0] === 'defaults') return defaultsSectionForPath(canonicalPath)
  return segments.slice(0, -1).join('.')
}

function categoryForDefaultsPath(canonicalPath: string) {
  if (canonicalPath.startsWith('gpu.')) return 'runtime'
  if (canonicalPath.startsWith('logging.audit.')) return 'logs-audit'
  if (canonicalPath.startsWith('logging.webhook.')) return 'logs-webhooks'
  if (canonicalPath.startsWith('logging.artifact.')) return 'logs-artifacts'
  if (
    canonicalPath === 'logging.retention_ttl_secs' ||
    canonicalPath === 'logging.retention_max_rows' ||
    canonicalPath === 'logging.cleanup_cadence_secs'
  ) {
    return 'logs-retention'
  }
  if (
    canonicalPath === 'logging.queue_capacity' ||
    canonicalPath === 'logging.event_buffer_size' ||
    canonicalPath === 'logging.replay_capacity'
  ) {
    return 'logs-buffers'
  }
  if (canonicalPath.startsWith('logging.')) return 'logs-general'
  if (canonicalPath.startsWith('telemetry.')) return 'telemetry'
  if (canonicalPath === 'runtime.debug') return 'meshllm'
  if (canonicalPath === 'runtime.listen_all') return 'network'
  if (canonicalPath.startsWith('runtime.')) return 'runtime-policy'
  if (canonicalPath.startsWith('owner_control.')) return 'network'
  if (canonicalPath.startsWith('mesh_requirements.')) return 'attestation'
  if (canonicalPath === 'defaults.hardware.safety_margin_gb') return 'memory'
  if (canonicalPath.startsWith('defaults.model_fit.')) return 'memory'
  if (canonicalPath.startsWith('defaults.hardware.') || canonicalPath.startsWith('defaults.throughput.')) {
    return 'runtime'
  }
  if (canonicalPath.startsWith('defaults.speculative.')) return 'speculative-decoding'
  if (canonicalPath.startsWith('defaults.request_defaults.')) return 'request-defaults'
  if (canonicalPath.startsWith('defaults.skippy.')) return 'skippy-transport'
  if (canonicalPath.startsWith('defaults.multimodal.')) return 'multimodal'
  if (canonicalPath.startsWith('defaults.topology.')) return 'topology'
  if (canonicalPath.startsWith('defaults.advanced.server.')) return 'advanced-server'
  return 'advanced'
}

function controlNameForPath(canonicalPath: string) {
  return lastPathSegment(canonicalPath)
}

function segmentedControl(
  name: string,
  value: string,
  options: readonly string[],
  presentation: ChoicePresentation = 'segmented'
): ConfigurationDefaultsControl {
  return {
    kind: 'choice',
    name,
    value,
    presentation,
    options: options.map((option) => ({ value: option, label: option }))
  }
}

function bespokeControlForRenderer(entry: RuntimeConfigSchemaEntry): ConfigurationDefaultsControl | undefined {
  const rendererId = rendererIdForEntry(entry)
  const name = controlNameForPath(entry.canonical_path)

  if (rendererId === 'slot-meter') {
    return { kind: 'range', name, value: '4', min: 1, max: 16, step: 1, unit: entry.presentation?.unit ?? 'slots' }
  }

  if (rendererId === 'context-slider') {
    return {
      kind: 'range',
      name,
      value: '2048',
      min: 2048,
      max: 262144,
      step: 512,
      unit: entry.presentation?.unit ?? 'tokens'
    }
  }

  if (rendererId === 'kv-cache-policy') {
    return segmentedControl(name, 'auto', ['auto', 'quality', 'balanced', 'saver'])
  }

  return undefined
}

function fallbackControlForSchema(
  entry: RuntimeConfigSchemaEntry,
  controlState?: ConfigurationRuntimeControlStateEntry
): ConfigurationDefaultsControl {
  return createSchemaControl({
    entry,
    name: controlNameForPath(entry.canonical_path),
    bespoke: bespokeControlForRenderer(entry),
    runtimeControlState: controlState
  })
}

function isEditableSchemaEntry(entry: RuntimeConfigSchemaEntry) {
  return (
    entry.support === 'supported' &&
    entry.visibility !== 'hidden' &&
    entry.visibility !== 'internal' &&
    entry.control_surfaces.includes('config_file')
  )
}

function controlStateForPath(
  controlState: RuntimeConfigControlStatePayload | undefined,
  canonicalPath: string
): ConfigurationRuntimeControlStateEntry | undefined {
  return controlState?.settings?.[canonicalPath]
}

type SchemaSettingFromEntryInput = {
  entry: RuntimeConfigSchemaEntry
  context: SchemaSettingContext
  categoryId?: ConfigurationDefaultsCategory['id']
  controlState?: ConfigurationRuntimeControlStateEntry
}

function resolvedVisibilityForPath(
  canonicalPath: string,
  schemaVisibility: RuntimeConfigSchemaEntry['visibility']
): 'standard' | 'advanced' {
  if (canonicalPath.startsWith('logging.audit.')) return 'advanced'
  // Core logging controls remain visible without expanding advanced settings.
  if (canonicalPath.startsWith('logging.')) return 'standard'
  return schemaVisibility === 'advanced' ? 'advanced' : 'standard'
}

function schemaMutability(entry: RuntimeConfigSchemaEntry): ConfigurationDefaultsSetting['mutability'] {
  return entry.apply_mode === 'dynamic_apply' && entry.restart_scope === 'none' ? 'runtime' : 'restart-required'
}

const ENABLED_SETTING_ORDER = 10

function settingOrderForEntry(entry: RuntimeConfigSchemaEntry) {
  if (entry.presentation?.setting_order !== undefined) return entry.presentation.setting_order
  // Keep enablement toggles at the top of their category group.
  if (lastPathSegment(entry.canonical_path) === 'enabled') return ENABLED_SETTING_ORDER
  return DEFAULT_SETTING_ORDER
}

function categoryFromEntry(
  entry: RuntimeConfigSchemaEntry,
  context: SchemaSettingContext
): ConfigurationDefaultsCategory {
  const categoryId =
    entry.presentation?.category_id ??
    (context === 'settings'
      ? categoryForDefaultsPath(entry.canonical_path)
      : `plugin:${pluginNameFromSchemaEntry(entry)}`)
  const fallback =
    context === 'settings'
      ? (DEFAULTS_CATEGORY_FALLBACKS[categoryId] ?? FALLBACK_DEFAULTS_CATEGORY)
      : ({
          id: categoryId,
          label: titleCaseIdentifier(String(categoryId).replace(/^plugin:/, '')),
          summary: 'Plugin configuration settings',
          help: 'Settings exported by the installed plugin schema',
          order: entry.presentation?.category_order ?? DEFAULT_CATEGORY_ORDER
        } satisfies ConfigurationDefaultsCategory)

  return {
    ...fallback,
    id: categoryId,
    label: entry.presentation?.category_label ?? fallback.label,
    summary: entry.presentation?.category_summary ?? fallback.summary,
    help: entry.presentation?.category_summary ?? fallback.help,
    tomlSection: context === 'settings' ? configSectionForPath(entry.canonical_path) : fallback.tomlSection,
    order: entry.presentation?.category_order ?? fallback.order ?? DEFAULT_CATEGORY_ORDER
  }
}

function hasCategoryPresentation(entry: RuntimeConfigSchemaEntry) {
  const presentation = entry.presentation
  return Boolean(
    presentation?.category_label || presentation?.category_summary || presentation?.category_order !== undefined
  )
}

function schemaSettingFromEntry(input: SchemaSettingFromEntryInput): ConfigurationDefaultsSetting {
  const { entry, context, categoryId, controlState } = input
  const key = controlNameForPath(entry.canonical_path)
  const rendererId = rendererIdForEntry(entry)
  const category = categoryFromEntry(entry, context)
  const resolvedCategoryId = categoryId ?? category.id

  return {
    id: settingIdFromPath(entry.canonical_path),
    categoryId: resolvedCategoryId,
    canonicalPath: entry.canonical_path,
    tomlSection:
      context === 'settings'
        ? configSectionForPath(entry.canonical_path)
        : entry.canonical_path.includes('.settings.')
          ? `plugin.${pluginNameFromSchemaEntry(entry) ?? 'plugin'}.settings`
          : `plugin.${pluginNameFromSchemaEntry(entry) ?? 'plugin'}`,
    tomlKey: key,
    rendererId,
    controlHint: entry.presentation?.control_hint,
    displayUnits: entry.presentation?.display_units,
    settingOrder: settingOrderForEntry(entry),
    icon: CATEGORY_ICON_BY_ID[String(resolvedCategoryId)] ?? 'cog',
    label: entry.presentation?.label ?? titleCaseIdentifier(key),
    description:
      entry.presentation?.help ??
      (entry.description && entry.description !== entry.canonical_path ? entry.description : entry.canonical_path),
    inheritedLabel:
      context === 'settings' && entry.canonical_path.startsWith('defaults.')
        ? 'Inherited by placements that do not override this setting'
        : context === 'settings'
          ? 'Written to the local mesh-llm config file'
          : `Provided by ${pluginNameFromSchemaEntry(entry) ?? 'plugin'}`,
    valueSchema: entry.value_schema,
    control: fallbackControlForSchema(entry, controlState),
    controlBehavior: entry.control_behavior,
    controlState,
    visibility: resolvedVisibilityForPath(entry.canonical_path, entry.visibility),
    mutability: schemaMutability(entry),
    applyMode: entry.apply_mode,
    restartScope: entry.restart_scope,
    validationConstraints: entry.constraints,
    categoryOrder: category.order ?? DEFAULT_CATEGORY_ORDER
  }
}

export function createConfigurationDefaultsFromSchema(
  schema: RuntimeConfigSchemaReference | undefined,
  controlState?: RuntimeConfigControlStatePayload
): ConfigurationDefaultsHarnessData {
  if (!schema) return CONFIGURATION_HARNESS.defaults

  return createConfigurationSettingsFromSchema(
    schema,
    (entry) => entry.canonical_path.startsWith('defaults.'),
    'Generated defaults',
    controlState
  )
}

function createConfigurationSettingsFromSchema(
  schema: RuntimeConfigSchemaReference | undefined,
  includeEntry: (entry: RuntimeConfigSchemaEntry) => boolean,
  previewLabel: string,
  controlState?: RuntimeConfigControlStatePayload
): ConfigurationSettingsHarnessData {
  if (!schema) return { categories: [], settings: [], preview: [] }

  const settings = (schema?.settings ?? [])
    .filter((entry) => isEditableSchemaEntry(entry) && includeEntry(entry))
    .map((entry) =>
      schemaSettingFromEntry({
        entry,
        context: 'settings',
        controlState: controlStateForPath(controlState, entry.canonical_path)
      })
    )
  const categoryById = new Map<string, ConfigurationDefaultsCategory>()
  for (const entry of schema?.settings ?? []) {
    if (!isEditableSchemaEntry(entry) || !includeEntry(entry)) continue
    const category = categoryFromEntry(entry, 'settings')
    const categoryId = String(category.id)
    if (!categoryById.has(categoryId) || hasCategoryPresentation(entry)) categoryById.set(categoryId, category)
  }

  return {
    categories: sortCategories(Array.from(categoryById.values())),
    settings: sortSettings(settings),
    preview: [
      { label: previewLabel, value: `${settings.length} settings`, meta: 'schema' },
      { label: 'Source', value: '/api/runtime/config-schema', meta: 'live' }
    ]
  }
}

export function createConfigurationMeshLLMSettingsFromSchema(
  schema: RuntimeConfigSchemaReference | undefined,
  controlState?: RuntimeConfigControlStatePayload
): ConfigurationSettingsHarnessData {
  return createConfigurationSettingsFromSchema(
    schema,
    (entry) => entry.canonical_path.startsWith('telemetry.') || entry.canonical_path === 'runtime.debug',
    'Generated General settings',
    controlState
  )
}

export function createConfigurationRuntimeSettingsFromSchema(
  schema: RuntimeConfigSchemaReference | undefined,
  controlState?: RuntimeConfigControlStatePayload
): ConfigurationSettingsHarnessData {
  return combineSettingsHarnessData(
    createRuntimePolicySettingsFromSchema(schema, controlState),
    createConfigurationSettingsFromSchema(
      schema,
      (entry) =>
        entry.canonical_path.startsWith('defaults.throughput.') ||
        entry.canonical_path.startsWith('defaults.skippy.') ||
        entry.canonical_path.startsWith('defaults.advanced.server.'),
      'Generated runtime settings',
      controlState
    )
  )
}

export function createConfigurationModelSettingsFromSchema(
  schema: RuntimeConfigSchemaReference | undefined,
  controlState?: RuntimeConfigControlStatePayload
): ConfigurationSettingsHarnessData {
  return createConfigurationSettingsFromSchema(
    schema,
    (entry) =>
      entry.canonical_path.startsWith('gpu.') ||
      entry.canonical_path.startsWith('defaults.model_fit.') ||
      entry.canonical_path.startsWith('defaults.hardware.') ||
      entry.canonical_path.startsWith('defaults.topology.') ||
      entry.canonical_path.startsWith('defaults.speculative.') ||
      entry.canonical_path.startsWith('defaults.request_defaults.') ||
      entry.canonical_path.startsWith('defaults.multimodal.'),
    'Generated model settings',
    controlState
  )
}

export function createConfigurationNetworkSettingsFromSchema(
  schema: RuntimeConfigSchemaReference | undefined,
  controlState?: RuntimeConfigControlStatePayload
): ConfigurationSettingsHarnessData {
  return createConfigurationSettingsFromSchema(
    schema,
    (entry) => entry.canonical_path.startsWith('owner_control.') || entry.canonical_path === 'runtime.listen_all',
    'Generated network settings',
    controlState
  )
}

export function createConfigurationAttestationSettingsFromSchema(
  schema: RuntimeConfigSchemaReference | undefined,
  controlState?: RuntimeConfigControlStatePayload
): ConfigurationSettingsHarnessData {
  return createConfigurationSettingsFromSchema(
    schema,
    (entry) => entry.canonical_path.startsWith('mesh_requirements.'),
    'Generated attestation settings',
    controlState
  )
}

export function createConfigurationAuditSettingsFromSchema(
  schema: RuntimeConfigSchemaReference | undefined,
  controlState?: RuntimeConfigControlStatePayload
): ConfigurationSettingsHarnessData {
  const result = createConfigurationSettingsFromSchema(
    schema,
    (entry) => entry.canonical_path.startsWith('logging.'),
    'Generated logs settings',
    controlState
  )

  return result
}

function pluginNameFromSchemaEntry(entry: RuntimeConfigSchemaEntry) {
  if (entry.source.kind === 'plugin') return entry.source.plugin_name
  if (entry.presentation?.category_id?.startsWith('plugin:')) {
    return entry.presentation.category_id.slice('plugin:'.length)
  }
  const match = /^plugin\.([^.]+)\./.exec(entry.canonical_path)
  return match?.[1]
}

function pluginSettingKeyFromPath(canonicalPath: string, pluginName?: string) {
  const prefix = pluginName ? `plugin.${pluginName}.settings.` : undefined
  if (prefix && canonicalPath.startsWith(prefix)) return canonicalPath.slice(prefix.length)
  return canonicalPath.match(/^plugin\.[^.]+\.settings\.(.+)$/)?.[1] ?? lastPathSegment(canonicalPath)
}

function pluginTemplateEntries(schema: RuntimeConfigSchemaReference) {
  return schema.settings.filter(
    (entry) =>
      isEditableSchemaEntry(entry) &&
      entry.source.kind === 'built_in' &&
      entry.canonical_path.startsWith('plugin.<plugin-name>.') &&
      entry.canonical_path !== 'plugin.<plugin-name>.name'
  )
}

function pluginOwnedEntries(schema: RuntimeConfigSchemaReference) {
  return schema.settings.filter(
    (entry) =>
      isEditableSchemaEntry(entry) && entry.source.kind === 'plugin' && entry.canonical_path.includes('.settings.')
  )
}

export function pluginInstanceByName(schema: RuntimeConfigSchemaReference) {
  const instances = new Map((schema.plugin_instances ?? []).map((instance) => [instance.name, instance] as const))
  if (!instances.has('blobstore') && pluginTemplateEntries(schema).length > 0) {
    instances.set('blobstore', {
      name: 'blobstore',
      enabled: true,
      source_repository: 'built-in',
      installed_version: 'bundled',
      has_config_schema: false,
      allow_unvalidated_config: false
    })
  }
  return instances
}

function pluginNamesForIntegrations(schema: RuntimeConfigSchemaReference) {
  const names = new Set<string>()
  for (const instance of schema.plugin_instances ?? []) names.add(instance.name)
  for (const entry of pluginOwnedEntries(schema)) {
    const pluginName = pluginNameFromSchemaEntry(entry)
    if (pluginName) names.add(pluginName)
  }
  if (pluginTemplateEntries(schema).length > 0) names.add('blobstore')
  return Array.from(names).sort((left, right) => left.localeCompare(right))
}

function pluginCategory(
  pluginName: string,
  instance: RuntimeConfigPluginInstance | undefined,
  order: number
): ConfigurationDefaultsCategory {
  return {
    id: `plugin:${pluginName}`,
    label: titleCaseIdentifier(pluginName),
    summary: instance?.has_config_schema
      ? `Installed ${pluginName} plugin settings`
      : `Installed ${pluginName} plugin host settings`,
    help: instance?.source_repository ?? `${pluginName} plugin settings`,
    tomlSection: `plugin.${pluginName}`,
    order
  }
}

function instantiatePluginTemplateEntry(entry: RuntimeConfigSchemaEntry, pluginName: string): RuntimeConfigSchemaEntry {
  return {
    ...entry,
    canonical_path: entry.canonical_path.replace('plugin.<plugin-name>.', `plugin.${pluginName}.`),
    presentation: {
      ...entry.presentation,
      category_id: `plugin:${pluginName}`,
      category_label: titleCaseIdentifier(pluginName),
      category_summary: entry.presentation?.category_summary ?? `${pluginName} plugin host settings`
    }
  }
}

function settingWithPluginBaseline(
  setting: ConfigurationDefaultsSetting,
  instance: RuntimeConfigPluginInstance | undefined
): ConfigurationDefaultsSetting {
  if (!instance || !setting.canonicalPath?.endsWith('.enabled') || setting.control.kind !== 'choice') return setting
  return {
    ...setting,
    control: {
      ...setting.control,
      value: instance.enabled ? 'on' : 'off'
    }
  }
}

export function createConfigurationIntegrationsFromSchema(
  schema: RuntimeConfigSchemaReference | undefined,
  controlState?: RuntimeConfigControlStatePayload
): ConfigurationIntegrationsHarnessData | undefined {
  if (!schema) return undefined

  const pluginNames = pluginNamesForIntegrations(schema)
  if (pluginNames.length === 0) return undefined

  const instances = pluginInstanceByName(schema)
  const hostTemplates = pluginTemplateEntries(schema)
  const customSettings = pluginOwnedEntries(schema)
  const categories = pluginNames.map((pluginName, index) =>
    pluginCategory(pluginName, instances.get(pluginName), index)
  )
  const settings: ConfigurationDefaultsSetting[] = []

  for (const pluginName of pluginNames) {
    const categoryId = `plugin:${pluginName}`
    const instance = instances.get(pluginName)
    const templates =
      pluginName === 'blobstore'
        ? hostTemplates.filter((entry) => entry.canonical_path.endsWith('.enabled'))
        : hostTemplates
    for (const template of templates) {
      const entry = instantiatePluginTemplateEntry(template, pluginName)
      settings.push(
        settingWithPluginBaseline(
          schemaSettingFromEntry({
            entry,
            context: 'integrations',
            categoryId,
            controlState: controlStateForPath(controlState, entry.canonical_path)
          }),
          instance
        )
      )
    }
  }

  for (const entry of customSettings) {
    const pluginName = pluginNameFromSchemaEntry(entry) ?? 'plugin'
    const setting = schemaSettingFromEntry({
      entry,
      context: 'integrations',
      categoryId: `plugin:${pluginName}`,
      controlState: controlStateForPath(controlState, entry.canonical_path)
    })
    settings.push({
      ...setting,
      tomlSection: `plugin.${pluginName}.settings`,
      tomlKey: pluginSettingKeyFromPath(entry.canonical_path, pluginName)
    })
  }

  return { categories, settings: sortSettings(settings), preview: [] }
}
