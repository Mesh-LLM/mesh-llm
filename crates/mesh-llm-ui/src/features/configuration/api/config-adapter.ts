import type { StatusPayload, MeshModelRaw, PeerInfo, GpuInfo } from '@/lib/api/types'
import type {
  ConfigurationDefaultsCategory,
  ConfigurationDefaultsHarnessData,
  ConfigurationDefaultsControl,
  ConfigurationDefaultsSetting,
  ConfigurationDefaultsValues,
  ConfigurationHarnessData,
  ConfigurationIntegrationsHarnessData,
  ConfigurationModelPlacementOptions,
  ConfigurationModelPlacementPaths,
  ConfigurationRuntimeControlStateEntry,
  ConfigurationSettingsHarnessData,
  ConfigurationSettingControlBehavior,
  ConfigurationSettingValueSchema,
  ConfigAssign,
  ConfigAssignModelConfig,
  ConfigNode,
  ConfigModel
} from '@/features/app-tabs/types'
import { CONFIGURATION_HARNESS } from '@/features/app-tabs/data'
import { createSchemaControl } from '@/features/configuration/api/schema-control-factory'
import {
  evaluateSettingControlState,
  getSettingBaselineValue,
  getSettingDisabledReason,
  getSettingWriteDisposition
} from '@/features/configuration/lib/settings-utils'
import { ApiError, parseApiErrorBody } from '@/lib/api/errors'
import { env } from '@/lib/env'
import { gpuAllocatableVramGB, gpuRatedVramGB, gpuReservedVramGB, gpuSystemReportedVramGB } from '@/lib/vram'

export type RuntimeControlBootstrapPayload = {
  enabled: boolean
  local_only: boolean
  requires_explicit_remote_endpoint: boolean
  endpoint?: string
  disabled_reason?: string
  message?: string
  suggested_commands?: string[]
}

export type RuntimeControlDefaultsConfig = Record<string, unknown>

export type RuntimeControlMeshConfig = {
  defaults?: RuntimeControlDefaultsConfig
  models?: RuntimeControlModelConfigEntry[]
  plugin?: RuntimeControlPluginConfigEntry[]
  [key: string]: unknown
}

export type RuntimeControlModelConfigEntry = {
  model?: string
  ctx_size?: number
  model_fit?: Record<string, unknown>
  hardware?: Record<string, unknown>
  [key: string]: unknown
}

export type RuntimeControlPluginConfigEntry = {
  name?: string
  enabled?: boolean
  command?: string
  args?: unknown[]
  url?: string
  settings?: Record<string, unknown>
  startup?: Record<string, unknown>
  [key: string]: unknown
}

export type RuntimeConfigSchemaReference = {
  settings: RuntimeConfigSchemaEntry[]
  plugin_instances?: RuntimeConfigPluginInstance[]
}

export type RuntimeConfigControlStatePayload = {
  settings?: Record<string, ConfigurationRuntimeControlStateEntry>
}

export type RuntimeConfigPluginInstance = {
  name: string
  enabled: boolean
  source_repository: string
  installed_version: string
  last_status?: string
  last_error?: string
  has_config_schema: boolean
  allow_unvalidated_config: boolean
}

export type RuntimeConfigSchemaEntry = {
  canonical_path: string
  owner: 'built_in' | 'engine' | 'plugin'
  source: RuntimeConfigSchemaSource
  value_schema: ConfigurationSettingValueSchema
  support: 'supported' | 'experimental' | 'deprecated_alias' | 'unwired' | 'unsupported' | 'rejected'
  control_surfaces: string[]
  apply_mode: 'static_on_load' | 'dynamic_validation_only' | 'dynamic_apply'
  restart_scope: 'none' | 'model_reload' | 'process_restart' | 'mesh_restart'
  visibility: 'user' | 'advanced' | 'hidden' | 'internal'
  constraints?: RuntimeConfigConstraint[]
  description?: string
  presentation?: RuntimeConfigPresentation
  control_behavior?: ConfigurationSettingControlBehavior
}

export type RuntimeConfigPresentation = {
  label?: string
  help?: string
  category_id?: string
  category_label?: string
  category_summary?: string
  category_order?: number
  setting_order?: number
  unit?: string
  placeholder?: string
  control_hint?: string
  renderer_id?: string
}

export type RuntimeConfigSchemaSource =
  | { kind: 'built_in' }
  | { kind: 'engine'; engine_id: string }
  | { kind: 'plugin'; plugin_name: string; allow_unvalidated_config: boolean }

type RuntimeConfigConstraint =
  | { kind: 'non_empty' }
  | { kind: 'positive' }
  | { kind: 'range'; min?: string; max?: string }
  | { kind: 'requires'; path: unknown }
  | { kind: 'allowed_values'; values: string[] }
  | { kind: 'allowed_pattern'; pattern: string }

export type RuntimeControlConfigSnapshot = {
  revision: number
  config: RuntimeControlMeshConfig
  [key: string]: unknown
}

type RuntimeControlConfigResponse = {
  snapshot: RuntimeControlConfigSnapshot
}

export type RuntimeControlConfigResult = {
  bootstrap: RuntimeControlBootstrapPayload
  snapshot?: RuntimeControlConfigSnapshot
  schema?: RuntimeConfigSchemaReference
  controlState: RuntimeConfigControlStatePayload
}

export type RuntimeControlApplyResponse = {
  success: boolean
  current_revision: number
  config_hash: string
  apply_mode: string
  error?: unknown
  diagnostics?: RuntimeControlDiagnostic[]
}

export type RuntimeControlDiagnostic = {
  code: string
  severity: string
  source: string
  schema_source?: string
  path?: string
  canonical_path?: string
  message: string
  help?: string
}

class RuntimeControlSaveBlockedError extends Error {
  readonly diagnostics: readonly RuntimeControlDiagnostic[]

  constructor(diagnostics: readonly RuntimeControlDiagnostic[]) {
    super(diagnostics[0]?.message ?? 'Configuration save was blocked.')
    this.name = 'RuntimeControlSaveBlockedError'
    this.diagnostics = diagnostics
  }
}

/**
 * Format a severity level as a readable uppercase badge.
 */
function formatSeverity(severity: string): string {
  return `\`${severity.toUpperCase()}\``
}

/**
 * Format a single diagnostic as a pretty markdown block.
 *
 * Produces output like:
 *
 *   **`path.to.setting`** · `ERROR`
 *
 *   The validation message explaining what is wrong.
 *
 *   > **Help:** guidance on how to fix the issue
 */
function formatDiagnosticBlock(diagnostic: RuntimeControlDiagnostic): string {
  const lines: string[] = []

  // Header: path + severity
  const headerParts: string[] = []
  if (diagnostic.path) headerParts.push(`**\`${diagnostic.path}\`**`)
  headerParts.push(formatSeverity(diagnostic.severity))
  lines.push(headerParts.join(' · '))
  lines.push('')

  // Body: message
  lines.push(diagnostic.message)

  // Footer: help text
  if (diagnostic.help) {
    lines.push('')
    lines.push(`> **Help:** ${diagnostic.help}`)
  }

  return lines.join('\n')
}

/**
 * Format an array of runtime-control diagnostics as pretty-printed markdown.
 *
 * Each diagnostic is rendered as a block with path, severity, message, and
 * optional help text. Multiple diagnostics are separated by horizontal rules.
 *
 * Returns `undefined` when the array is empty.
 */
export function formatConfigDiagnostics(diagnostics: readonly RuntimeControlDiagnostic[]): string | undefined {
  if (diagnostics.length === 0) return undefined
  return diagnostics.map(formatDiagnosticBlock).join('\n\n---\n\n')
}

export type RuntimeConfigValidateResponse = {
  ok: boolean
  path?: string
  error?: string
  diagnostics: RuntimeControlDiagnostic[]
}

export type RuntimeControlApplyInput = {
  values: ConfigurationDefaultsValues
  nodes: ConfigNode[]
  assigns: ConfigAssign[]
  catalog: ConfigModel[]
  modelPlacementPaths?: ConfigurationModelPlacementPaths
}

type MergeConfigurationIntoMeshConfigOptions = {
  includeModelAssignments?: boolean
  controlState?: RuntimeConfigControlStatePayload
}

export type ConfigurationDefaultsSchemaPathEntry = {
  id: string
  canonicalPath: string
}

export function runtimeControlApplyErrorMessage(response: RuntimeControlApplyResponse | null | undefined) {
  if (!response) return undefined
  const errorText = runtimeControlErrorMessage(response.error)
  if (errorText) return errorText

  const errorDiagnostics = response.diagnostics?.filter((d) => d.severity === 'error')
  if (errorDiagnostics && errorDiagnostics.length > 0) {
    return formatConfigDiagnostics(errorDiagnostics) ?? errorDiagnostics[0].message
  }

  return response.diagnostics?.find((d) => d.severity === 'warning')?.message ?? undefined
}

function runtimeControlErrorMessage(error: unknown): string | undefined {
  if (typeof error === 'string') return error.trim() || undefined
  if (!error || typeof error !== 'object') return undefined

  const details = error as Record<string, unknown>
  if (typeof details['message'] === 'string') return details['message'].trim() || undefined
  if (typeof details['code'] === 'string') return details['code'].replace(/_/g, ' ')
  return undefined
}

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
  'advanced-server': 'server'
}

const FALLBACK_DEFAULTS_CATEGORY: ConfigurationDefaultsCategory = {
  id: 'advanced',
  label: 'Advanced',
  summary: 'Schema-derived advanced settings.',
  help: 'Additional supported config settings from the exported schema'
}

type SchemaSettingContext = 'settings' | 'integrations'

const DEFAULT_CATEGORY_ORDER = 1000
const DEFAULT_SETTING_ORDER = 1000

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
  'advanced-server': {
    id: 'advanced-server',
    label: 'Advanced Server',
    summary: 'Advanced server defaults and identity overrides',
    help: 'Advanced server defaults inherited by placements',
    tomlSection: 'defaults.advanced.server',
    order: 70
  }
}

const PATH_RENDERER_FALLBACKS: Record<string, string> = {
  'defaults.throughput.parallel': 'slot-meter',
  'defaults.model_fit.kv_cache_policy': 'kv-cache-policy',
  'defaults.model_fit.ctx_size': 'context-slider'
}

type ChoicePresentation = Extract<ConfigurationDefaultsControl, { kind: 'choice' }>['presentation']

function titleCaseIdentifier(value: string) {
  return value
    .replaceAll('_', ' ')
    .replaceAll('-', ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (match) => match.toUpperCase())
}

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
  if (canonicalPath.startsWith('defaults.advanced.server.')) return 'advanced-server'
  return 'advanced'
}

function controlNameForPath(canonicalPath: string) {
  return lastPathSegment(canonicalPath)
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

function rendererIdForEntry(entry: RuntimeConfigSchemaEntry): string | undefined {
  return entry.presentation?.renderer_id ?? PATH_RENDERER_FALLBACKS[entry.canonical_path]
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

function schemaMutability(entry: RuntimeConfigSchemaEntry): ConfigurationDefaultsSetting['mutability'] {
  return entry.apply_mode === 'dynamic_apply' && entry.restart_scope === 'none' ? 'runtime' : 'restart-required'
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
    settingOrder: entry.presentation?.setting_order ?? DEFAULT_SETTING_ORDER,
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
    visibility: entry.visibility === 'advanced' ? 'advanced' : 'standard',
    mutability: schemaMutability(entry),
    validationConstraints: entry.constraints,
    categoryOrder: category.order ?? DEFAULT_CATEGORY_ORDER
  }
}

function sortCategories(categories: readonly ConfigurationDefaultsCategory[]) {
  return [...categories].sort(
    (left, right) =>
      (left.order ?? DEFAULT_CATEGORY_ORDER) - (right.order ?? DEFAULT_CATEGORY_ORDER) ||
      left.label.localeCompare(right.label)
  )
}

function sortSettings(settings: readonly ConfigurationDefaultsSetting[]) {
  return [...settings].sort(
    (left, right) =>
      (left.categoryOrder ?? DEFAULT_CATEGORY_ORDER) - (right.categoryOrder ?? DEFAULT_CATEGORY_ORDER) ||
      (left.settingOrder ?? DEFAULT_SETTING_ORDER) - (right.settingOrder ?? DEFAULT_SETTING_ORDER) ||
      left.label.localeCompare(right.label) ||
      left.id.localeCompare(right.id)
  )
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
    categoryById.set(String(category.id), category)
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
  return createConfigurationSettingsFromSchema(
    schema,
    (entry) =>
      (entry.canonical_path.startsWith('runtime.') &&
        entry.canonical_path !== 'runtime.debug' &&
        entry.canonical_path !== 'runtime.listen_all') ||
      entry.canonical_path.startsWith('defaults.throughput.') ||
      entry.canonical_path.startsWith('defaults.skippy.') ||
      entry.canonical_path.startsWith('defaults.advanced.server.'),
    'Generated runtime settings',
    controlState
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

function pluginInstanceByName(schema: RuntimeConfigSchemaReference) {
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

const DEFAULT_MODEL_PLACEMENT_PATHS: ConfigurationModelPlacementPaths = {
  model: 'models.<model-ref>.model',
  ctxSize: 'models.<model-ref>.model_fit.ctx_size',
  device: 'models.<model-ref>.hardware.device',
  gpuLayers: 'models.<model-ref>.hardware.gpu_layers',
  cacheTypeK: 'models.<model-ref>.model_fit.cache_type_k',
  cacheTypeV: 'models.<model-ref>.model_fit.cache_type_v',
  kvCachePolicy: 'models.<model-ref>.model_fit.kv_cache_policy',
  flashAttention: 'models.<model-ref>.model_fit.flash_attention',
  mmproj: 'models.<model-ref>.multimodal.mmproj'
}

function modelPlacementPathsFromSchema(
  schema: RuntimeConfigSchemaReference | undefined
): ConfigurationModelPlacementPaths {
  if (!schema) return DEFAULT_MODEL_PLACEMENT_PATHS

  const pathByRenderer = new Map(
    schema.settings
      .filter((entry) => entry.canonical_path.startsWith('models.<model-ref>.'))
      .map((entry) => [rendererIdForEntry(entry), entry.canonical_path] as const)
      .filter((entry): entry is readonly [string, string] => Boolean(entry[0]))
  )

  return {
    model: pathByRenderer.get('model-placement-model') ?? DEFAULT_MODEL_PLACEMENT_PATHS.model,
    ctxSize: pathByRenderer.get('model-placement-context') ?? DEFAULT_MODEL_PLACEMENT_PATHS.ctxSize,
    device: pathByRenderer.get('model-placement-device') ?? DEFAULT_MODEL_PLACEMENT_PATHS.device,
    gpuLayers: pathByRenderer.get('model-placement-gpu-layers') ?? DEFAULT_MODEL_PLACEMENT_PATHS.gpuLayers,
    cacheTypeK: DEFAULT_MODEL_PLACEMENT_PATHS.cacheTypeK,
    cacheTypeV: DEFAULT_MODEL_PLACEMENT_PATHS.cacheTypeV,
    kvCachePolicy: DEFAULT_MODEL_PLACEMENT_PATHS.kvCachePolicy,
    flashAttention: DEFAULT_MODEL_PLACEMENT_PATHS.flashAttention,
    mmproj: DEFAULT_MODEL_PLACEMENT_PATHS.mmproj
  }
}

function schemaEnumValuesForPath(schema: RuntimeConfigSchemaReference | undefined, canonicalPath: string): string[] {
  const setting = schema?.settings.find((entry) => entry.canonical_path === canonicalPath)
  if (setting?.value_schema.kind === 'enum') return [...setting.value_schema.values]
  return []
}

function modelPlacementOptionsFromSchema(
  schema: RuntimeConfigSchemaReference | undefined
): ConfigurationModelPlacementOptions {
  return {
    cacheTypeK: schemaEnumValuesForPath(schema, DEFAULT_MODEL_PLACEMENT_PATHS.cacheTypeK!),
    cacheTypeV: schemaEnumValuesForPath(schema, DEFAULT_MODEL_PLACEMENT_PATHS.cacheTypeV!)
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

function mapNodeState(state: string | undefined): 'online' | 'degraded' | 'offline' {
  if (state === 'client') return 'offline'
  return 'online'
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

function modelEntries(config: RuntimeControlMeshConfig | undefined): RuntimeControlModelConfigEntry[] {
  return Array.isArray(config?.models) ? config.models : []
}

function modelNameFromEntry(
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
  config.draftModelPath = stringModelEntryValue(entry, 'models.<model-ref>.speculative.draft_model_path')

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
    .filter((model): model is ConfigModel => model !== undefined && !existingIds.has(model.id))
  return [...catalog, ...configuredModels]
}

function cloneControlValue(
  control: ConfigurationDefaultsSetting['control'],
  value: string
): ConfigurationDefaultsSetting['control'] {
  return { ...control, value }
}

function sectionPathSegments(section: string | undefined): string[] {
  return section?.split('.').filter(Boolean) ?? []
}

function resolveConfigSettingPath(setting: ConfigurationDefaultsSetting): string[] {
  if (setting.canonicalPath?.startsWith('defaults.')) {
    return setting.canonicalPath.slice('defaults.'.length).split('.').filter(Boolean)
  }
  if (setting.canonicalPath && !setting.canonicalPath.startsWith('plugin.')) {
    return setting.canonicalPath.split('.').filter(Boolean)
  }

  const key = 'name' in setting.control ? setting.control.name : setting.id
  return [...sectionPathSegments(setting.tomlSection), key]
}

export function configurationDefaultsSchemaPathEntries(): ConfigurationDefaultsSchemaPathEntry[] {
  return []
}

function overlayDefaultsValues(
  harnessDefaults: ConfigurationDefaultsHarnessData,
  defaultsValues: ConfigurationDefaultsValues
): ConfigurationDefaultsHarnessData {
  return {
    ...harnessDefaults,
    settings: harnessDefaults.settings.map((setting) => ({
      ...setting,
      baselineValue: setting.baselineValue ?? setting.control.value,
      control: cloneControlValue(setting.control, defaultsValues[setting.id] ?? setting.control.value)
    }))
  }
}

function combineSettingsHarnessData(
  ...groups: readonly (ConfigurationSettingsHarnessData | undefined)[]
): ConfigurationSettingsHarnessData {
  const categoryById = new Map<string, ConfigurationDefaultsCategory>()
  const settingById = new Map<string, ConfigurationDefaultsSetting>()
  const preview = groups.flatMap((group) => group?.preview ?? [])

  for (const group of groups) {
    for (const category of group?.categories ?? []) categoryById.set(String(category.id), category)
    for (const setting of group?.settings ?? []) settingById.set(setting.id, setting)
  }

  return {
    categories: sortCategories(Array.from(categoryById.values())),
    settings: sortSettings(Array.from(settingById.values())),
    preview
  }
}

function readPath(source: unknown, path: readonly string[]): unknown {
  let current = source
  for (const segment of path) {
    if (!current || typeof current !== 'object' || !(segment in current)) return undefined
    current = (current as Record<string, unknown>)[segment]
  }
  return current
}

function writePath(target: Record<string, unknown>, path: readonly string[], value: unknown) {
  let current = target
  path.forEach((segment, index) => {
    const isLeaf = index === path.length - 1
    if (isLeaf) {
      current[segment] = value
      return
    }

    const next = current[segment]
    if (!next || typeof next !== 'object' || Array.isArray(next)) {
      current[segment] = {}
    }
    current = current[segment] as Record<string, unknown>
  })
}

function deletePath(target: Record<string, unknown>, path: readonly string[]): boolean {
  const [segment, ...rest] = path
  if (!segment || !(segment in target)) return Object.keys(target).length === 0

  if (rest.length === 0) {
    delete target[segment]
    return Object.keys(target).length === 0
  }

  const next = target[segment]
  if (!next || typeof next !== 'object' || Array.isArray(next)) return Object.keys(target).length === 0

  if (deletePath(next as Record<string, unknown>, rest)) delete target[segment]
  return Object.keys(target).length === 0
}

function serializeDefaultSettingValue(setting: ConfigurationDefaultsSetting, value: unknown): string | undefined {
  if (value == null) return undefined
  if (typeof value === 'boolean' && setting.control.kind === 'choice') {
    const optionValues = new Set(setting.control.options.map((option) => option.value))
    if (optionValues.has('on') && optionValues.has('off')) return value ? 'on' : 'off'
  }
  if (typeof value === 'string' && setting.control.kind === 'choice') return normalizedChoiceValue(value)
  if (Array.isArray(value) && setting.control.kind === 'text') return value.join(',')
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
    createConfigurationAttestationSettingsFromSchema(schema, controlState)
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

function modelEntryPathSegments(path: string): string[] {
  return path
    .replace(/^models\.<model-ref>\.?/, '')
    .split('.')
    .filter(Boolean)
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
  writeOptionalModelEntryPath(entry, 'models.<model-ref>.speculative.draft_model_path', config?.draftModelPath?.trim())
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

async function expectJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const message = await parseApiErrorBody(response)
    throw new ApiError(response.status, message, message)
  }

  return response.json() as Promise<T>
}

export async function fetchRuntimeControlBootstrap(): Promise<RuntimeControlBootstrapPayload> {
  const response = await fetch(`${env.managementApiUrl}/api/runtime/control-bootstrap`)
  return expectJson<RuntimeControlBootstrapPayload>(response)
}

export async function fetchRuntimeConfigSchema(): Promise<RuntimeConfigSchemaReference> {
  const response = await fetch(`${env.managementApiUrl}/api/runtime/config-schema`)
  return expectJson<RuntimeConfigSchemaReference>(response)
}

export async function fetchRuntimeConfigControlState(): Promise<RuntimeConfigControlStatePayload> {
  const response = await fetch(`${env.managementApiUrl}/api/runtime/config-control-state`)
  try {
    const payload = await expectJson<RuntimeConfigControlStatePayload>(response)
    return { settings: payload.settings ?? {} }
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) return { settings: {} }
    throw error
  }
}

export async function fetchRuntimeControlConfigSnapshot(endpoint: string): Promise<RuntimeControlConfigSnapshot> {
  const response = await fetch(`${env.managementApiUrl}/api/runtime/control/get-config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ endpoint })
  })
  const payload = await expectJson<RuntimeControlConfigResponse>(response)
  return payload.snapshot
}

export async function fetchRuntimeControlConfig(): Promise<RuntimeControlConfigResult> {
  const [bootstrap, schema, controlState] = await Promise.all([
    fetchRuntimeControlBootstrap(),
    fetchRuntimeConfigSchema(),
    fetchRuntimeConfigControlState()
  ])
  const endpoint = bootstrap.endpoint?.trim()

  if (!bootstrap.enabled || !endpoint) return { bootstrap, schema, controlState }

  const snapshot = await fetchRuntimeControlConfigSnapshot(endpoint)
  return { bootstrap: { ...bootstrap, endpoint }, schema, snapshot, controlState }
}

export async function applyRuntimeControlConfig(
  endpoint: string,
  snapshot: RuntimeControlConfigSnapshot,
  input: RuntimeControlApplyInput,
  schema?: RuntimeConfigSchemaReference,
  controlState?: RuntimeConfigControlStatePayload
): Promise<{ response: RuntimeControlApplyResponse; snapshot: RuntimeControlConfigSnapshot }> {
  const config = mergeConfigurationIntoMeshConfig(snapshot.config, input, schema, {
    includeModelAssignments: true,
    controlState
  })
  const response = await fetch(`${env.managementApiUrl}/api/runtime/control/apply-config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      endpoint,
      expected_revision: snapshot.revision,
      config
    })
  })
  const payload = await expectJson<RuntimeControlApplyResponse>(response)

  return {
    response: payload,
    snapshot: {
      ...snapshot,
      revision: payload.current_revision,
      config
    }
  }
}

export async function validateRuntimeConfigToml(toml: string, path?: string): Promise<RuntimeConfigValidateResponse> {
  const response = await fetch(`${env.managementApiUrl}/api/runtime/config/validate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ toml, path })
  })
  return expectJson<RuntimeConfigValidateResponse>(response)
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
  const schemaIntegrations = createConfigurationIntegrationsFromSchema(schema, controlState)
  const overlay = (settings: ConfigurationSettingsHarnessData) =>
    defaultsValues ? overlayDefaultsValues(settings, defaultsValues) : settings
  const plugins =
    schemaIntegrations && defaultsValues
      ? overlayDefaultsValues(schemaIntegrations, defaultsValues)
      : schemaIntegrations
  const legacyDefaults = overlay(
    combineSettingsHarnessData(meshllmSettings, runtimeSettings, modelSettings, network, attestation)
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
