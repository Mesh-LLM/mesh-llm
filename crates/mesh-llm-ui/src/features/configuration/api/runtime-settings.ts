import type {
  ConfigurationDefaultsCategory,
  ConfigurationDefaultsControl,
  ConfigurationDefaultsSetting as ConfigDefaultsSetting,
  ConfigurationHarnessData,
  ConfigurationRuntimeControlStateEntry
} from '@/features/app-tabs/types'

import { CONFIGURATION_HARNESS } from '@/features/app-tabs/data'
import type { RuntimeConfigControlStatePayload, RuntimeConfigSchemaReference } from './config-adapter'
import { createSchemaControl } from './schema-control-factory'
import {
  DEFAULT_CATEGORY_ORDER,
  DEFAULT_SETTING_ORDER,
  sortCategories,
  sortSettings,
  titleCaseIdentifier
} from './schema-setting-order'

const ENABLED_SETTING_ORDER = 10

/** Build runtime-policy settings harness data (runtime.mode, activity.*, etc.) from schema. */
export function createRuntimePolicySettingsFromSchema(
  schema: RuntimeConfigSchemaReference | undefined,
  controlState?: RuntimeConfigControlStatePayload
): ConfigurationHarnessData['defaults'] {
  if (!schema) return CONFIGURATION_HARNESS.defaults

  const entries = (schema.settings ?? []).filter(isRuntimePolicyEntry)
  const settings = entries.map((entry) => schemaSettingFromEntry(entry, controlState))
  const categoryById = new Map<string, ConfigurationDefaultsCategory>()

  for (const entry of entries) {
    const category = categoryFromEntry(entry)
    categoryById.set(String(category.id), category)
  }

  return {
    categories: sortCategories(Array.from(categoryById.values())),
    settings: sortSettings(settings),
    preview: [
      { label: 'Runtime Policy', value: `${settings.length} settings`, meta: 'schema' },
      { label: 'Source', value: '/api/runtime/config-schema', meta: 'live' }
    ]
  }
}

// ── Internal helpers ────────────────────────────────────────────────

function isRuntimePolicyEntry(entry: RuntimeConfigSchemaReference['settings'][number]): boolean {
  return (
    entry.canonical_path.startsWith('runtime.') &&
    entry.canonical_path !== 'runtime.debug' &&
    entry.canonical_path !== 'runtime.listen_all'
  )
}

const FALLBACK_DEFAULTS_CATEGORY: ConfigurationDefaultsCategory = {
  id: 'advanced',
  label: 'Advanced',
  summary: 'Schema-derived advanced settings.',
  help: 'Additional supported config settings from the exported schema'
}

function categoryForPath(canonicalPath: string) {
  if (canonicalPath.startsWith('runtime.')) return 'runtime-policy'
  return 'advanced'
}

const CATEGORY_FALLBACKS: Record<string, ConfigurationDefaultsCategory> = {
  'runtime-policy': {
    id: 'runtime-policy',
    label: 'Runtime Policy',
    summary: 'Daemon mode and activity policy settings',
    help: 'Settings that control how the daemon behaves at runtime',
    tomlSection: 'runtime',
    order: 10
  }
}

const CATEGORY_ICON_BY_ID: Record<string, ConfigDefaultsSetting['icon']> = {
  'runtime-policy': 'cog'
}

function categoryFromEntry(entry: RuntimeConfigSchemaReference['settings'][number]): ConfigurationDefaultsCategory {
  const categoryId = entry.presentation?.category_id ?? categoryForPath(entry.canonical_path)
  const fallback = CATEGORY_FALLBACKS[categoryId] ?? FALLBACK_DEFAULTS_CATEGORY

  return {
    ...fallback,
    id: categoryId,
    label: entry.presentation?.category_label ?? fallback.label,
    summary: entry.presentation?.category_summary ?? fallback.summary,
    help: entry.presentation?.help ?? fallback.help,
    tomlSection: entry.canonical_path.startsWith('runtime.') ? 'runtime' : undefined,
    order: entry.presentation?.category_order ?? fallback.order ?? DEFAULT_CATEGORY_ORDER
  }
}

function controlFromEntry(
  entry: RuntimeConfigSchemaReference['settings'][number],
  controlState?: ConfigurationRuntimeControlStateEntry
): ConfigurationDefaultsControl {
  const name = lastPathSegment(entry.canonical_path)

  if ((controlState?.options?.length ?? 0) > 0) {
    return createSchemaControl({ entry, name, runtimeControlState: controlState })
  }

  if (entry.value_schema.kind === 'enum') {
    return {
      kind: 'choice',
      name,
      value: '',
      presentation: entry.presentation?.control_hint === 'select' ? 'select' : 'segmented',
      options: entry.value_schema.values.map((v) => ({ value: v, label: titleCaseIdentifier(v) }))
    }
  }

  if (entry.value_schema.kind === 'boolean') {
    return {
      kind: 'choice',
      name,
      value: '',
      presentation: 'toggle',
      options: [
        { value: 'on', label: 'On' },
        { value: 'off', label: 'Off' }
      ]
    }
  }

  if (entry.value_schema.kind === 'integer') {
    const range = entry.constraints?.find((c) => c.kind === 'range') as
      { kind: 'range'; min?: string; max?: string } | undefined
    return {
      kind: 'text',
      name,
      value: '',
      placeholder: `${range?.min ?? '1'}–${range?.max ?? '3600'}`
    }
  }

  if (entry.value_schema.kind === 'float') {
    const range = entry.constraints?.find((c) => c.kind === 'range') as
      { kind: 'range'; min?: string; max?: string } | undefined
    return {
      kind: 'text',
      name,
      value: '',
      placeholder: `${range?.min ?? '0'}–${range?.max ?? '1'}`
    }
  }

  return { kind: 'text', name, value: '' }
}

function schemaMutability(
  entry: RuntimeConfigSchemaReference['settings'][number]
): ConfigDefaultsSetting['mutability'] {
  return entry.apply_mode === 'dynamic_apply' && entry.restart_scope === 'none' ? 'runtime' : 'restart-required'
}

function runtimeControlStateForPath(
  controlState: RuntimeConfigControlStatePayload | undefined,
  canonicalPath: string
): ConfigurationRuntimeControlStateEntry | undefined {
  return controlState?.settings?.[canonicalPath]
}

function schemaSettingFromEntry(
  entry: RuntimeConfigSchemaReference['settings'][number],
  controlState?: RuntimeConfigControlStatePayload
): ConfigDefaultsSetting {
  const name = lastPathSegment(entry.canonical_path)
  const category = categoryFromEntry(entry)
  const entryControlState = runtimeControlStateForPath(controlState, entry.canonical_path)

  return {
    id: entry.canonical_path,
    categoryId: category.id,
    canonicalPath: entry.canonical_path,
    tomlSection: 'runtime',
    tomlKey: name,
    rendererId: undefined,
    controlHint: entry.presentation?.control_hint,
    settingOrder:
      entry.presentation?.setting_order ?? (name === 'enabled' ? ENABLED_SETTING_ORDER : DEFAULT_SETTING_ORDER),
    icon: CATEGORY_ICON_BY_ID[category.id] ?? 'cog',
    label: entry.presentation?.label ?? titleCaseIdentifier(name),
    description: entry.presentation?.help ?? (entry.description ? entry.description : entry.canonical_path),
    inheritedLabel: 'Written to the local mesh-llm config file',
    valueSchema: entry.value_schema,
    control: controlFromEntry(entry, entryControlState),
    controlBehavior: entry.control_behavior,
    controlState: entryControlState,
    visibility: entry.visibility === 'advanced' ? 'advanced' : 'standard',
    mutability: schemaMutability(entry),
    applyMode: entry.apply_mode,
    restartScope: entry.restart_scope,
    validationConstraints: entry.constraints,
    categoryOrder: category.order ?? DEFAULT_CATEGORY_ORDER
  }
}

function lastPathSegment(canonicalPath: string) {
  return canonicalPath.split('.').filter(Boolean).at(-1) ?? canonicalPath
}
