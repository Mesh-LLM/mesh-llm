import type {
  ConfigurationDefaultsChoice,
  ConfigurationDefaultsValues,
  ConfigurationModelPlacementPaths,
  ConfigurationRuntimeControlStateEntry,
  ConfigurationSettingControlBehavior,
  ConfigurationSettingValueSchema,
  ConfigAssign,
  ConfigNode,
  ConfigModel
} from '@/features/app-tabs/types'

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
  choices?: readonly ConfigurationDefaultsChoice[]
  display_units?: readonly { value: string; label: string; multiplier: number }[]
}

export type RuntimeConfigSchemaSource =
  | { kind: 'built_in' }
  | { kind: 'engine'; engine_id: string }
  | { kind: 'plugin'; plugin_name: string; allow_unvalidated_config: boolean }

export type RuntimeConfigConstraint =
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

export type ConfigurationDefaultsSchemaPathEntry = {
  id: string
  canonicalPath: string
}
