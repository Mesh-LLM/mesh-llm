import { readFileSync } from 'node:fs'
import type { StatusPayload } from '@/lib/api/types'
import type { RuntimeConfigSchemaEntry, RuntimeConfigSchemaReference } from './config-adapter-types'

export type DefaultsUiSchemaReference = {
  readonly settings: readonly {
    readonly canonical_path: string
    readonly support: string
    readonly source: { readonly kind: string }
  }[]
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isRuntimeConfigSchemaEntry(value: unknown): value is RuntimeConfigSchemaEntry {
  return (
    isRecord(value) &&
    typeof value.canonical_path === 'string' &&
    isRecord(value.source) &&
    typeof value.source.kind === 'string' &&
    isRecord(value.value_schema) &&
    typeof value.value_schema.kind === 'string' &&
    typeof value.support === 'string' &&
    Array.isArray(value.control_surfaces) &&
    typeof value.apply_mode === 'string' &&
    typeof value.restart_scope === 'string' &&
    typeof value.visibility === 'string'
  )
}

function isRuntimeConfigSchemaReference(value: unknown): value is RuntimeConfigSchemaReference {
  if (!isRecord(value) || !Array.isArray(value.settings)) return false
  return value.settings.every(isRuntimeConfigSchemaEntry)
}

function isDefaultsUiSchemaReference(value: unknown): value is DefaultsUiSchemaReference {
  return (
    isRecord(value) &&
    Array.isArray(value.settings) &&
    value.settings.every(
      (entry) =>
        isRecord(entry) &&
        typeof entry.canonical_path === 'string' &&
        typeof entry.support === 'string' &&
        isRecord(entry.source) &&
        typeof entry.source.kind === 'string'
    )
  )
}

function loadFixture(relativePath: string): unknown {
  return JSON.parse(readFileSync(new URL(relativePath, import.meta.url), 'utf8'))
}

function loadRuntimeConfigSchemaReferenceFixture(relativePath: string): RuntimeConfigSchemaReference {
  const fixture = loadFixture(relativePath)
  if (!isRuntimeConfigSchemaReference(fixture)) {
    throw new Error(`Invalid runtime schema fixture: ${relativePath}`)
  }
  return fixture
}

function loadDefaultsUiSchemaReferenceFixture(relativePath: string): DefaultsUiSchemaReference {
  const fixture = loadFixture(relativePath)
  if (!isDefaultsUiSchemaReference(fixture)) {
    throw new Error(`Invalid defaults UI schema fixture: ${relativePath}`)
  }
  return fixture
}

export const BACKEND_SCHEMA_REFERENCE = loadRuntimeConfigSchemaReferenceFixture(
  '../../../../../mesh-llm-host-runtime/tests/fixtures/config_schema_reference.json'
)

export const BACKEND_DEFAULTS_UI_REFERENCE = loadDefaultsUiSchemaReferenceFixture(
  '../../../../../mesh-llm-host-runtime/tests/fixtures/config_schema_defaults_ui_reference.json'
)

export const STATUS_PAYLOAD: StatusPayload = {
  node_id: 'self',
  node_state: 'serving',
  model_name: 'Hermes-2-Pro-Mistral-7B-Q4_K_M',
  peers: [],
  models: [],
  my_vram_gb: 0,
  gpus: [],
  serving_models: []
}

export const SCHEMA_REFERENCE: RuntimeConfigSchemaReference = {
  plugin_instances: [
    {
      name: 'blackboard',
      enabled: true,
      source_repository: 'mesh-llm/blackboard',
      installed_version: '0.1.0',
      has_config_schema: true,
      allow_unvalidated_config: false
    }
  ],
  settings: [
    {
      canonical_path: 'gpu.assignment',
      owner: 'built_in',
      source: { kind: 'built_in' },
      value_schema: { kind: 'enum', values: ['auto', 'pinned'] },
      support: 'supported',
      control_surfaces: ['config_file'],
      apply_mode: 'static_on_load',
      restart_scope: 'model_reload',
      visibility: 'user',
      presentation: {
        label: 'GPU assignment',
        help: 'Choose automatic GPU placement or require configured models to pick a GPU.',
        category_id: 'runtime',
        category_label: 'Runtime',
        category_summary: 'Runtime defaults',
        category_order: 10,
        setting_order: 5,
        control_hint: 'segmented'
      }
    },
    {
      canonical_path: 'gpu.parallel',
      owner: 'built_in',
      source: { kind: 'built_in' },
      value_schema: { kind: 'integer' },
      support: 'supported',
      control_surfaces: ['config_file'],
      apply_mode: 'static_on_load',
      restart_scope: 'model_reload',
      visibility: 'user',
      presentation: {
        label: 'GPU parallelism',
        help: 'Limit the local GPU startup parallelism used when configured models are launched.',
        category_id: 'runtime',
        category_label: 'Runtime',
        category_summary: 'Runtime defaults',
        category_order: 10,
        setting_order: 6,
        unit: 'models',
        control_hint: 'number'
      }
    },
    {
      canonical_path: 'defaults.throughput.parallel',
      owner: 'built_in',
      source: { kind: 'built_in' },
      value_schema: { kind: 'integer' },
      support: 'supported',
      control_surfaces: ['config_file'],
      apply_mode: 'static_on_load',
      restart_scope: 'model_reload',
      visibility: 'user',
      presentation: {
        label: 'Default slots / parallel requests',
        help: 'Sets the default parallel slots.',
        category_id: 'runtime',
        category_label: 'Runtime',
        category_summary: 'Runtime defaults',
        category_order: 10,
        setting_order: 10,
        unit: 'slots',
        control_hint: 'range',
        renderer_id: 'slot-meter'
      }
    },
    {
      canonical_path: 'defaults.hardware.safety_margin_gb',
      owner: 'built_in',
      source: { kind: 'built_in' },
      value_schema: { kind: 'float' },
      support: 'supported',
      control_surfaces: ['config_file'],
      apply_mode: 'static_on_load',
      restart_scope: 'model_reload',
      visibility: 'user',
      presentation: {
        label: 'Memory / safety margin',
        help: 'Keep GPU memory free.',
        category_id: 'memory',
        category_label: 'Memory',
        category_summary: 'Memory defaults',
        category_order: 20,
        setting_order: 10,
        unit: 'GB',
        control_hint: 'range'
      }
    },
    {
      canonical_path: 'defaults.model_fit.ctx_size',
      owner: 'built_in',
      source: { kind: 'built_in' },
      value_schema: { kind: 'integer' },
      support: 'supported',
      control_surfaces: ['config_file'],
      apply_mode: 'static_on_load',
      restart_scope: 'model_reload',
      visibility: 'user',
      presentation: {
        label: 'Context window size',
        help: 'Set the default context window size in tokens.',
        category_id: 'memory',
        category_label: 'Memory',
        category_summary: 'Memory defaults',
        category_order: 20,
        setting_order: 15,
        unit: 'tokens',
        control_hint: 'range',
        renderer_id: 'context-slider'
      }
    },
    {
      canonical_path: 'defaults.model_fit.kv_cache_policy',
      owner: 'built_in',
      source: { kind: 'built_in' },
      value_schema: { kind: 'string' },
      support: 'supported',
      control_surfaces: ['config_file'],
      apply_mode: 'static_on_load',
      restart_scope: 'model_reload',
      visibility: 'user',
      presentation: {
        label: 'KV cache policy',
        help: 'Select KV cache policy.',
        category_id: 'memory',
        category_label: 'Memory',
        category_summary: 'Memory defaults',
        category_order: 20,
        setting_order: 20,
        control_hint: 'segmented',
        renderer_id: 'kv-cache-policy'
      }
    },
    {
      canonical_path: 'defaults.request_defaults.temperature',
      owner: 'built_in',
      source: { kind: 'built_in' },
      value_schema: { kind: 'float' },
      support: 'supported',
      control_surfaces: ['config_file'],
      apply_mode: 'static_on_load',
      restart_scope: 'model_reload',
      visibility: 'advanced',
      constraints: [{ kind: 'range', min: '0', max: '2' }],
      presentation: {
        label: 'Temperature',
        help: 'Fallback sampling temperature.',
        category_id: 'request-defaults',
        category_label: 'Request Defaults',
        category_summary: 'Request defaults',
        category_order: 30,
        setting_order: 10,
        control_hint: 'range'
      }
    },
    {
      canonical_path: 'defaults.request_defaults.reasoning_enabled',
      owner: 'built_in',
      source: { kind: 'built_in' },
      value_schema: {
        kind: 'one_of',
        variants: [{ kind: 'boolean' }, { kind: 'enum', values: ['auto', 'off', 'on'] }]
      },
      support: 'supported',
      control_surfaces: ['config_file'],
      apply_mode: 'dynamic_apply',
      restart_scope: 'none',
      visibility: 'user',
      description: 'Choose whether reasoning is enabled by default.',
      presentation: {
        label: 'Reasoning enabled',
        help: 'Choose whether reasoning is enabled by default.',
        category_id: 'request-defaults',
        category_label: 'Request Defaults',
        category_summary: 'Request defaults',
        category_order: 30,
        setting_order: 20,
        control_hint: 'segmented'
      }
    },
    {
      canonical_path: 'runtime.debug',
      owner: 'built_in',
      source: { kind: 'built_in' },
      value_schema: { kind: 'boolean' },
      support: 'supported',
      control_surfaces: ['config_file', 'api'],
      apply_mode: 'dynamic_validation_only',
      restart_scope: 'model_reload',
      visibility: 'user',
      presentation: {
        label: 'Debug output',
        help: 'Enable mesh runtime debug output on startup.',
        category_id: 'meshllm',
        category_label: 'General',
        category_summary: 'Local node startup and observability settings',
        category_order: 10,
        setting_order: 30,
        control_hint: 'toggle'
      }
    },
    {
      canonical_path: 'runtime.listen_all',
      owner: 'built_in',
      source: { kind: 'built_in' },
      value_schema: { kind: 'boolean' },
      support: 'supported',
      control_surfaces: ['config_file', 'api'],
      apply_mode: 'dynamic_validation_only',
      restart_scope: 'model_reload',
      visibility: 'user',
      presentation: {
        label: 'Listen on all interfaces',
        help: 'Bind listeners to 0.0.0.0 instead of 127.0.0.1.',
        category_id: 'network',
        category_label: 'Network',
        category_summary: 'Owner-control listener and advertised control endpoint settings',
        category_order: 20,
        setting_order: 30,
        control_hint: 'toggle'
      }
    },
    {
      canonical_path: 'plugin.<plugin-name>.enabled',
      owner: 'built_in',
      source: { kind: 'built_in' },
      value_schema: { kind: 'boolean' },
      support: 'supported',
      control_surfaces: ['config_file', 'plugin_manifest'],
      apply_mode: 'static_on_load',
      restart_scope: 'process_restart',
      visibility: 'user',
      presentation: {
        label: 'Enabled',
        help: 'Enable or disable the plugin.',
        category_id: 'plugin-host',
        category_label: 'Plugin Host',
        category_summary: 'Plugin host settings',
        category_order: 10,
        setting_order: 10,
        control_hint: 'toggle'
      }
    },
    {
      canonical_path: 'plugin.<plugin-name>.url',
      owner: 'built_in',
      source: { kind: 'built_in' },
      value_schema: { kind: 'string' },
      support: 'supported',
      control_surfaces: ['config_file', 'plugin_manifest'],
      apply_mode: 'static_on_load',
      restart_scope: 'process_restart',
      visibility: 'user',
      presentation: {
        label: 'Base URL',
        help: 'Plugin endpoint URL.',
        category_id: 'plugin-host',
        category_label: 'Plugin Host',
        category_summary: 'Plugin host settings',
        category_order: 10,
        setting_order: 20,
        placeholder: 'http://localhost:8000/v1',
        control_hint: 'text'
      }
    },
    {
      canonical_path: 'plugin.blackboard.settings.retention_days',
      owner: 'plugin',
      source: { kind: 'plugin', plugin_name: 'blackboard', allow_unvalidated_config: false },
      value_schema: { kind: 'integer' },
      support: 'supported',
      control_surfaces: ['config_file', 'owner_control', 'plugin_manifest'],
      apply_mode: 'dynamic_apply',
      restart_scope: 'process_restart',
      visibility: 'advanced',
      constraints: [{ kind: 'range', min: '1', max: '365' }],
      description: 'Retention period in days',
      presentation: {
        label: 'Retention days',
        help: 'Retention period in days',
        category_id: 'retention',
        category_label: 'Retention',
        category_summary: 'Retention settings',
        category_order: 20,
        setting_order: 10,
        unit: 'days',
        control_hint: 'range'
      }
    }
  ]
}

export function schemaSetting(
  canonicalPath: string,
  rendererId: string,
  valueSchema: RuntimeConfigSchemaEntry['value_schema']
): RuntimeConfigSchemaEntry {
  return {
    canonical_path: canonicalPath,
    owner: 'built_in',
    source: { kind: 'built_in' },
    value_schema: valueSchema,
    support: 'supported',
    control_surfaces: ['config_file'],
    apply_mode: 'static_on_load',
    restart_scope: 'model_reload',
    visibility: 'user',
    presentation: {
      label: canonicalPath,
      category_id: 'models',
      category_label: 'Models',
      category_summary: 'Model placement',
      renderer_id: rendererId,
      control_hint: 'text'
    }
  }
}

export function loggingSchemaSetting(
  canonicalPath: string,
  valueSchema: RuntimeConfigSchemaEntry['value_schema'],
  overrides: Partial<RuntimeConfigSchemaEntry> = {}
): RuntimeConfigSchemaEntry {
  return {
    canonical_path: canonicalPath,
    owner: 'built_in',
    source: { kind: 'built_in' },
    value_schema: valueSchema,
    support: 'supported',
    control_surfaces: ['config_file'],
    apply_mode: 'static_on_load',
    restart_scope: 'process_restart',
    visibility: 'advanced',
    ...overrides
  }
}

export const CUSTOM_MODEL_PLACEMENT_SCHEMA: RuntimeConfigSchemaReference = {
  ...SCHEMA_REFERENCE,
  settings: [
    ...SCHEMA_REFERENCE.settings,
    schemaSetting('models.<model-ref>.runtime.source', 'model-placement-model', { kind: 'string' }),
    schemaSetting('models.<model-ref>.runtime.context', 'model-placement-context', { kind: 'integer' }),
    schemaSetting('models.<model-ref>.accelerator.target', 'model-placement-device', { kind: 'string' }),
    schemaSetting('models.<model-ref>.accelerator.layers', 'model-placement-gpu-layers', { kind: 'integer' })
  ]
}

export const AUDIT_SCHEMA: RuntimeConfigSchemaReference = {
  ...SCHEMA_REFERENCE,
  settings: [
    ...SCHEMA_REFERENCE.settings,
    { ...schemaSetting('logging.audit.enabled', 'audit-enabled', { kind: 'boolean' }), presentation: undefined },
    {
      ...schemaSetting('logging.audit.log_format', 'audit-log-format', {
        kind: 'enum',
        values: ['json', 'json_lines']
      }),
      presentation: undefined
    }
  ]
}
