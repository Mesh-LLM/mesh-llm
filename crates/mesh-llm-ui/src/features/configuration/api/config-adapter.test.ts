import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import {
  adaptStatusToConfiguration,
  createConfigurationDefaultsFromSchema,
  createConfigurationDefaultsValuesFromMeshConfig,
  createConfigurationIntegrationsFromSchema,
  createConfigurationMeshLLMSettingsFromSchema,
  createConfigurationModelSettingsFromSchema,
  createConfigurationNetworkSettingsFromSchema,
  formatConfigDiagnostics,
  mergeConfigurationIntoMeshConfig,
  mergeConfigurationDefaultsIntoMeshConfig,
  runtimeControlApplyErrorMessage,
  type RuntimeConfigSchemaEntry,
  type RuntimeConfigSchemaReference,
  type RuntimeControlMeshConfig
} from '@/features/configuration/api/config-adapter'
import type { MeshModelRaw, StatusPayload } from '@/lib/api/types'
import { validateConfigurationSettingValue } from '@/features/configuration/components/settings/schema-field-validation'

type DefaultsUiSchemaReference = {
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

const BACKEND_SCHEMA_REFERENCE = loadRuntimeConfigSchemaReferenceFixture(
  '../../../../../mesh-llm-host-runtime/tests/fixtures/config_schema_reference.json'
)

const BACKEND_DEFAULTS_UI_REFERENCE = loadDefaultsUiSchemaReferenceFixture(
  '../../../../../mesh-llm-host-runtime/tests/fixtures/config_schema_defaults_ui_reference.json'
)

const STATUS_PAYLOAD: StatusPayload = {
  node_id: 'self',
  node_state: 'serving',
  model_name: 'Hermes-2-Pro-Mistral-7B-Q4_K_M',
  peers: [],
  models: [],
  my_vram_gb: 0,
  gpus: [],
  serving_models: []
}

const SCHEMA_REFERENCE: RuntimeConfigSchemaReference = {
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

function schemaSetting(
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

const CUSTOM_MODEL_PLACEMENT_SCHEMA: RuntimeConfigSchemaReference = {
  ...SCHEMA_REFERENCE,
  settings: [
    ...SCHEMA_REFERENCE.settings,
    schemaSetting('models.<model-ref>.runtime.source', 'model-placement-model', { kind: 'string' }),
    schemaSetting('models.<model-ref>.runtime.context', 'model-placement-context', { kind: 'integer' }),
    schemaSetting('models.<model-ref>.accelerator.target', 'model-placement-device', { kind: 'string' }),
    schemaSetting('models.<model-ref>.accelerator.layers', 'model-placement-gpu-layers', { kind: 'integer' })
  ]
}

describe('adaptStatusToConfiguration', () => {
  const [blackboardPluginInstance] = SCHEMA_REFERENCE.plugin_instances ?? []

  it('includes the local status node in model deployment data when there are no peers', () => {
    const configuration = adaptStatusToConfiguration(
      {
        ...STATUS_PAYLOAD,
        hostname: 'carrack.local',
        region: 'tor-1',
        gpus: [
          {
            idx: 0,
            name: 'RTX 5090',
            total_vram_gb: 34.2,
            reserved_bytes: 1073741824
          }
        ],
        peers: []
      },
      []
    )

    expect(configuration.nodes).toHaveLength(1)
    expect(configuration.nodes[0]).toEqual(
      expect.objectContaining({
        id: 'self',
        hostname: 'carrack.local',
        region: 'tor-1',
        status: 'online',
        gpus: [{ idx: 0, name: 'RTX 5090', totalGB: 34.2, systemTotalGB: 34.2, reservedGB: 1.073741824 }]
      })
    )
  })

  it('maps live Apple SOC status to unified-memory placement data', () => {
    const configuration = adaptStatusToConfiguration(
      {
        ...STATUS_PAYLOAD,
        my_is_soc: true,
        gpus: [
          {
            name: 'Apple M4 Pro',
            vram_bytes: 40200896512
          }
        ],
        peers: []
      },
      []
    )

    expect(configuration.nodes[0]).toEqual(
      expect.objectContaining({
        memoryTopology: 'unified',
        gpus: [
          expect.objectContaining({
            idx: 0,
            name: 'Apple M4 Pro',
            totalGB: 40,
            systemTotalGB: 40200896512 / 1_000_000_000
          })
        ]
      })
    )
  })

  it('accepts public status peers without node_id', () => {
    const configuration = adaptStatusToConfiguration(
      {
        ...STATUS_PAYLOAD,
        peers: [
          {
            id: 'aeac0d8e53',
            state: 'client',
            role: 'Client',
            hostname: '1266a345aeb9',
            serving_models: [],
            vram_gb: 0
          }
        ]
      },
      []
    )

    expect(configuration.nodes[0]).toEqual(expect.objectContaining({ id: 'self' }))
    expect(configuration.nodes[1]).toEqual(
      expect.objectContaining({
        id: 'aeac0d8e53',
        hostname: '1266a345aeb9',
        status: 'offline'
      })
    )
  })

  it('accepts public API model rows without a nested capabilities object', () => {
    const models: MeshModelRaw[] = [
      {
        name: 'Hermes-2-Pro-Mistral-7B-Q4_K_M',
        status: 'warm',
        size_gb: 4.4,
        node_count: 1,
        quantization: 'Q4_K_M',
        tokenizer: 'gpt2',
        layer_count: 32,
        head_count: 32,
        embedding_size: 4096,
        moe: false,
        vision: false
      }
    ]

    const configuration = adaptStatusToConfiguration(STATUS_PAYLOAD, models)

    expect(configuration.catalog[0]).toEqual(
      expect.objectContaining({
        id: 'Hermes-2-Pro-Mistral-7B-Q4_K_M',
        sizeGB: 4.4,
        ctxMaxK: 0,
        layers: 32,
        heads: 32,
        embed: 4096,
        tokenizer: 'gpt2',
        moe: false,
        vision: false
      })
    )
  })

  it('hydrates configured models from schema-derived placement paths', () => {
    const configuration = adaptStatusToConfiguration(STATUS_PAYLOAD, [], undefined, CUSTOM_MODEL_PLACEMENT_SCHEMA, {
      models: [
        {
          runtime: { source: 'hf://meshllm/custom@main:Q4_K_M', context: 6144 },
          accelerator: { target: 'cuda:2', layers: -1 },
          model: 'hf://meshllm/legacy@main:Q4_K_M'
        }
      ]
    })

    expect(configuration.assigns[0]).toEqual(
      expect.objectContaining({
        modelId: 'hf://meshllm/custom@main:Q4_K_M',
        containerIdx: 2,
        ctx: 6144
      })
    )
    expect(configuration.catalog.map((model) => model.id)).toContain('hf://meshllm/custom@main:Q4_K_M')
  })

  it('overlays hydrated runtime-control defaults onto the harness settings', () => {
    const defaultsValues = createConfigurationDefaultsValuesFromMeshConfig(
      {
        defaults: {
          throughput: {
            parallel: 8
          },
          hardware: {
            safety_margin_gb: 3.5
          },
          model_fit: {
            kv_cache_policy: 'quality'
          },
          request_defaults: {
            temperature: 0.8,
            reasoning_enabled: false
          }
        }
      },
      SCHEMA_REFERENCE
    )

    const configuration = adaptStatusToConfiguration(STATUS_PAYLOAD, [], defaultsValues, SCHEMA_REFERENCE)
    const values = Object.fromEntries(
      configuration.defaults.settings.map((setting) => [setting.id, setting.control.value])
    )

    expect(values['defaults.throughput.parallel']).toBe('8')
    expect(values['defaults.hardware.safety_margin_gb']).toBe('3.5')
    expect(values['defaults.model_fit.kv_cache_policy']).toBe('quality')
    expect(values['defaults.request_defaults.temperature']).toBe('0.8')
    expect(values['defaults.request_defaults.reasoning_enabled']).toBe('off')
  })

  it('builds defaults controls entirely from exported schema metadata', () => {
    const defaults = createConfigurationDefaultsFromSchema(SCHEMA_REFERENCE)
    const temperature = defaults.settings.find((setting) => setting.id === 'defaults.request_defaults.temperature')
    const reasoningEnabled = defaults.settings.find(
      (setting) => setting.id === 'defaults.request_defaults.reasoning_enabled'
    )
    const kvCache = defaults.settings.find((setting) => setting.id === 'defaults.model_fit.kv_cache_policy')
    const ctxSize = defaults.settings.find((setting) => setting.id === 'defaults.model_fit.ctx_size')

    expect(temperature).toMatchObject({
      id: 'defaults.request_defaults.temperature',
      canonicalPath: 'defaults.request_defaults.temperature',
      label: 'Temperature',
      control: expect.objectContaining({ kind: 'range', name: 'temperature' })
    })
    expect(reasoningEnabled).toMatchObject({
      canonicalPath: 'defaults.request_defaults.reasoning_enabled',
      label: 'Reasoning enabled',
      mutability: 'runtime',
      control: expect.objectContaining({
        kind: 'choice',
        value: 'auto',
        options: [
          { value: 'auto', label: 'auto' },
          { value: 'off', label: 'off' },
          { value: 'on', label: 'on' }
        ]
      })
    })
    expect(kvCache).toMatchObject({
      rendererId: 'kv-cache-policy',
      control: expect.objectContaining({
        kind: 'choice',
        options: expect.arrayContaining([{ value: 'quality', label: 'quality' }])
      })
    })
    expect(ctxSize).toMatchObject({
      rendererId: 'context-slider',
      control: expect.objectContaining({
        kind: 'range',
        value: '2048',
        min: 2048,
        max: 262144,
        step: 512
      })
    })
  })

  it('plumbs schema constraints onto generated UI settings and validation honors them', () => {
    const schema: RuntimeConfigSchemaReference = {
      settings: [
        {
          canonical_path: 'telemetry.service_name',
          owner: 'built_in',
          source: { kind: 'built_in' },
          value_schema: { kind: 'string' },
          support: 'supported',
          control_surfaces: ['config_file'],
          apply_mode: 'static_on_load',
          restart_scope: 'model_reload',
          visibility: 'user',
          constraints: [{ kind: 'allowed_pattern', pattern: '^[A-Za-z0-9_-]+$' }],
          presentation: {
            label: 'Service name',
            help: 'Human-readable service name.',
            category_id: 'telemetry',
            category_label: 'Telemetry',
            category_summary: 'Telemetry settings',
            category_order: 10,
            setting_order: 10,
            control_hint: 'text'
          }
        }
      ]
    }

    const meshllmSettings = createConfigurationMeshLLMSettingsFromSchema(schema)
    const serviceName = meshllmSettings.settings.find((setting) => setting.id === 'telemetry.service_name')

    expect(serviceName).toMatchObject({
      id: 'telemetry.service_name',
      canonicalPath: 'telemetry.service_name',
      validationConstraints: [{ kind: 'allowed_pattern', pattern: '^[A-Za-z0-9_-]+$' }]
    })

    expect(serviceName).not.toBeUndefined()
    if (!serviceName) return

    expect(validateConfigurationSettingValue(serviceName, 'good_service-name_01')).toEqual({ valid: true })
    expect(validateConfigurationSettingValue(serviceName, '@@*(!111---aa')).toMatchObject({
      valid: false,
      message: expect.stringContaining('invalid format')
    })
  })

  it('prefers schema enum metadata over legacy path heuristics for covered choices', () => {
    const schema: RuntimeConfigSchemaReference = {
      settings: [
        {
          ...schemaSetting('defaults.model_fit.flash_attention', 'flash-attention', {
            kind: 'enum',
            values: ['auto', 'on', 'off']
          }),
          presentation: {
            label: 'Flash attention',
            category_id: 'runtime',
            category_label: 'Runtime',
            category_summary: 'Runtime defaults',
            control_hint: 'segmented'
          }
        },
        {
          ...schemaSetting('defaults.throughput.tuning_profile', 'tuning-profile', {
            kind: 'enum',
            values: ['latency', 'balanced', 'throughput']
          }),
          presentation: {
            label: 'Tuning profile',
            category_id: 'runtime',
            category_label: 'Runtime',
            category_summary: 'Runtime defaults',
            control_hint: 'segmented'
          }
        },
        {
          ...schemaSetting('defaults.model_fit.cache_type_k', 'cache-type-k', {
            kind: 'enum',
            values: ['f16', 'q8_0', 'q6_k']
          }),
          presentation: {
            label: 'Cache type K',
            category_id: 'memory',
            category_label: 'Memory',
            category_summary: 'Memory defaults',
            control_hint: 'select'
          }
        },
        {
          ...schemaSetting('defaults.speculative.mode', 'speculative-mode', {
            kind: 'enum',
            values: ['auto', 'off', 'draft_only']
          }),
          presentation: {
            label: 'Speculative mode',
            category_id: 'speculative-decoding',
            category_label: 'Speculative Decoding',
            category_summary: 'Speculative defaults',
            control_hint: 'segmented'
          }
        }
      ]
    }

    const defaults = createConfigurationDefaultsFromSchema(schema)

    expect(
      defaults.settings.find((setting) => setting.id === 'defaults.model_fit.flash_attention')?.control
    ).toMatchObject({
      kind: 'choice',
      value: 'auto',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'on', label: 'on' },
        { value: 'off', label: 'off' }
      ]
    })
    expect(
      defaults.settings.find((setting) => setting.id === 'defaults.throughput.tuning_profile')?.control
    ).toMatchObject({
      kind: 'choice',
      value: 'latency',
      options: [
        { value: 'latency', label: 'latency' },
        { value: 'balanced', label: 'balanced' },
        { value: 'throughput', label: 'throughput' }
      ]
    })
    expect(
      defaults.settings.find((setting) => setting.id === 'defaults.model_fit.cache_type_k')?.control
    ).toMatchObject({
      kind: 'choice',
      value: 'f16',
      options: [
        { value: 'f16', label: 'f16' },
        { value: 'q8_0', label: 'q8_0' },
        { value: 'q6_k', label: 'q6_k' }
      ],
      presentation: 'select'
    })
    expect(defaults.settings.find((setting) => setting.id === 'defaults.speculative.mode')?.control).toMatchObject({
      kind: 'choice',
      value: 'auto',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'off', label: 'off' },
        { value: 'draft_only', label: 'draft_only' }
      ]
    })
  })

  it('keeps schema-covered open strings and structured values on text controls without path heuristics', () => {
    const schema: RuntimeConfigSchemaReference = {
      settings: [
        {
          ...schemaSetting('defaults.model_fit.flash_attention', 'flash-attention', { kind: 'string' }),
          presentation: {
            label: 'Flash attention policy',
            category_id: 'runtime',
            category_label: 'Runtime',
            category_summary: 'Runtime defaults',
            control_hint: 'text'
          }
        },
        {
          ...schemaSetting('defaults.multimodal.mmproj_path', 'projector-path', { kind: 'path' }),
          presentation: {
            label: 'Projector path',
            category_id: 'multimodal',
            category_label: 'Multimodal',
            category_summary: 'Multimodal defaults',
            control_hint: 'text'
          }
        },
        {
          ...schemaSetting('defaults.advanced.server.allowed_hosts', 'allowed-hosts', {
            kind: 'array',
            items: { kind: 'string' }
          }),
          presentation: {
            label: 'Allowed hosts',
            category_id: 'advanced-server',
            category_label: 'Advanced Server',
            category_summary: 'Advanced server defaults',
            control_hint: 'text'
          }
        },
        {
          ...schemaSetting('defaults.multimodal.embeddings', 'embeddings', { kind: 'object' }),
          presentation: {
            label: 'Embeddings override',
            category_id: 'multimodal',
            category_label: 'Multimodal',
            category_summary: 'Multimodal defaults',
            control_hint: 'text'
          }
        }
      ]
    }

    const defaults = createConfigurationDefaultsFromSchema(schema)

    expect(defaults.settings.find((setting) => setting.id === 'defaults.model_fit.flash_attention')?.control).toEqual({
      kind: 'text',
      name: 'flash_attention',
      value: '',
      placeholder: undefined
    })
    expect(defaults.settings.find((setting) => setting.id === 'defaults.multimodal.mmproj_path')?.control).toEqual({
      kind: 'text',
      name: 'mmproj_path',
      value: '',
      placeholder: undefined
    })
    expect(
      defaults.settings.find((setting) => setting.id === 'defaults.advanced.server.allowed_hosts')?.control
    ).toEqual({
      kind: 'text',
      name: 'allowed_hosts',
      value: '',
      placeholder: undefined
    })
    expect(defaults.settings.find((setting) => setting.id === 'defaults.multimodal.embeddings')?.control).toEqual({
      kind: 'text',
      name: 'embeddings',
      value: '',
      placeholder: 'JSON object'
    })
  })

  it('preserves current editability when control behavior metadata is missing', () => {
    const defaults = createConfigurationDefaultsFromSchema(SCHEMA_REFERENCE)
    const reasoningEnabled = defaults.settings.find(
      (setting) => setting.id === 'defaults.request_defaults.reasoning_enabled'
    )

    expect(
      SCHEMA_REFERENCE.settings.find((entry) => entry.canonical_path === reasoningEnabled?.id)?.control_behavior
    ).toBeUndefined()
    expect(reasoningEnabled).toMatchObject({
      mutability: 'runtime',
      control: expect.objectContaining({
        kind: 'choice',
        value: 'auto',
        options: [
          { value: 'auto', label: 'auto' },
          { value: 'off', label: 'off' },
          { value: 'on', label: 'on' }
        ]
      })
    })
    expect(reasoningEnabled?.controlState).toBeUndefined()
    expect(reasoningEnabled?.controlBehavior).toBeUndefined()
  })

  it('keeps path and url schema kinds on adapted settings', () => {
    const schema: RuntimeConfigSchemaReference = {
      settings: [
        {
          ...schemaSetting('defaults.multimodal.mmproj_path', 'projector-path', { kind: 'path' }),
          presentation: {
            label: 'Projector path',
            category_id: 'multimodal',
            category_label: 'Multimodal',
            category_summary: 'Multimodal defaults',
            control_hint: 'text'
          }
        },
        {
          ...schemaSetting('defaults.multimodal.mmproj_url', 'projector-url', { kind: 'url' }),
          presentation: {
            label: 'Projector URL',
            category_id: 'multimodal',
            category_label: 'Multimodal',
            category_summary: 'Multimodal defaults',
            control_hint: 'text'
          }
        }
      ]
    }

    const defaults = createConfigurationDefaultsFromSchema(schema)

    expect(defaults.settings.find((setting) => setting.id === 'defaults.multimodal.mmproj_path')?.valueSchema).toEqual({
      kind: 'path'
    })
    expect(defaults.settings.find((setting) => setting.id === 'defaults.multimodal.mmproj_url')?.valueSchema).toEqual({
      kind: 'url'
    })
  })

  it('attaches runtime control-state options to schema settings', () => {
    const schema: RuntimeConfigSchemaReference = {
      settings: [
        {
          ...schemaSetting('defaults.hardware.device', 'runtime-gpu-choice', { kind: 'string' }),
          control_behavior: {
            options_source: 'runtime_gpus',
            write_policy: 'preserve_existing'
          },
          presentation: {
            label: 'GPU device',
            category_id: 'runtime',
            category_label: 'Runtime',
            category_summary: 'Runtime defaults',
            control_hint: 'select'
          }
        }
      ]
    }

    const modelSettings = createConfigurationModelSettingsFromSchema(schema, {
      settings: {
        'defaults.hardware.device': {
          enabled: true,
          source: 'runtime',
          write_policy: 'preserve_existing',
          options: [
            {
              value: { kind: 'string', value: 'cuda:0' },
              label: 'NVIDIA RTX 5090 (cuda:0)',
              note: '31.8 GiB VRAM',
              disabled: false,
              source: 'runtime_gpus'
            }
          ]
        }
      }
    })
    const device = modelSettings.settings.find((setting) => setting.id === 'defaults.hardware.device')

    expect(device?.controlBehavior).toEqual({ options_source: 'runtime_gpus', write_policy: 'preserve_existing' })
    expect(device?.controlState).toMatchObject({ enabled: true, source: 'runtime' })
    expect(device?.control).toMatchObject({
      kind: 'choice',
      value: '',
      presentation: 'select',
      options: [
        { value: '', label: 'Select GPU' },
        { value: 'cuda:0', label: 'NVIDIA RTX 5090 (cuda:0)', description: '31.8 GiB VRAM' }
      ]
    })
  })

  it('keeps the backend defaults UI fixture and generated defaults settings in exact path parity', () => {
    const defaults = createConfigurationDefaultsFromSchema(BACKEND_SCHEMA_REFERENCE)
    const expectedDefaultPaths = BACKEND_DEFAULTS_UI_REFERENCE.settings
      .map((entry) => entry.canonical_path)
      .filter((canonicalPath) =>
        BACKEND_SCHEMA_REFERENCE.settings.some((entry) => entry.canonical_path === canonicalPath)
      )

    expect([...defaults.settings.map((setting) => setting.id)].sort()).toEqual([...expectedDefaultPaths].sort())
    expect(defaults.settings.every((setting) => setting.id === setting.canonicalPath)).toBe(true)
  })

  it('uses backend-exported metadata for schema-covered controls without reviving hard-coded fallbacks', () => {
    const modelSettings = createConfigurationModelSettingsFromSchema(BACKEND_SCHEMA_REFERENCE)
    const networkSettings = createConfigurationNetworkSettingsFromSchema(BACKEND_SCHEMA_REFERENCE)
    const integrations = createConfigurationIntegrationsFromSchema(BACKEND_SCHEMA_REFERENCE)

    const defaultsDevice = modelSettings.settings.find((setting) => setting.id === 'defaults.hardware.device')
    expect(defaultsDevice).toMatchObject({
      valueSchema: { kind: 'string' },
      control: { kind: 'text', name: 'device', value: '' },
      controlBehavior: {
        options_source: 'runtime_gpus',
        enable_when: [
          {
            operator: 'equals',
            path: {
              segments: [
                { kind: 'field', name: 'gpu' },
                { kind: 'field', name: 'assignment' }
              ]
            },
            values: [{ kind: 'string', value: 'pinned' }]
          }
        ]
      }
    })

    const legacyMmproj = modelSettings.settings.find((setting) => setting.id === 'defaults.hardware.mmproj')
    expect(legacyMmproj).toMatchObject({
      valueSchema: { kind: 'path' },
      control: { kind: 'text', name: 'mmproj', value: '' },
      controlBehavior: {
        write_policy: 'preserve_existing',
        availability: {
          enabled: false,
          source: 'static',
          reason: 'Edit defaults.multimodal.mmproj instead of the legacy hardware duplicate.',
          note: 'Existing values are preserved on save unless you change defaults.multimodal.mmproj.'
        }
      }
    })

    const multimodalMmprojOffload = modelSettings.settings.find(
      (setting) => setting.id === 'defaults.multimodal.mmproj_offload'
    )
    expect(multimodalMmprojOffload?.control).toMatchObject({
      kind: 'choice',
      value: 'auto',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'on', label: 'on' },
        { value: 'off', label: 'off' }
      ]
    })

    const multimodalMmprojUrl = modelSettings.settings.find(
      (setting) => setting.id === 'defaults.multimodal.mmproj_url'
    )
    expect(multimodalMmprojUrl).toMatchObject({
      valueSchema: { kind: 'url' },
      control: {
        kind: 'text',
        name: 'mmproj_url',
        value: '',
        placeholder: 'e.g. https://example.com/mmproj.gguf'
      },
      controlBehavior: { text_format: 'url' }
    })

    const advertiseAddr = networkSettings.settings.find((setting) => setting.id === 'owner_control.advertise_addr')
    expect(advertiseAddr).toMatchObject({
      valueSchema: { kind: 'socket_addr' },
      control: { kind: 'text', name: 'advertise_addr', value: '' },
      controlBehavior: {
        enable_when: [
          {
            operator: 'present',
            path: {
              segments: [
                { kind: 'field', name: 'owner_control' },
                { kind: 'field', name: 'bind' }
              ]
            }
          }
        ],
        disable_when: [
          {
            condition: {
              operator: 'absent',
              path: {
                segments: [
                  { kind: 'field', name: 'owner_control' },
                  { kind: 'field', name: 'bind' }
                ]
              }
            },
            reason:
              'owner_control.advertise_addr requires owner_control.bind so the advertised port is actually listening',
            write_policy: 'omit_when_disabled'
          }
        ]
      }
    })

    expect(integrations?.categories.map((category) => category.id)).toEqual(['plugin:blackboard', 'plugin:blobstore'])

    const pluginUrl = integrations?.settings.find((setting) => setting.id === 'plugin.blackboard.url')
    expect(pluginUrl).toMatchObject({
      valueSchema: { kind: 'url' },
      control: { kind: 'text', name: 'url', value: '' }
    })

    const pluginTimeout = integrations?.settings.find(
      (setting) => setting.id === 'plugin.blackboard.startup.connect_timeout_secs'
    )
    expect(pluginTimeout).toMatchObject({
      valueSchema: { kind: 'integer' },
      control: { kind: 'text', name: 'connect_timeout_secs', value: '' },
      controlBehavior: { numeric: { min: 1, unit: 'sec' } }
    })
  })

  it('preserves current editability when runtime control-state is empty', () => {
    const modelSettings = createConfigurationModelSettingsFromSchema(SCHEMA_REFERENCE, { settings: {} })
    const assignment = modelSettings.settings.find((setting) => setting.id === 'gpu.assignment')

    expect(assignment).toMatchObject({
      label: 'GPU assignment',
      control: expect.objectContaining({
        kind: 'choice',
        value: 'auto'
      })
    })
    expect(assignment?.controlState).toBeUndefined()
  })

  it('instantiates integration controls from plugin instances and plugin-owned schema settings', () => {
    const integrations = createConfigurationIntegrationsFromSchema(SCHEMA_REFERENCE)

    expect(integrations?.categories[0]).toMatchObject({
      id: 'plugin:blackboard',
      label: 'Blackboard'
    })
    expect(integrations?.settings.find((setting) => setting.id === 'plugin.blackboard.enabled')).toMatchObject({
      id: 'plugin.blackboard.enabled',
      canonicalPath: 'plugin.blackboard.enabled',
      label: 'Enabled',
      control: expect.objectContaining({
        kind: 'choice',
        value: 'on'
      })
    })
    expect(
      integrations?.settings.find((setting) => setting.id === 'plugin.blackboard.settings.retention_days')
    ).toMatchObject({
      id: 'plugin.blackboard.settings.retention_days',
      canonicalPath: 'plugin.blackboard.settings.retention_days',
      label: 'Retention days',
      control: expect.objectContaining({
        kind: 'range',
        min: 1,
        max: 365
      })
    })
  })

  it('places debug and listen-all settings on their requested tabs and writes runtime config', () => {
    const meshllm = createConfigurationMeshLLMSettingsFromSchema(SCHEMA_REFERENCE)
    const network = createConfigurationNetworkSettingsFromSchema(SCHEMA_REFERENCE)

    expect(meshllm.settings.find((setting) => setting.id === 'runtime.debug')).toMatchObject({
      label: 'Debug output',
      categoryId: 'meshllm',
      tomlSection: 'runtime',
      tomlKey: 'debug',
      control: expect.objectContaining({ kind: 'choice', value: 'off' })
    })
    expect(network.settings.find((setting) => setting.id === 'runtime.listen_all')).toMatchObject({
      label: 'Listen on all interfaces',
      categoryId: 'network',
      tomlSection: 'runtime',
      tomlKey: 'listen_all',
      control: expect.objectContaining({ kind: 'choice', value: 'off' })
    })

    const merged = mergeConfigurationDefaultsIntoMeshConfig(
      { version: 1 },
      {
        'runtime.debug': 'on',
        'runtime.listen_all': 'on'
      },
      SCHEMA_REFERENCE
    )

    expect(merged.runtime).toMatchObject({
      debug: true,
      listen_all: true
    })
  })

  it('places gpu assignment controls on the models tab instead of meshllm', () => {
    const meshllm = createConfigurationMeshLLMSettingsFromSchema(SCHEMA_REFERENCE)
    const models = createConfigurationModelSettingsFromSchema(SCHEMA_REFERENCE)

    expect(meshllm.settings.find((setting) => setting.id === 'gpu.assignment')).toBeUndefined()
    expect(meshllm.settings.find((setting) => setting.id === 'gpu.parallel')).toBeUndefined()
    expect(models.settings.find((setting) => setting.id === 'gpu.assignment')).toMatchObject({
      label: 'GPU assignment',
      categoryId: 'runtime',
      tomlSection: 'gpu',
      tomlKey: 'assignment'
    })
    expect(models.settings.find((setting) => setting.id === 'gpu.parallel')).toMatchObject({
      label: 'GPU parallelism',
      categoryId: 'runtime',
      tomlSection: 'gpu',
      tomlKey: 'parallel'
    })
  })

  it('hydrates and merges schema-derived defaults and plugin settings', () => {
    const values = createConfigurationDefaultsValuesFromMeshConfig(
      {
        telemetry: {
          headers: {}
        },
        defaults: {
          request_defaults: {
            reasoning_enabled: false
          }
        },
        plugin: [
          {
            name: 'blackboard',
            enabled: false,
            url: 'http://localhost:8000/v1',
            settings: {
              retention_days: 30
            }
          }
        ]
      },
      SCHEMA_REFERENCE
    )

    expect(values['defaults.request_defaults.reasoning_enabled']).toBe('off')
    expect(values['plugin.blackboard.enabled']).toBe('off')
    expect(values['plugin.blackboard.url']).toBe('http://localhost:8000/v1')
    expect(values['plugin.blackboard.settings.retention_days']).toBe('30')

    const merged = mergeConfigurationDefaultsIntoMeshConfig(
      { version: 1, plugin: [{ name: 'blackboard', enabled: true }] },
      {
        ...values,
        'defaults.request_defaults.reasoning_enabled': 'on',
        'plugin.blackboard.enabled': 'off',
        'plugin.blackboard.settings.retention_days': '45'
      },
      SCHEMA_REFERENCE
    )

    expect(merged).toMatchObject({
      version: 1,
      defaults: {
        request_defaults: {
          reasoning_enabled: true
        }
      },
      plugin: [
        {
          name: 'blackboard',
          enabled: false,
          url: 'http://localhost:8000/v1',
          settings: {
            retention_days: 45
          }
        }
      ]
    })
  })

  it('hydrates empty telemetry headers as an empty editable object value', () => {
    const schema: RuntimeConfigSchemaReference = {
      settings: [
        {
          ...schemaSetting('telemetry.headers', 'telemetry-headers', { kind: 'object' }),
          presentation: {
            label: 'Telemetry headers',
            category_id: 'telemetry',
            category_label: 'Telemetry',
            category_summary: 'Telemetry settings',
            control_hint: 'text'
          }
        }
      ]
    }

    const values = createConfigurationDefaultsValuesFromMeshConfig(
      {
        telemetry: {
          headers: {}
        }
      },
      schema
    )

    expect(values['telemetry.headers']).toBe('')
  })

  it('preserves dotted plugin names and literal dotted plugin setting keys in runtime-control merges', () => {
    const dottedPluginSchema: RuntimeConfigSchemaReference = {
      ...SCHEMA_REFERENCE,
      plugin_instances: [
        {
          name: 'com.example.tool',
          enabled: true,
          source_repository: 'mesh-llm/com-example-tool',
          installed_version: '0.2.0',
          has_config_schema: true,
          allow_unvalidated_config: false
        }
      ],
      settings: [
        ...SCHEMA_REFERENCE.settings.filter(
          (entry) =>
            entry.canonical_path !== 'plugin.blackboard.settings.retention_days' &&
            !entry.canonical_path.startsWith('plugin.blackboard.settings.')
        ),
        {
          canonical_path: 'plugin.com.example.tool.settings.foo-bar',
          owner: 'plugin',
          source: { kind: 'plugin', plugin_name: 'com.example.tool', allow_unvalidated_config: false },
          value_schema: { kind: 'string' },
          support: 'supported',
          control_surfaces: ['config_file', 'owner_control', 'plugin_manifest'],
          apply_mode: 'dynamic_apply',
          restart_scope: 'none',
          visibility: 'advanced',
          description: 'Preserve dashed plugin setting keys',
          presentation: {
            label: 'Foo bar',
            help: 'Preserve dashed plugin setting keys',
            category_id: 'plugin:com.example.tool',
            category_label: 'Com Example Tool',
            category_summary: 'Plugin settings',
            category_order: 20,
            setting_order: 10,
            control_hint: 'text'
          }
        },
        {
          canonical_path: 'plugin.com.example.tool.settings.nested.key',
          owner: 'plugin',
          source: { kind: 'plugin', plugin_name: 'com.example.tool', allow_unvalidated_config: false },
          value_schema: { kind: 'string' },
          support: 'supported',
          control_surfaces: ['config_file', 'owner_control', 'plugin_manifest'],
          apply_mode: 'dynamic_apply',
          restart_scope: 'none',
          visibility: 'advanced',
          description: 'Preserve literal dotted plugin setting keys',
          presentation: {
            label: 'Nested key',
            help: 'Preserve literal dotted plugin setting keys',
            category_id: 'plugin:com.example.tool',
            category_label: 'Com Example Tool',
            category_summary: 'Plugin settings',
            category_order: 20,
            setting_order: 20,
            control_hint: 'text'
          }
        }
      ]
    }

    const values = createConfigurationDefaultsValuesFromMeshConfig(
      {
        plugin: [
          {
            name: 'com.example.tool',
            enabled: true,
            url: 'http://localhost:7010/v1',
            settings: {
              'foo-bar': 'kept',
              'nested.key': 'literal'
            }
          }
        ]
      },
      dottedPluginSchema
    )

    expect(values['plugin.com.example.tool.enabled']).toBe('on')
    expect(values['plugin.com.example.tool.url']).toBe('http://localhost:7010/v1')
    expect(values['plugin.com.example.tool.settings.foo-bar']).toBe('kept')
    expect(values['plugin.com.example.tool.settings.nested.key']).toBe('literal')

    const merged = mergeConfigurationDefaultsIntoMeshConfig({ version: 1 }, values, dottedPluginSchema)

    expect(merged.plugin).toEqual([
      {
        name: 'com.example.tool',
        url: 'http://localhost:7010/v1',
        settings: {
          'foo-bar': 'kept',
          'nested.key': 'literal'
        }
      }
    ])
  })

  it('preserves opaque plugin settings and dotted plugin keys when applying a subset for allow_unvalidated_config plugins', () => {
    const dottedPluginSchema: RuntimeConfigSchemaReference = {
      ...SCHEMA_REFERENCE,
      plugin_instances: [
        {
          name: 'com.example.tool',
          enabled: true,
          source_repository: 'mesh-llm/com-example-tool',
          installed_version: '0.2.0',
          has_config_schema: true,
          allow_unvalidated_config: true
        }
      ],
      settings: [
        ...SCHEMA_REFERENCE.settings.filter(
          (entry) =>
            entry.canonical_path !== 'plugin.blackboard.settings.retention_days' &&
            !entry.canonical_path.startsWith('plugin.blackboard.settings.')
        ),
        {
          canonical_path: 'plugin.com.example.tool.settings.foo-bar',
          owner: 'plugin',
          source: { kind: 'plugin', plugin_name: 'com.example.tool', allow_unvalidated_config: true },
          value_schema: { kind: 'string' },
          support: 'supported',
          control_surfaces: ['config_file', 'owner_control', 'plugin_manifest'],
          apply_mode: 'dynamic_apply',
          restart_scope: 'none',
          visibility: 'advanced',
          presentation: {
            label: 'Foo bar',
            help: 'Preserve dashed plugin setting keys',
            category_id: 'plugin:com.example.tool',
            category_label: 'Com Example Tool',
            category_summary: 'Plugin settings',
            category_order: 20,
            setting_order: 10,
            control_hint: 'text'
          }
        },
        {
          canonical_path: 'plugin.com.example.tool.settings.nested.key',
          owner: 'plugin',
          source: { kind: 'plugin', plugin_name: 'com.example.tool', allow_unvalidated_config: true },
          value_schema: { kind: 'string' },
          support: 'supported',
          control_surfaces: ['config_file', 'owner_control', 'plugin_manifest'],
          apply_mode: 'dynamic_apply',
          restart_scope: 'none',
          visibility: 'advanced',
          presentation: {
            label: 'Nested key',
            help: 'Preserve literal dotted plugin setting keys',
            category_id: 'plugin:com.example.tool',
            category_label: 'Com Example Tool',
            category_summary: 'Plugin settings',
            category_order: 20,
            setting_order: 20,
            control_hint: 'text'
          }
        }
      ]
    }
    const meshConfig: RuntimeControlMeshConfig = {
      version: 1,
      plugin: [
        {
          name: 'com.example.tool',
          enabled: true,
          url: 'http://localhost:7010/v1',
          settings: {
            'foo-bar': 'kept',
            'nested.key': 'literal',
            opaque_json: '{"keep":true}'
          }
        },
        {
          name: 'telemetry',
          enabled: true
        }
      ]
    }

    const merged = mergeConfigurationDefaultsIntoMeshConfig(
      meshConfig,
      {
        'plugin.com.example.tool.settings.foo-bar': 'updated'
      },
      dottedPluginSchema
    )

    expect(merged.plugin).toEqual([
      {
        name: 'com.example.tool',
        enabled: true,
        url: 'http://localhost:7010/v1',
        settings: {
          'foo-bar': 'updated',
          'nested.key': 'literal',
          opaque_json: '{"keep":true}'
        }
      },
      {
        name: 'telemetry',
        enabled: true
      }
    ])
  })

  it('keeps disabled installed plugins disabled when writing custom settings', () => {
    if (!blackboardPluginInstance) throw new Error('Expected blackboard plugin fixture')

    const disabledSchema: RuntimeConfigSchemaReference = {
      ...SCHEMA_REFERENCE,
      plugin_instances: [
        {
          ...blackboardPluginInstance,
          enabled: false
        }
      ]
    }

    const merged = mergeConfigurationDefaultsIntoMeshConfig(
      { version: 1 },
      {
        'plugin.blackboard.enabled': 'off',
        'plugin.blackboard.settings.retention_days': '45'
      },
      disabledSchema
    )

    expect(merged.plugin).toEqual([
      {
        name: 'blackboard',
        enabled: false,
        settings: {
          retention_days: 45
        }
      }
    ])
  })

  it('merges only modified defaults back into the full mesh config without dropping unrelated fields', () => {
    const meshConfig: RuntimeControlMeshConfig = {
      version: 1,
      owner_control: {
        bind: '127.0.0.1:7447'
      },
      telemetry: {
        enabled: true
      },
      models: [{ model: 'hf://meshllm/base@main:Q4_K_M', ctx_size: 8192 }],
      plugin: [{ name: 'telemetry', enabled: true }],
      defaults: {
        throughput: {
          threads: 6,
          parallel: 5
        },
        hardware: {
          mlock: false,
          safety_margin_gb: 1.5
        },
        request_defaults: {
          temperature: 0.8,
          reasoning_format: 'deepseek'
        },
        speculative: {
          draft_max_tokens: 16
        },
        advanced: {
          server: {
            alias: 'existing-alias'
          }
        }
      }
    }

    const merged = mergeConfigurationDefaultsIntoMeshConfig(
      meshConfig,
      {
        'defaults.throughput.parallel': '8',
        'defaults.hardware.safety_margin_gb': '3.5',
        'defaults.request_defaults.temperature': '1.0'
      },
      SCHEMA_REFERENCE
    )

    expect(merged).toEqual({
      version: 1,
      owner_control: {
        bind: '127.0.0.1:7447'
      },
      telemetry: {
        enabled: true
      },
      models: [{ model: 'hf://meshllm/base@main:Q4_K_M', ctx_size: 8192 }],
      plugin: [{ name: 'telemetry', enabled: true }],
      defaults: {
        throughput: {
          threads: 6,
          parallel: 8
        },
        hardware: {
          mlock: false,
          safety_margin_gb: 3.5
        },
        request_defaults: {
          reasoning_format: 'deepseek',
          temperature: 1
        },
        speculative: {
          draft_max_tokens: 16
        },
        advanced: {
          server: {
            alias: 'existing-alias'
          }
        }
      }
    })
    expect(meshConfig.defaults).toEqual({
      throughput: {
        threads: 6,
        parallel: 5
      },
      hardware: {
        mlock: false,
        safety_margin_gb: 1.5
      },
      request_defaults: {
        temperature: 0.8,
        reasoning_format: 'deepseek'
      },
      speculative: {
        draft_max_tokens: 16
      },
      advanced: {
        server: {
          alias: 'existing-alias'
        }
      }
    })
  })

  it('preserves known defaults plus unrelated models and plugins when applying a subset of values', () => {
    const meshConfig: RuntimeControlMeshConfig = {
      version: 1,
      models: [{ model: 'hf://meshllm/base@main:Q4_K_M', ctx_size: 8192 }],
      plugin: [{ name: 'telemetry', enabled: true }],
      defaults: {
        request_defaults: {
          temperature: 0.8,
          reasoning_enabled: false,
          reasoning_format: 'deepseek'
        }
      }
    }

    const merged = mergeConfigurationDefaultsIntoMeshConfig(
      meshConfig,
      {
        'defaults.request_defaults.temperature': '1.0'
      },
      SCHEMA_REFERENCE
    )

    expect(merged).toEqual({
      version: 1,
      models: [{ model: 'hf://meshllm/base@main:Q4_K_M', ctx_size: 8192 }],
      plugin: [{ name: 'telemetry', enabled: true }],
      defaults: {
        request_defaults: {
          temperature: 1,
          reasoning_enabled: false,
          reasoning_format: 'deepseek'
        }
      }
    })
  })

  it('keeps synthetic blobstore integration behavior for built-in plugin host templates', () => {
    const blobstoreSchema: RuntimeConfigSchemaReference = {
      settings: [
        {
          ...schemaSetting('plugin.<plugin-name>.enabled', 'plugin-enabled', { kind: 'boolean' }),
          presentation: {
            label: 'Enabled',
            category_id: 'plugin-host',
            category_label: 'Plugin Host',
            category_summary: 'Plugin host settings',
            control_hint: 'toggle'
          }
        },
        {
          ...schemaSetting('plugin.<plugin-name>.url', 'plugin-url', { kind: 'url' }),
          presentation: {
            label: 'URL',
            category_id: 'plugin-host',
            category_label: 'Plugin Host',
            category_summary: 'Plugin host settings',
            control_hint: 'text'
          }
        }
      ]
    }

    const integrations = createConfigurationIntegrationsFromSchema(blobstoreSchema)

    expect(integrations?.categories).toEqual([
      expect.objectContaining({
        id: 'plugin:blobstore',
        label: 'Blobstore'
      })
    ])
    expect(integrations?.settings.map((setting) => setting.id)).toEqual(['plugin.blobstore.enabled'])
  })

  it('preserves disabled values when the evaluator resolves preserve_existing', () => {
    const schema: RuntimeConfigSchemaReference = {
      settings: [
        {
          ...schemaSetting('gpu.assignment', 'gpu-assignment', { kind: 'enum', values: ['auto', 'pinned'] }),
          presentation: {
            label: 'GPU assignment',
            category_id: 'runtime',
            category_label: 'Runtime',
            category_summary: 'Runtime defaults',
            renderer_id: 'gpu-assignment',
            control_hint: 'segmented'
          }
        },
        {
          ...schemaSetting('defaults.hardware.device', 'gpu-device', { kind: 'string' }),
          control_behavior: {
            enable_when: [
              {
                path: { segments: ['gpu', 'assignment'] },
                operator: 'equals',
                values: [{ kind: 'string', value: 'pinned' }]
              }
            ],
            write_policy: 'preserve_existing'
          },
          presentation: {
            label: 'GPU device',
            category_id: 'runtime',
            category_label: 'Runtime',
            category_summary: 'Runtime defaults',
            renderer_id: 'gpu-device',
            control_hint: 'text'
          }
        }
      ]
    }
    const meshConfig: RuntimeControlMeshConfig = {
      version: 1,
      gpu: { assignment: 'auto' },
      defaults: { hardware: { device: 'cuda:0' } }
    }
    const values = {
      'gpu.assignment': 'auto',
      'defaults.hardware.device': 'cuda:0'
    }

    const merged = mergeConfigurationDefaultsIntoMeshConfig(meshConfig, values, schema)

    expect(merged).toMatchObject({
      version: 1,
      defaults: { hardware: { device: 'cuda:0' } }
    })
  })

  it('omits dependency-disabled values when no write policy override is provided', () => {
    const schema: RuntimeConfigSchemaReference = {
      settings: [
        {
          ...schemaSetting('defaults.speculative.mode', 'speculative-mode', {
            kind: 'enum',
            values: ['draft', 'disabled']
          }),
          presentation: {
            label: 'Speculative mode',
            category_id: 'runtime',
            category_label: 'Runtime',
            category_summary: 'Runtime defaults',
            renderer_id: 'speculative-mode',
            control_hint: 'segmented'
          }
        },
        {
          ...schemaSetting('defaults.speculative.draft_max_tokens', 'draft-max-tokens', { kind: 'integer' }),
          control_behavior: {
            enable_when: [
              {
                path: { segments: ['defaults', 'speculative', 'mode'] },
                operator: 'equals',
                values: [{ kind: 'string', value: 'draft' }]
              }
            ]
          },
          presentation: {
            label: 'Draft max tokens',
            category_id: 'runtime',
            category_label: 'Runtime',
            category_summary: 'Runtime defaults',
            renderer_id: 'draft-max-tokens',
            control_hint: 'number'
          }
        }
      ]
    }
    const meshConfig: RuntimeControlMeshConfig = {
      version: 1,
      defaults: {
        speculative: {
          mode: 'disabled',
          draft_max_tokens: 16
        }
      }
    }
    const values = createConfigurationDefaultsValuesFromMeshConfig(meshConfig, schema)

    const merged = mergeConfigurationDefaultsIntoMeshConfig(meshConfig, values, schema)

    expect(merged).toEqual({
      version: 1,
      defaults: {
        speculative: {
          mode: 'disabled'
        }
      }
    })
  })

  it('blocks saving disabled values when the evaluator resolves reject_when_disabled', () => {
    const schema: RuntimeConfigSchemaReference = {
      settings: [
        {
          ...schemaSetting('runtime.rpc_backend', 'rpc-backend', { kind: 'string' }),
          control_behavior: {
            availability: {
              enabled: false,
              reason: 'External RPC backends are not supported.',
              source: 'static'
            },
            write_policy: 'reject_when_disabled'
          },
          presentation: {
            label: 'RPC backend',
            category_id: 'runtime',
            category_label: 'Runtime',
            category_summary: 'Runtime settings',
            renderer_id: 'rpc-backend',
            control_hint: 'text'
          }
        }
      ]
    }

    try {
      mergeConfigurationDefaultsIntoMeshConfig(
        { version: 1 },
        {
          'runtime.rpc_backend': 'remote'
        },
        schema
      )
      expect.unreachable('Expected merge to block reject_when_disabled writes')
    } catch (error: unknown) {
      expect(error).toBeInstanceOf(Error)
      if (!(error instanceof Error)) throw error
      expect(error.message).toContain('runtime.rpc_backend')
      expect(error.message).toContain('External RPC backends are not supported.')
      if (typeof error === 'object' && error && 'diagnostics' in error) {
        expect(error).toMatchObject({
          diagnostics: [
            expect.objectContaining({
              code: 'disabled_write_rejected',
              canonical_path: 'runtime.rpc_backend',
              severity: 'error'
            })
          ]
        })
      }
    }
  })

  it('uses runtime control-state overlays when merge-time write policy is runtime-disabled', () => {
    const schema: RuntimeConfigSchemaReference = {
      settings: [
        {
          ...schemaSetting('runtime.rpc_backend', 'rpc-backend', { kind: 'string' }),
          control_behavior: {
            options_source: 'runtime_native_backends'
          },
          presentation: {
            label: 'RPC backend',
            category_id: 'runtime',
            category_label: 'Runtime',
            category_summary: 'Runtime settings',
            renderer_id: 'rpc-backend',
            control_hint: 'text'
          }
        }
      ]
    }

    try {
      mergeConfigurationDefaultsIntoMeshConfig(
        { version: 1 },
        {
          'runtime.rpc_backend': 'remote'
        },
        schema,
        {
          settings: {
            'runtime.rpc_backend': {
              enabled: false,
              source: 'runtime',
              reason: 'Runtime backends are unavailable on this host.',
              write_policy: 'reject_when_disabled'
            }
          }
        }
      )
      expect.unreachable('Expected runtime overlay to block reject_when_disabled writes')
    } catch (error: unknown) {
      expect(error).toBeInstanceOf(Error)
      if (!(error instanceof Error)) throw error
      expect(error.message).toContain('runtime.rpc_backend')
      expect(error.message).toContain('Runtime backends are unavailable on this host.')
    }
  })

  it('consumes duplicate model entries in order and clears stale GPU targeting for pooled placement', () => {
    const merged = mergeConfigurationIntoMeshConfig(
      {
        version: 1,
        models: [
          {
            model: 'hf://meshllm/dupe@main:Q4_K_M',
            model_fit: {
              ctx_size: 2048,
              cache_type_k: 'q8_0',
              cache_type_v: 'q4_0',
              kv_cache_policy: 'balanced'
            },
            hardware: { device: 'cuda:0', gpu_layers: -1 },
            keep: 'first'
          },
          {
            model: 'hf://meshllm/dupe@main:Q4_K_M',
            model_fit: { ctx_size: 4096, cache_type_k: 'f16', cache_type_v: 'f16' },
            hardware: { device: 'cuda:1', gpu_layers: -1 },
            keep: 'second'
          }
        ]
      },
      {
        values: {},
        nodes: [
          {
            id: 'self',
            hostname: 'local',
            region: 'local',
            status: 'online',
            cpu: 'cpu',
            ramGB: 64,
            gpus: [],
            placement: 'pooled'
          }
        ],
        assigns: [
          {
            id: 'assign-1',
            modelId: 'hf://meshllm/dupe@main:Q4_K_M',
            nodeId: 'self',
            containerIdx: 0,
            ctx: 8192,
            config: {
              slots: 3,
              batchProfile: 'balanced',
              splitMode: 'layer',
              tensorSplit: '60,40',
              mmproj: '/models/mmproj.gguf',
              draftModelPath: '/models/draft.gguf',
              flashAttention: 'enabled',
              cacheTypeK: 'q8_0',
              cacheTypeV: 'q5_1',
              kvCachePolicy: 'balanced'
            }
          },
          { id: 'assign-2', modelId: 'hf://meshllm/dupe@main:Q4_K_M', nodeId: 'self', containerIdx: 1, ctx: 16384 }
        ],
        catalog: []
      },
      SCHEMA_REFERENCE,
      { includeModelAssignments: true }
    )

    expect(merged.models).toEqual([
      {
        model: 'hf://meshllm/dupe@main:Q4_K_M',
        model_fit: {
          ctx_size: 8192,
          batch: 512,
          ubatch: 128,
          cache_type_k: 'q8_0',
          cache_type_v: 'q5_1',
          kv_cache_policy: 'balanced',
          flash_attention: 'enabled'
        },
        hardware: {
          split_mode: 'layer',
          tensor_split: '60,40'
        },
        multimodal: {
          mmproj: '/models/mmproj.gguf'
        },
        speculative: {
          draft_model: '/models/draft.gguf'
        },
        throughput: {
          parallel: 3
        },
        keep: 'first'
      },
      {
        model: 'hf://meshllm/dupe@main:Q4_K_M',
        model_fit: { ctx_size: 16384, cache_type_k: 'f16', cache_type_v: 'f16' },
        keep: 'second'
      }
    ])
  })

  it('parses exact array text controls with their item schema', () => {
    const arraySchema: RuntimeConfigSchemaReference = {
      ...SCHEMA_REFERENCE,
      settings: [
        ...SCHEMA_REFERENCE.settings,
        {
          ...schemaSetting('defaults.skippy.integer_list', 'integer-list', {
            kind: 'array',
            items: { kind: 'integer' }
          }),
          presentation: {
            label: 'Integer list',
            category_id: 'test',
            category_label: 'Test',
            category_summary: 'Test settings',
            control_hint: 'text'
          }
        }
      ]
    }

    const merged = mergeConfigurationDefaultsIntoMeshConfig(
      { version: 1 },
      { 'defaults.skippy.integer_list': '1, 2, invalid' },
      arraySchema
    )

    expect(merged.defaults).toEqual({ skippy: { integer_list: [1, 2, 'invalid'] } })
  })

  it('removes known defaults when saved UI values return to canonical defaults', () => {
    const meshConfig: RuntimeControlMeshConfig = {
      version: 1,
      defaults: {
        request_defaults: {
          temperature: 0.7,
          reasoning_format: 'qwen'
        },
        custom_extension: {
          keep: true
        }
      }
    }

    const merged = mergeConfigurationDefaultsIntoMeshConfig(
      meshConfig,
      {
        'defaults.request_defaults.temperature': '0'
      },
      SCHEMA_REFERENCE
    )

    expect(merged).toEqual({
      version: 1,
      defaults: {
        request_defaults: {
          reasoning_format: 'qwen'
        },
        custom_extension: {
          keep: true
        }
      }
    })
  })

  it('extracts runtime-control apply error messages from structured payloads', () => {
    expect(
      runtimeControlApplyErrorMessage({
        success: false,
        current_revision: 7,
        config_hash: 'abc123',
        apply_mode: 'unspecified',
        error: { code: 'revision_conflict', message: 'config revision changed on disk' }
      })
    ).toBe('config revision changed on disk')

    expect(
      runtimeControlApplyErrorMessage({
        success: false,
        current_revision: 7,
        config_hash: 'abc123',
        apply_mode: 'unspecified',
        error: { code: 'control_unavailable' }
      })
    ).toBe('control unavailable')

    expect(
      runtimeControlApplyErrorMessage({
        success: false,
        current_revision: 7,
        config_hash: 'abc123',
        apply_mode: 'unspecified',
        diagnostics: [
          {
            code: 'invalid_value',
            severity: 'error',
            source: 'validation',
            path: 'models[0].request_defaults.reasoning_format',
            canonical_path: 'models.<model-ref>.request_defaults.reasoning_format',
            message: 'reasoning_format must be one of: auto, none, deepseek, deepseek-legacy, hidden',
            help: 'choose one of the supported reasoning formats'
          }
        ]
      })
    ).toBe(
      [
        '**`models[0].request_defaults.reasoning_format`** · `ERROR`',
        '',
        'reasoning_format must be one of: auto, none, deepseek, deepseek-legacy, hidden',
        '',
        '> **Help:** choose one of the supported reasoning formats'
      ].join('\n')
    )
  })

  it('formats a single error diagnostic as markdown', () => {
    expect(
      formatConfigDiagnostics([
        {
          code: 'invalid_value',
          severity: 'error',
          source: 'validation',
          path: 'mesh_requirements.require_release_attestation',
          message:
            'mesh_requirements.require_release_attestation is true but mesh_requirements.release_signer_keys is empty',
          help: 'set at least one release signer key or disable require_release_attestation'
        }
      ])
    ).toBe(
      [
        '**`mesh_requirements.require_release_attestation`** · `ERROR`',
        '',
        'mesh_requirements.require_release_attestation is true but mesh_requirements.release_signer_keys is empty',
        '',
        '> **Help:** set at least one release signer key or disable require_release_attestation'
      ].join('\n')
    )
  })

  it('formats multiple diagnostics separated by a horizontal rule', () => {
    const result = formatConfigDiagnostics([
      {
        code: 'missing_value',
        severity: 'error',
        source: 'validation',
        path: 'mesh_requirements.release_signer_keys',
        message: 'release_signer_keys is empty',
        help: 'add at least one signer key'
      },
      {
        code: 'conflict',
        severity: 'warning',
        source: 'validation',
        path: 'mesh_requirements.some_other',
        message: 'this setting conflicts with another',
        help: 'resolve the conflict'
      }
    ])

    const blocks = result!.split('\n\n---\n\n')
    expect(blocks).toHaveLength(2)

    expect(blocks[0]).toBe(
      [
        '**`mesh_requirements.release_signer_keys`** · `ERROR`',
        '',
        'release_signer_keys is empty',
        '',
        '> **Help:** add at least one signer key'
      ].join('\n')
    )
    expect(blocks[1]).toBe(
      [
        '**`mesh_requirements.some_other`** · `WARNING`',
        '',
        'this setting conflicts with another',
        '',
        '> **Help:** resolve the conflict'
      ].join('\n')
    )
  })

  it('omits path and help when they are not provided', () => {
    expect(
      formatConfigDiagnostics([
        {
          code: 'general_error',
          severity: 'error',
          source: 'validation',
          message: 'something went wrong'
        }
      ])
    ).toBe(['`ERROR`', '', 'something went wrong'].join('\n'))
  })

  it('returns undefined for an empty diagnostics array', () => {
    expect(formatConfigDiagnostics([])).toBeUndefined()
  })

  it('keeps values keyed by canonical schema paths', () => {
    const defaults = createConfigurationDefaultsFromSchema(SCHEMA_REFERENCE)

    expect(defaults.settings.map((setting) => setting.id)).toEqual([
      'defaults.throughput.parallel',
      'defaults.hardware.safety_margin_gb',
      'defaults.model_fit.ctx_size',
      'defaults.model_fit.kv_cache_policy',
      'defaults.request_defaults.temperature',
      'defaults.request_defaults.reasoning_enabled'
    ])
    expect(defaults.settings.every((setting) => setting.id === setting.canonicalPath)).toBe(true)
  })
})
