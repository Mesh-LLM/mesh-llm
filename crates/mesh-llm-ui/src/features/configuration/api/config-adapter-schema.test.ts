import { describe, expect, it } from 'vitest'
import {
  adaptStatusToConfiguration,
  createConfigurationDefaultsFromSchema,
  createConfigurationDefaultsValuesFromMeshConfig,
  createConfigurationIntegrationsFromSchema,
  createConfigurationMeshLLMSettingsFromSchema,
  createConfigurationModelSettingsFromSchema,
  createConfigurationNetworkSettingsFromSchema
} from '@/features/configuration/api/config-adapter'
import type { MeshModelRaw } from '@/lib/api/types'
import { validateConfigurationSettingValue } from '@/features/configuration/components/settings/schema-field-validation'
import {
  BACKEND_DEFAULTS_UI_REFERENCE,
  BACKEND_SCHEMA_REFERENCE,
  CUSTOM_MODEL_PLACEMENT_SCHEMA,
  SCHEMA_REFERENCE,
  STATUS_PAYLOAD,
  schemaSetting
} from './config-adapter-test-support'
import type { RuntimeConfigSchemaReference } from './config-adapter-types'

describe('configuration schema and status adaptation', () => {
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

  it('maps loading and unknown node states without reporting them online', () => {
    const configuration = adaptStatusToConfiguration(
      {
        ...STATUS_PAYLOAD,
        node_state: 'loading',
        peers: [
          {
            id: 'unknown-peer',
            state: 'future-state',
            role: 'Worker',
            hostname: 'future.local',
            serving_models: [],
            vram_gb: 0
          }
        ]
      },
      []
    )

    expect(configuration.nodes[0]?.status).toBe('degraded')
    expect(configuration.nodes[1]?.status).toBe('offline')
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

  it('deduplicates repeated configured placeholder models', () => {
    const configuration = adaptStatusToConfiguration(STATUS_PAYLOAD, [], undefined, CUSTOM_MODEL_PLACEMENT_SCHEMA, {
      models: [
        { runtime: { source: 'hf://meshllm/dupe@main:Q4_K_M' } },
        { runtime: { source: 'hf://meshllm/dupe@main:Q4_K_M' } }
      ]
    })

    expect(configuration.catalog.filter((model) => model.id === 'hf://meshllm/dupe@main:Q4_K_M')).toHaveLength(1)
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

  it('projects typed topology stages into model settings', () => {
    const topologyStageSchema = {
      kind: 'array' as const,
      items: {
        kind: 'object' as const,
        properties: [
          {
            name: 'node',
            label: 'Node',
            required: true,
            value_schema: {
              kind: 'object' as const,
              properties: [
                {
                  name: 'endpoint_id',
                  label: 'Endpoint ID',
                  required: false,
                  value_schema: { kind: 'string' as const }
                },
                { name: 'hostname', label: 'Hostname', required: false, value_schema: { kind: 'string' as const } }
              ]
            }
          },
          { name: 'layer_start', label: 'Layer start', required: true, value_schema: { kind: 'integer' as const } },
          { name: 'layer_end', label: 'Layer end', required: true, value_schema: { kind: 'integer' as const } }
        ]
      }
    }
    const schema: RuntimeConfigSchemaReference = {
      settings: [schemaSetting('defaults.topology.stages', 'topology-stages', topologyStageSchema)]
    }

    const topologyStages = createConfigurationModelSettingsFromSchema(schema).settings.find(
      (setting) => setting.id === 'defaults.topology.stages'
    )

    expect(topologyStages?.valueSchema).toEqual(topologyStageSchema)
    expect(topologyStages?.control).toMatchObject({ kind: 'text', name: 'stages', value: '' })
  })

  it('places schema topology settings in the topology category when presentation is absent', () => {
    const schema: RuntimeConfigSchemaReference = {
      settings: [
        {
          ...schemaSetting('defaults.topology.mode', 'topology-mode', { kind: 'enum', values: ['locked'] }),
          presentation: undefined
        }
      ]
    }

    const topology = createConfigurationDefaultsFromSchema(schema)

    expect(topology.categories).toEqual([
      expect.objectContaining({
        id: 'topology',
        label: 'Topology',
        summary: 'Locked staged topology defaults.',
        help: 'Ordered layer ranges and node selectors for locked staged serving',
        tomlSection: 'defaults.topology'
      })
    ])
    expect(topology.settings[0]).toMatchObject({ categoryId: 'topology', icon: 'layers' })
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
})
