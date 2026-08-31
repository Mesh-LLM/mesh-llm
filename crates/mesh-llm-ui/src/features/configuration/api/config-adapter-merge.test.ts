import { describe, expect, it } from 'vitest'
import {
  createConfigurationDefaultsValuesFromMeshConfig,
  createConfigurationIntegrationsFromSchema,
  createConfigurationMeshLLMSettingsFromSchema,
  createConfigurationModelSettingsFromSchema,
  mergeConfigurationDefaultsIntoMeshConfig
} from '@/features/configuration/api/config-adapter'
import { SCHEMA_REFERENCE, schemaSetting } from './config-adapter-test-support'
import type { RuntimeConfigSchemaReference, RuntimeControlMeshConfig } from './config-adapter-types'

describe('runtime-control configuration merges', () => {
  const [blackboardPluginInstance] = SCHEMA_REFERENCE.plugin_instances ?? []
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

  it('hydrates and merges typed topology stage arrays as JSON', () => {
    const topologyStageSchema = {
      kind: 'array' as const,
      items: {
        kind: 'object' as const,
        properties: [
          { name: 'node', label: 'Node', required: true, value_schema: { kind: 'object' as const } },
          { name: 'layer_start', label: 'Layer start', required: true, value_schema: { kind: 'integer' as const } },
          { name: 'layer_end', label: 'Layer end', required: true, value_schema: { kind: 'integer' as const } }
        ]
      }
    }
    const schema: RuntimeConfigSchemaReference = {
      settings: [schemaSetting('defaults.topology.stages', 'topology-stages', topologyStageSchema)]
    }
    const stages = [
      { node: { endpoint_id: 'endpoint-a' }, layer_start: 0, layer_end: 16 },
      { node: { hostname: 'worker-b' }, layer_start: 16, layer_end: 32 }
    ]

    const values = createConfigurationDefaultsValuesFromMeshConfig({ defaults: { topology: { stages } } }, schema)
    const merged = mergeConfigurationDefaultsIntoMeshConfig(
      { version: 1 },
      { 'defaults.topology.stages': JSON.stringify(stages) },
      schema
    )

    expect(values['defaults.topology.stages']).toBe(JSON.stringify(stages))
    expect(merged.defaults).toEqual({ topology: { stages } })
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
})
