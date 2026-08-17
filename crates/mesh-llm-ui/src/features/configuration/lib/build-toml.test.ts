import { describe, expect, it } from 'vitest'
import { CONFIGURATION_DEFAULTS, CONFIGURATION_HARNESS } from '@/features/app-tabs/data'
import type { ConfigAssign, ConfigModel, ConfigNode, ConfigurationDefaultsHarnessData } from '@/features/app-tabs/types'
import { buildTOML, defaultSettingTomlScalar } from '@/features/configuration/lib/build-toml'

describe('buildTOML', () => {
  it('escapes generated string values and omits legacy model keys', () => {
    const node: ConfigNode = {
      id: 'node "quoted"',
      hostname: 'mesh\\host\nalpha',
      region: 'iad "1"',
      status: 'online',
      cpu: 'test cpu',
      ramGB: 64,
      gpus: [],
      placement: 'pooled'
    }
    const assign: ConfigAssign = {
      id: 'assign "quoted"',
      modelId: 'custom\\model\nname',
      nodeId: node.id,
      containerIdx: 0,
      ctx: 4096
    }

    const toml = buildTOML([node], [assign])

    expect(toml).toContain('version = 1')
    expect(toml).toContain(`model = ${JSON.stringify(assign.modelId)}`)
    expect(toml).toContain('ctx_size = 4096')
    expect(toml).not.toContain('ctx = 4096')
    expect(toml).not.toContain('gpu_index =')
    expect(toml).not.toContain('[node]')
    expect(toml).not.toContain(`id = ${JSON.stringify(assign.id)}`)
  })

  it('emits changed canonical defaults and compact model override aliases', () => {
    const toml = buildTOML(CONFIGURATION_HARNESS.nodes, CONFIGURATION_HARNESS.assigns, CONFIGURATION_HARNESS.catalog, {
      defaults: CONFIGURATION_DEFAULTS,
      defaultsValues: {
        'ctx-size': '32768',
        batch: '1024',
        ubatch: '256',
        'cache-type-k': 'q8_0',
        'cache-type-v': 'q4_0',
        'flash-attention': 'enabled',
        'parallel-slots': '2',
        'memory-margin': '3.5',
        mmproj: '/models/projector.gguf',
        'speculation-mode': 'draft',
        'incompatible-pairing-behavior': 'fail_closed',
        'draft-max-tokens': '32',
        temperature: '0.55',
        'reasoning-format': 'deepseek-legacy'
      }
    })

    expect(toml).toContain('version = 1')
    expect(toml).toContain('[defaults]')
    expect(toml).toContain('ctx_size = 32768')
    expect(toml).toContain('batch = 1024')
    expect(toml).toContain('ubatch = 256')
    expect(toml).toContain('cache_type_k = "q8_0"')
    expect(toml).toContain('cache_type_v = "q4_0"')
    expect(toml).toContain('flash_attention = "enabled"')
    expect(toml).toContain('parallel = 2')
    expect(toml).toContain('mmproj = "/models/projector.gguf"')
    expect(toml).not.toContain('[defaults.model_fit]')
    expect(toml).toContain('[defaults.hardware]')
    expect(toml).toContain('safety_margin_gb = 3.5')
    expect(toml).toContain('[defaults.speculative]')
    expect(toml).toContain('mode = "draft"')
    expect(toml).toContain('pairing_fault = "fail_closed"')
    expect(toml).toContain('draft_max_tokens = 32')
    expect(toml).toContain('[defaults.request_defaults]')
    expect(toml).toContain('temperature = 0.55')
    expect(toml).toContain('reasoning_format = "deepseek-legacy"')
    expect(toml).toContain('[[models]]')
    expect(toml).toContain('ctx_size = 16384')
    expect(toml).not.toContain('[models.model_fit]')
    expect(toml).toContain('[models.hardware]')
    expect(toml).toContain('gpu_id = "cuda:0"')
    expect(toml).toContain('gpu_layers = -1')
    expect(toml).not.toContain('[defaults.speculative_decoding]')
    expect(toml).not.toContain('draft_selection_policy = "auto"')
    expect(toml).not.toContain('top_p = 0.95')
    expect(toml).not.toContain('reasoning_budget = 0')
    expect(toml).not.toContain('ctx = ')
    expect(toml).not.toContain('gpu_index =')
    expect(toml).not.toContain('[node]')
    expect(toml).not.toContain('fail_launch')
    expect(toml).not.toContain('chat_template =')
    expect(toml).not.toContain('model_runtime')
    expect(toml).not.toContain(`id = ${JSON.stringify(CONFIGURATION_HARNESS.assigns[0]?.id)}`)
  })

  it('omits empty telemetry headers while preserving non-empty header objects', () => {
    const telemetryDefaults: ConfigurationDefaultsHarnessData = {
      categories: [
        {
          id: 'telemetry',
          label: 'Telemetry',
          summary: 'Telemetry defaults.',
          help: 'Telemetry defaults',
          tomlSection: 'telemetry'
        }
      ],
      settings: [
        {
          id: 'telemetry.headers',
          categoryId: 'telemetry',
          canonicalPath: 'telemetry.headers',
          tomlSection: 'telemetry',
          tomlKey: 'headers',
          icon: 'gauge',
          label: 'Telemetry headers',
          description: 'Telemetry request headers.',
          inheritedLabel: 'Inherited by telemetry exporters',
          valueSchema: { kind: 'object' },
          control: {
            kind: 'text',
            name: 'headers',
            value: ''
          }
        }
      ],
      preview: []
    }

    const emptyToml = buildTOML([], [], [], {
      defaults: telemetryDefaults,
      defaultsValues: { 'telemetry.headers': '{}' }
    })
    const nonEmptyToml = buildTOML([], [], [], {
      defaults: telemetryDefaults,
      defaultsValues: { 'telemetry.headers': '{"Authorization":"Bearer test"}' }
    })

    expect(emptyToml).not.toContain('[telemetry]')
    expect(emptyToml).not.toContain('headers =')
    expect(nonEmptyToml).toContain('[telemetry]')
    expect(nonEmptyToml).toContain('headers = { Authorization = "Bearer test" }')
  })

  it('omits disabled draft speculative settings unless draft mode is selected and changed', () => {
    const draftToml = buildTOML([], [], [], {
      defaults: CONFIGURATION_DEFAULTS,
      defaultsValues: { 'speculation-mode': 'draft', 'draft-max-tokens': '32' }
    })
    const autoToml = buildTOML([], [], [], {
      defaults: CONFIGURATION_DEFAULTS,
      defaultsValues: { 'speculation-mode': 'auto', 'incompatible-pairing-behavior': 'fail_closed' }
    })
    expect(draftToml).toContain('[defaults.speculative]')
    expect(draftToml).toContain('mode = "draft"')
    expect(draftToml).toContain('draft_max_tokens = 32')
    expect(draftToml).not.toContain('pairing_fault = "warn_disable"')
    expect(autoToml).not.toContain('[defaults.speculative]')
    expect(autoToml).not.toContain('pairing_fault')
  })

  it('keeps preserve-existing disabled values, omits omit-when-disabled values, and serializes rejected disabled values for validation', () => {
    const defaults: ConfigurationDefaultsHarnessData = {
      categories: [
        {
          id: 'runtime',
          label: 'Runtime',
          summary: 'Runtime defaults.',
          help: 'Runtime defaults',
          tomlSection: 'defaults.hardware'
        },
        {
          id: 'speculative-decoding',
          label: 'Speculative',
          summary: 'Speculative defaults.',
          help: 'Speculative defaults'
        }
      ],
      settings: [
        {
          id: 'gpu-assignment',
          categoryId: 'runtime',
          canonicalPath: 'gpu.assignment',
          icon: 'cpu',
          label: 'GPU assignment',
          description: 'Runtime assignment mode.',
          inheritedLabel: 'Inherited by runtime defaults',
          valueSchema: { kind: 'enum', values: ['auto', 'pinned'] },
          control: {
            kind: 'choice',
            name: 'gpu_assignment',
            value: 'auto',
            options: [
              { value: 'auto', label: 'auto' },
              { value: 'pinned', label: 'pinned' }
            ]
          }
        },
        {
          id: 'pinned-device',
          categoryId: 'runtime',
          canonicalPath: 'defaults.hardware.device',
          tomlSection: 'defaults.hardware',
          icon: 'cpu',
          label: 'Pinned GPU device',
          description: 'Pinned runtime device.',
          inheritedLabel: 'Inherited by runtime defaults',
          valueSchema: { kind: 'string' },
          controlBehavior: {
            enable_when: [
              {
                path: { segments: ['gpu', 'assignment'] },
                operator: 'equals',
                values: [{ kind: 'string', value: 'pinned' }]
              }
            ],
            write_policy: 'preserve_existing'
          },
          control: {
            kind: 'text',
            name: 'device',
            value: 'cuda:0'
          },
          baselineValue: ''
        },
        {
          id: 'speculative-mode',
          categoryId: 'speculative-decoding',
          canonicalPath: 'defaults.speculative.mode',
          icon: 'brain',
          label: 'Speculative mode',
          description: 'Speculative runtime mode.',
          inheritedLabel: 'Inherited by speculative defaults',
          valueSchema: { kind: 'enum', values: ['draft', 'disabled'] },
          control: {
            kind: 'choice',
            name: 'mode',
            value: 'disabled',
            options: [
              { value: 'draft', label: 'draft' },
              { value: 'disabled', label: 'disabled' }
            ]
          },
          baselineValue: 'draft'
        },
        {
          id: 'draft-max-tokens',
          categoryId: 'speculative-decoding',
          canonicalPath: 'defaults.speculative.draft_max_tokens',
          icon: 'gauge',
          label: 'Draft max tokens',
          description: 'Draft token budget.',
          inheritedLabel: 'Inherited by speculative defaults',
          valueSchema: { kind: 'integer' },
          controlBehavior: {
            enable_when: [
              {
                path: { segments: ['defaults', 'speculative', 'mode'] },
                operator: 'equals',
                values: [{ kind: 'string', value: 'draft' }]
              }
            ]
          },
          control: {
            kind: 'text',
            name: 'draft_max_tokens',
            value: '16'
          },
          baselineValue: '0'
        },
        {
          id: 'rpc-backend',
          categoryId: 'runtime',
          canonicalPath: 'runtime.rpc_backend',
          icon: 'cpu',
          label: 'RPC backend',
          description: 'Unsupported backend setting.',
          inheritedLabel: 'Rejected in current runtime',
          valueSchema: { kind: 'string' },
          controlBehavior: {
            availability: {
              enabled: false,
              reason: 'External RPC backends are not supported.',
              source: 'static'
            },
            write_policy: 'reject_when_disabled'
          },
          control: {
            kind: 'text',
            name: 'rpc_backend',
            value: 'remote'
          },
          baselineValue: ''
        }
      ],
      preview: []
    }

    const toml = buildTOML([], [], [], {
      defaults,
      defaultsValues: { 'draft-max-tokens': '24' }
    })

    expect(toml).toContain('[defaults.hardware]')
    expect(toml).toContain('[defaults]')
    expect(toml).toContain('gpu_id = "cuda:0"')
    expect(toml).toContain('rpc_backend = "remote"')
    expect(toml).toContain('mode = "disabled"')
    expect(toml).not.toContain('draft_max_tokens = 24')
  })

  it('uses canonical inventory defaults when live defaults are hydrated into control values', () => {
    const liveHydratedDefaults = {
      ...CONFIGURATION_DEFAULTS,
      settings: CONFIGURATION_DEFAULTS.settings.map((setting) => {
        if (setting.id === 'activation-wire-dtype') {
          return {
            ...setting,
            baselineValue: setting.control.value,
            control: {
              ...setting.control,
              value: 'q8'
            }
          }
        }

        if (setting.id === 'image-min-tokens') {
          return {
            ...setting,
            baselineValue: setting.control.value,
            control: {
              ...setting.control,
              value: '64'
            }
          }
        }

        if (setting.id === 'server-alias') {
          return {
            ...setting,
            baselineValue: setting.control.value,
            control: {
              ...setting.control,
              value: 'carrack-mesh'
            }
          }
        }

        return setting
      })
    } satisfies ConfigurationDefaultsHarnessData

    const toml = buildTOML([], [], [], {
      defaults: liveHydratedDefaults,
      defaultsValues: {}
    })

    expect(toml).toContain('[defaults.skippy]')
    expect(toml).toContain('activation_wire_dtype = "q8"')
    expect(toml).toContain('[defaults.multimodal]')
    expect(toml).toContain('image_min_tokens = 64')
    expect(toml).toContain('[defaults.advanced.server]')
    expect(toml).toContain('alias = "carrack-mesh"')
    expect(toml).not.toContain('[defaults.request_defaults]')
  })

  it('emits optional default hardware device and numeric gpu layer sentinel', () => {
    const toml = buildTOML([], [], [], {
      defaults: CONFIGURATION_DEFAULTS,
      defaultsValues: { 'hardware-device': 'cuda:0', 'gpu-layers': '-1' }
    })

    expect(toml).toContain('[defaults]')
    expect(toml).toContain('gpu_id = "cuda:0"')
    expect(toml).toContain('[defaults.hardware]')
    expect(toml).toContain('gpu_layers = -1')
    expect(toml).not.toContain('gpu_layers = "-1"')
  })

  it('emits bool-or-auto choices as booleans while preserving auto as a string sentinel', () => {
    const continuousBatching = CONFIGURATION_DEFAULTS.settings.find((setting) => setting.id === 'continuous-batching')
    expect(continuousBatching).toBeDefined()

    const enabledToml = buildTOML([], [], [], {
      defaults: CONFIGURATION_DEFAULTS,
      defaultsValues: { 'continuous-batching': 'on' }
    })
    const disabledToml = buildTOML([], [], [], {
      defaults: CONFIGURATION_DEFAULTS,
      defaultsValues: { 'continuous-batching': 'off' }
    })
    const autoToml = buildTOML([], [], [], {
      defaults: CONFIGURATION_DEFAULTS,
      defaultsValues: { 'continuous-batching': 'auto' }
    })

    expect(enabledToml).toContain('continuous_batching = true')
    expect(enabledToml).not.toContain('continuous_batching = "on"')
    expect(disabledToml).toContain('continuous_batching = false')
    expect(disabledToml).not.toContain('continuous_batching = "off"')
    expect(autoToml).not.toContain('continuous_batching')
    expect(defaultSettingTomlScalar(continuousBatching!, 'auto')).toBe('"auto"')
  })

  it('quotes numeric-looking text defaults while keeping numeric controls unquoted', () => {
    const toml = buildTOML([], [], [], {
      defaults: CONFIGURATION_DEFAULTS,
      defaultsValues: { 'hardware-device': '0', 'gpu-layers': '-1' }
    })

    expect(toml).toContain('gpu_id = "0"')
    expect(toml).not.toContain('gpu_id = 0')
    expect(toml).toContain('gpu_layers = -1')
  })

  it('emits numeric-schema text controls as numeric TOML scalars', () => {
    const defaults: ConfigurationDefaultsHarnessData = {
      categories: [
        {
          id: 'runtime',
          label: 'Runtime',
          summary: 'Runtime defaults.',
          help: 'Runtime defaults',
          tomlSection: 'defaults.throughput'
        }
      ],
      settings: [
        {
          id: 'slot-prompt-similarity',
          categoryId: 'runtime',
          canonicalPath: 'defaults.throughput.slot_prompt_similarity',
          icon: 'cpu',
          label: 'Slot prompt similarity',
          description: 'Prompt similarity threshold.',
          inheritedLabel: '0',
          valueSchema: { kind: 'float' },
          control: {
            kind: 'text',
            name: 'slot_prompt_similarity',
            value: '0'
          },
          baselineValue: '0'
        },
        {
          id: 'hardware-device',
          categoryId: 'runtime',
          canonicalPath: 'defaults.hardware.device',
          icon: 'cpu',
          label: 'Default GPU device',
          description: 'Pinned GPU assignment.',
          inheritedLabel: '',
          valueSchema: { kind: 'string' },
          control: {
            kind: 'text',
            name: 'device',
            value: ''
          },
          baselineValue: ''
        },
        {
          id: 'flexible-number',
          categoryId: 'runtime',
          canonicalPath: 'defaults.throughput.flexible_number',
          icon: 'cpu',
          label: 'Flexible number',
          description: 'Integer or float threshold.',
          inheritedLabel: '0',
          valueSchema: { kind: 'one_of', variants: [{ kind: 'integer' }, { kind: 'float' }] },
          control: {
            kind: 'text',
            name: 'flexible_number',
            value: '0'
          },
          baselineValue: '0'
        }
      ],
      preview: []
    }

    const toml = buildTOML([], [], [], {
      defaults,
      defaultsValues: { 'slot-prompt-similarity': '0.5', 'hardware-device': '0', 'flexible-number': '0.5' }
    })

    expect(toml).toContain('[defaults.throughput]')
    expect(toml).toContain('slot_prompt_similarity = 0.5')
    expect(toml).not.toContain('slot_prompt_similarity = "0.5"')
    expect(toml).toContain('flexible_number = 0.5')
    expect(toml).not.toContain('flexible_number = "0.5"')
    expect(toml).toContain('[defaults]')
    expect(toml).toContain('gpu_id = "0"')
  })

  it('preserves dotted plugin names and plugin-owned dashed or dotted keys', () => {
    const defaults: ConfigurationDefaultsHarnessData = {
      categories: [
        {
          id: 'plugin:com.example.tool',
          label: 'Example Tool',
          summary: 'Example plugin settings.',
          help: 'Example plugin settings'
        }
      ],
      settings: [
        {
          id: 'plugin.com.example.tool.url-base',
          categoryId: 'plugin:com.example.tool',
          canonicalPath: 'plugin.com.example.tool.url-base',
          tomlSection: 'plugin.com.example.tool',
          tomlKey: 'url-base',
          icon: 'cog',
          label: 'URL Base',
          description: 'Plugin URL.',
          inheritedLabel: 'Plugin default',
          valueSchema: { kind: 'string' },
          control: { kind: 'text', name: 'url-base', value: '' },
          baselineValue: ''
        },
        {
          id: 'plugin.com.example.tool.settings.foo-bar',
          categoryId: 'plugin:com.example.tool',
          canonicalPath: 'plugin.com.example.tool.settings.foo-bar',
          tomlSection: 'plugin.com.example.tool.settings',
          tomlKey: 'foo-bar',
          icon: 'cog',
          label: 'Foo Bar',
          description: 'Plugin setting.',
          inheritedLabel: 'Plugin default',
          valueSchema: { kind: 'string' },
          control: { kind: 'text', name: 'foo-bar', value: '' },
          baselineValue: ''
        },
        {
          id: 'plugin.com.example.tool.settings.nested.key',
          categoryId: 'plugin:com.example.tool',
          canonicalPath: 'plugin.com.example.tool.settings.nested.key',
          tomlSection: 'plugin.com.example.tool.settings',
          tomlKey: 'nested.key',
          icon: 'cog',
          label: 'Nested Key',
          description: 'Plugin setting.',
          inheritedLabel: 'Plugin default',
          valueSchema: { kind: 'string' },
          control: { kind: 'text', name: 'nested.key', value: '' },
          baselineValue: ''
        }
      ],
      preview: []
    }

    const toml = buildTOML([], [], [], {
      defaults,
      defaultsValues: {
        'plugin.com.example.tool.url-base': 'http://localhost:8000/v1',
        'plugin.com.example.tool.settings.foo-bar': 'kept',
        'plugin.com.example.tool.settings.nested.key': 'literal'
      }
    })

    expect(toml).toContain('name = "com.example.tool"')
    expect(toml).toContain('url-base = "http://localhost:8000/v1"')
    expect(toml).toContain('foo-bar = "kept"')
    expect(toml).toContain('"nested.key" = "literal"')
    expect(toml).not.toContain('url_base')
    expect(toml).not.toContain('foo_bar')
  })

  it('quotes non-finite numeric defaults instead of emitting invalid TOML numbers', () => {
    const toml = buildTOML([], [], [], {
      defaults: CONFIGURATION_DEFAULTS,
      defaultsValues: { 'memory-margin': 'Infinity', temperature: 'NaN' }
    })

    expect(toml).toContain('safety_margin_gb = "Infinity"')
    expect(toml).toContain('temperature = "NaN"')
    expect(toml).not.toContain('safety_margin_gb = Infinity')
    expect(toml).not.toContain('temperature = NaN')
  })

  it('appends model placement lines to their configured sections', () => {
    const node: ConfigNode = {
      id: 'self',
      hostname: 'local',
      region: 'local',
      status: 'online',
      cpu: 'cpu',
      ramGB: 64,
      gpus: [{ idx: 0, name: 'RTX 5090', totalGB: 32 }],
      placement: 'separate'
    }
    const assign: ConfigAssign = {
      id: 'assign-1',
      modelId: 'hf://meshllm/model@main:Q4_K_M',
      nodeId: 'self',
      containerIdx: 0,
      ctx: 8192
    }

    const toml = buildTOML([node], [assign], [], {
      modelPlacementPaths: {
        model: 'models.<model-ref>.model',
        ctxSize: 'models.<model-ref>.hardware.ctx_size',
        device: 'models.<model-ref>.hardware.device',
        gpuLayers: 'models.<model-ref>.hardware.gpu_layers'
      }
    })

    expect(toml).toContain('[models.hardware]\nctx_size = 8192\ngpu_layers = -1')
    expect(toml).toContain('gpu_id = "cuda:0"')
  })

  it('preserves hidden per-model KV overrides while updating placement context', () => {
    const node: ConfigNode = {
      id: 'self',
      hostname: 'local',
      region: 'local',
      status: 'online',
      cpu: 'cpu',
      ramGB: 64,
      gpus: [{ idx: 0, name: 'Apple M4 Pro', totalGB: 37.4 }],
      placement: 'pooled'
    }
    const dupeModel: ConfigModel = {
      id: 'unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL',
      name: 'unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL',
      family: 'unsloth',
      paramsB: 4,
      quant: 'Q4_K_XL',
      sizeGB: 2.9,
      diskGB: 2.9,
      ctxMaxK: 256,
      moe: false,
      vision: false,
      tags: []
    }
    const otherModel: ConfigModel = {
      ...dupeModel,
      id: 'unsloth/qwen3.5-9b-gguf:UD-Q4_K_XL',
      name: 'unsloth/qwen3.5-9b-gguf:UD-Q4_K_XL',
      paramsB: 9,
      sizeGB: 6,
      diskGB: 6
    }
    const assigns: ConfigAssign[] = [
      { id: 'assign-1', modelId: dupeModel.id, nodeId: 'self', containerIdx: 0, ctx: 131072 },
      { id: 'assign-2', modelId: dupeModel.id, nodeId: 'self', containerIdx: 0, ctx: 262144 },
      { id: 'assign-3', modelId: otherModel.id, nodeId: 'self', containerIdx: 0, ctx: 65536 }
    ]

    const toml = buildTOML([node], assigns, [dupeModel, otherModel], {
      modelConfigEntries: [
        {
          model: dupeModel.name,
          model_fit: {
            ctx_size: 2048,
            cache_type_k: 'q8_0',
            cache_type_v: 'q4_0',
            kv_cache_policy: 'balanced'
          }
        },
        {
          model: dupeModel.name,
          model_fit: {
            ctx_size: 4096,
            cache_type_k: 'f16',
            cache_type_v: 'f16'
          }
        },
        {
          model: otherModel.name,
          model_fit: {
            ctx_size: 4096,
            cache_type_v: 'q8_0'
          }
        }
      ]
    })

    expect(toml.match(/\[\[models\]\]/g)).toHaveLength(3)
    expect(toml).toContain('ctx_size = 131072\ncache_type_k = "q8_0"\ncache_type_v = "q4_0"')
    expect(toml).toContain('[models.model_fit]\nkv_cache_policy = "balanced"')
    expect(toml).toContain('ctx_size = 262144\ncache_type_k = "f16"\ncache_type_v = "f16"')
    expect(toml).toContain('ctx_size = 65536\ncache_type_v = "q8_0"')
    expect(toml).not.toContain('[models.model_fit]\nctx_size')
  })

  it('serializes selected model custom configuration to nested TOML settings', () => {
    const node: ConfigNode = {
      id: 'self',
      hostname: 'local',
      region: 'local',
      status: 'online',
      cpu: 'cpu',
      ramGB: 64,
      gpus: [{ idx: 1, name: 'RTX 6000 Pro', totalGB: 48 }],
      placement: 'separate'
    }
    const model: ConfigModel = {
      id: 'llama70',
      name: 'Llama-3.3-70B-Q4_K_M',
      family: 'llama',
      paramsB: 70,
      quant: 'Q4_K_M',
      sizeGB: 40.3,
      diskGB: 40.3,
      ctxMaxK: 256,
      moe: false,
      vision: false,
      tags: []
    }
    const assign: ConfigAssign = {
      id: 'assign-llama',
      modelId: model.id,
      nodeId: node.id,
      containerIdx: 1,
      ctx: 16384,
      config: {
        slots: 4,
        batchProfile: 'throughput',
        splitMode: 'row',
        tensorSplit: '50,50',
        mmproj: '/models/mmproj.gguf',
        draftModelPath: '/models/draft.gguf',
        flashAttention: 'enabled',
        cacheTypeK: 'q8_0',
        cacheTypeV: 'q5_1'
      }
    }

    const toml = buildTOML([node], [assign], [model])

    expect(toml).toContain('ctx_size = 16384')
    expect(toml).toContain('parallel = 4')
    expect(toml).toContain('batch = 1024')
    expect(toml).toContain('ubatch = 256')
    expect(toml).toContain('[models.hardware]')
    expect(toml).toContain('gpu_id = "cuda:1"')
    expect(toml).toContain('split_mode = "row"')
    expect(toml).toContain('tensor_split = "50,50"')
    expect(toml).toContain('[models.multimodal]\nmmproj = "/models/mmproj.gguf"')
    expect(toml).toContain('[models.speculative]\ndraft_model = "/models/draft.gguf"')
    expect(toml).toContain('flash_attention = "enabled"')
    expect(toml).toContain('cache_type_k = "q8_0"')
    expect(toml).toContain('cache_type_v = "q5_1"')
  })

  it('deduplicates keys when explicit config overlaps with preserved model config entry', () => {
    const node: ConfigNode = {
      id: 'self',
      hostname: 'local',
      region: 'local',
      status: 'online',
      cpu: 'cpu',
      ramGB: 64,
      gpus: [{ idx: 0, name: 'RTX 4090', totalGB: 24 }],
      placement: 'pooled'
    }
    const model: ConfigModel = {
      id: 'qwen4',
      name: 'Qwen3.5-4B-Q4_K_XL',
      family: 'qwen3',
      paramsB: 4,
      quant: 'Q4_K_XL',
      sizeGB: 2.5,
      diskGB: 2.5,
      ctxMaxK: 256,
      moe: false,
      vision: false,
      tags: []
    }
    const assign: ConfigAssign = {
      id: 'assign-qwen',
      modelId: model.id,
      nodeId: node.id,
      containerIdx: 0,
      ctx: 262144,
      config: {
        slots: 4,
        flashAttention: 'enabled',
        cacheTypeK: 'q8_0'
      }
    }

    const toml = buildTOML([node], [assign], [model], {
      modelConfigEntries: [
        {
          model: model.name,
          model_fit: {
            cache_type_k: 'f16',
            cache_type_v: 'q4_0',
            flash_attention: 'disabled'
          },
          throughput: {
            parallel: 2
          }
        }
      ]
    })

    // Explicit config values win over preserved entry values
    expect(toml).toContain('parallel = 4')
    expect(toml).toContain('flash_attention = "enabled"')
    expect(toml).toContain('cache_type_k = "q8_0"')
    // Preserved-only values still appear
    expect(toml).toContain('cache_type_v = "q4_0"')
    // No duplicate keys — regression test for parallel duplication
    expect(toml.match(/^parallel = /gm)).toHaveLength(1)
    expect(toml.match(/^flash_attention = /gm)).toHaveLength(1)
    expect(toml.match(/^cache_type_k = /gm)).toHaveLength(1)
  })

  it('writes pinned GPU defaults only after a runtime GPU option is explicitly selected', () => {
    const defaults: ConfigurationDefaultsHarnessData = {
      categories: [
        {
          id: 'runtime',
          label: 'Runtime',
          summary: 'Runtime defaults.',
          help: 'Runtime defaults',
          tomlSection: 'defaults.hardware'
        }
      ],
      settings: [
        {
          id: 'gpu.assignment',
          categoryId: 'runtime',
          canonicalPath: 'gpu.assignment',
          tomlSection: 'gpu',
          icon: 'cpu',
          label: 'GPU assignment',
          description: 'Choose automatic or pinned assignment.',
          inheritedLabel: 'Written to GPU policy',
          valueSchema: { kind: 'enum', values: ['auto', 'pinned'] },
          control: {
            kind: 'choice',
            name: 'assignment',
            value: 'auto',
            options: [
              { value: 'auto', label: 'auto' },
              { value: 'pinned', label: 'pinned' }
            ]
          }
        },
        {
          id: 'defaults.hardware.device',
          categoryId: 'runtime',
          canonicalPath: 'defaults.hardware.device',
          tomlSection: 'defaults.hardware',
          icon: 'cpu',
          label: 'Default GPU device',
          description: 'Pinned GPU target.',
          inheritedLabel: 'Inherited by model entries',
          valueSchema: { kind: 'string' },
          control: { kind: 'choice', name: 'device', value: '', options: [{ value: 'MTL0', label: 'Apple GPU' }] }
        }
      ],
      preview: []
    }
    const toml = buildTOML([], [], [], {
      defaults,
      defaultsValues: { 'gpu.assignment': 'pinned', 'defaults.hardware.device': 'MTL0' }
    })

    expect(toml).toContain('[gpu]\nassignment = "pinned"')
    expect(toml).toContain('[defaults]\ngpu_id = "MTL0"')
  })
})
