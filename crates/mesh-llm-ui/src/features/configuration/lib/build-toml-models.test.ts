import { describe, expect, it } from 'vitest'
import { CONFIGURATION_DEFAULTS } from '@/features/app-tabs/data'
import type { ConfigAssign, ConfigModel, ConfigNode, ConfigurationDefaultsHarnessData } from '@/features/app-tabs/types'
import { buildTOML } from '@/features/configuration/lib/build-toml'

describe('buildTOML model and plugin serialization', () => {
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
