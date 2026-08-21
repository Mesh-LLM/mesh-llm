import { describe, expect, it } from 'vitest'
import {
  createConfigurationDefaultsFromSchema,
  formatConfigDiagnostics,
  mergeConfigurationDefaultsIntoMeshConfig,
  mergeConfigurationIntoMeshConfig,
  runtimeControlApplyErrorMessage
} from '@/features/configuration/api/config-adapter'
import { SCHEMA_REFERENCE, schemaSetting } from './config-adapter-test-support'
import type { RuntimeConfigSchemaReference, RuntimeControlMeshConfig } from './config-adapter-types'

describe('configuration model merge and diagnostics', () => {
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

  it('uses the canonical path when a diagnostic omits its display path', () => {
    const result = formatConfigDiagnostics([
      {
        code: 'invalid_value',
        severity: 'error',
        source: 'validation',
        canonical_path: 'logging.retention_ttl_secs',
        message: 'retention must be positive'
      }
    ])

    expect(result).toContain('**`logging.retention_ttl_secs`** · `ERROR`')
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
