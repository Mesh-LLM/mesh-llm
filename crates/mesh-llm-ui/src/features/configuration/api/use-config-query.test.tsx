// @vitest-environment jsdom

import { act, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ReactNode } from 'react'

import { useConfigQuery } from '@/features/configuration/api/use-config-query'
import type {
  RuntimeConfigControlStatePayload,
  RuntimeConfigSchemaEntry,
  RuntimeConfigSchemaReference
} from '@/features/configuration/api/config-adapter'
import type { ModelsResponse, StatusPayload } from '@/lib/api/types'

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

const MODELS_RESPONSE: ModelsResponse = {
  mesh_models: [
    {
      name: 'Hermes-2-Pro-Mistral-7B-Q4_K_M',
      status: 'warm',
      node_count: 1,
      quantization: 'Q4_K_M'
    }
  ]
}

const CONFIG_SCHEMA_REFERENCE: RuntimeConfigSchemaReference = {
  settings: [
    schemaSetting('defaults.request_defaults.reasoning_format', { kind: 'string' }, 'Request Defaults', {
      control_hint: 'segmented',
      setting_order: 10
    }),
    schemaSetting(
      'defaults.request_defaults.top_k',
      { kind: 'integer' },
      'Request Defaults',
      {
        control_hint: 'range',
        setting_order: 20
      },
      [{ kind: 'range', min: '0', max: '100' }]
    ),
    schemaSetting('defaults.speculative.mode', { kind: 'string' }, 'Speculative Decoding', {
      control_hint: 'segmented',
      setting_order: 10
    }),
    schemaSetting('defaults.speculative.draft_max_tokens', { kind: 'integer' }, 'Speculative Decoding', {
      control_hint: 'range',
      setting_order: 20,
      unit: 'tokens'
    }),
    schemaSetting(
      'defaults.skippy.activation_wire_dtype',
      { kind: 'enum', values: ['f16', 'q8'] },
      'Skippy Transport',
      {
        control_hint: 'segmented',
        setting_order: 10
      }
    ),
    schemaSetting('defaults.multimodal.image_min_tokens', { kind: 'integer' }, 'Multimodal', {
      control_hint: 'range',
      setting_order: 10,
      unit: 'tokens'
    }),
    schemaSetting('defaults.advanced.server.alias', { kind: 'string' }, 'Advanced Server', {
      control_hint: 'text',
      setting_order: 10
    })
  ]
}

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('useConfigQuery', () => {
  it('waits for runtime-control hydration before exposing live schema defaults', async () => {
    const getConfigDeferred = createDeferredResponse({
      snapshot: {
        revision: 7,
        config: {
          version: 1,
          defaults: {
            request_defaults: {
              reasoning_format: 'qwen'
            }
          }
        }
      }
    })

    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)

      if (url.endsWith('/api/status')) return jsonResponse(STATUS_PAYLOAD)
      if (url.endsWith('/api/models')) return jsonResponse(MODELS_RESPONSE)
      if (url.endsWith('/api/runtime/control-bootstrap')) {
        return jsonResponse({
          enabled: true,
          local_only: true,
          requires_explicit_remote_endpoint: false,
          endpoint: 'control://owner'
        })
      }
      if (url.endsWith('/api/runtime/config-schema')) return jsonResponse(CONFIG_SCHEMA_REFERENCE)
      if (url.endsWith('/api/runtime/config-control-state')) return jsonResponse({ settings: {} })
      if (url.endsWith('/api/runtime/control/get-config')) {
        expect(JSON.parse(String(init?.body))).toEqual({ endpoint: 'control://owner' })
        return getConfigDeferred.promise
      }

      throw new Error(`Unexpected fetch request: ${url}`)
    })
    vi.stubGlobal('fetch', fetchMock)

    const { result } = renderHook(() => useConfigQuery({ enabled: true }), {
      wrapper: createWrapper()
    })

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/runtime/control/get-config',
        expect.objectContaining({ method: 'POST' })
      )
    )
    expect(result.current.data).toBeUndefined()

    getConfigDeferred.resolve(jsonResponse(getConfigDeferred.body))

    await waitFor(() =>
      expect(readSettingValue(result.current.data!, 'defaults.request_defaults.reasoning_format')).toBe('qwen')
    )
    expect(fetchMock).toHaveBeenCalledWith('/api/runtime/control-bootstrap')
    expect(fetchMock).toHaveBeenCalledWith('/api/runtime/config-control-state')
  })

  it('uses schema defaults and empty control-state without fetching a config snapshot when bootstrap is disabled', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)

      if (url.endsWith('/api/status')) return jsonResponse(STATUS_PAYLOAD)
      if (url.endsWith('/api/models')) return jsonResponse(MODELS_RESPONSE)
      if (url.endsWith('/api/runtime/control-bootstrap')) {
        return jsonResponse({
          enabled: false,
          local_only: true,
          requires_explicit_remote_endpoint: true,
          disabled_reason: 'missing_owner_identity',
          message: 'Configuration saving requires a local owner identity.',
          suggested_commands: [
            'mesh-llm auth status',
            'mesh-llm auth init --no-passphrase',
            'mesh-llm serve --owner-required'
          ]
        })
      }
      if (url.endsWith('/api/runtime/config-schema')) return jsonResponse(CONFIG_SCHEMA_REFERENCE)
      if (url.endsWith('/api/runtime/config-control-state')) return jsonResponse({})

      throw new Error(`Unexpected fetch request: ${url}`)
    })
    vi.stubGlobal('fetch', fetchMock)

    const { result } = renderHook(() => useConfigQuery({ enabled: true }), {
      wrapper: createWrapper()
    })

    await waitFor(() => expect(result.current.data).toBeDefined())

    expect(readSettingValue(result.current.data!, 'defaults.request_defaults.reasoning_format')).toBe('')
    expect(fetchMock).not.toHaveBeenCalledWith(
      '/api/runtime/control/get-config',
      expect.objectContaining({ method: 'POST' })
    )
    expect(result.current.isError).toBe(false)
    expect(result.current.controlConfigQuery.data?.bootstrap).toMatchObject({
      enabled: false,
      disabled_reason: 'missing_owner_identity',
      suggested_commands: expect.arrayContaining(['mesh-llm auth init --no-passphrase'])
    })
    expect(result.current.controlConfigQuery.data?.snapshot).toBeUndefined()
    expect(result.current.controlConfigQuery.data?.controlState).toEqual({ settings: {} })
  })

  it('exposes runtime control-state overlay entries with schema defaults', async () => {
    const deviceSetting = schemaSetting(
      'defaults.hardware.device',
      { kind: 'string' },
      'Runtime',
      {
        control_hint: 'select',
        setting_order: 10
      },
      []
    )
    const schema: RuntimeConfigSchemaReference = {
      settings: [
        {
          ...deviceSetting,
          control_behavior: {
            options_source: 'runtime_gpus',
            write_policy: 'preserve_existing'
          }
        }
      ]
    }
    const controlState = {
      settings: {
        'defaults.hardware.device': {
          enabled: true,
          source: 'runtime',
          write_policy: 'preserve_existing',
          options: [
            {
              value: { kind: 'string', value: 'metal:0' },
              label: 'Apple GPU (metal:0)',
              disabled: false,
              source: 'runtime_gpus'
            }
          ]
        }
      }
    } satisfies RuntimeConfigControlStatePayload
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)

      if (url.endsWith('/api/status')) return jsonResponse(STATUS_PAYLOAD)
      if (url.endsWith('/api/models')) return jsonResponse(MODELS_RESPONSE)
      if (url.endsWith('/api/runtime/control-bootstrap')) {
        return jsonResponse({
          enabled: false,
          local_only: true,
          requires_explicit_remote_endpoint: true
        })
      }
      if (url.endsWith('/api/runtime/config-schema')) return jsonResponse(schema)
      if (url.endsWith('/api/runtime/config-control-state')) return jsonResponse(controlState)

      throw new Error(`Unexpected fetch request: ${url}`)
    })
    vi.stubGlobal('fetch', fetchMock)

    const { result } = renderHook(() => useConfigQuery({ enabled: true }), {
      wrapper: createWrapper()
    })

    await waitFor(() => expect(result.current.data?.modelSettings?.settings[0]?.control.value).toBe(''))
    expect(result.current.data?.modelSettings?.settings[0]?.control).toMatchObject({
      kind: 'choice',
      presentation: 'select',
      options: [
        { value: '', label: 'Select GPU' },
        { value: 'metal:0', label: 'Apple GPU (metal:0)' }
      ]
    })

    expect(result.current.data?.modelSettings?.settings[0]?.controlState).toEqual(
      controlState.settings['defaults.hardware.device']
    )
    expect(result.current.controlConfigQuery.data?.controlState).toEqual(controlState)
  })

  it('applies full mesh config updates with expected revision and preserved fields', async () => {
    let applySucceeded = false
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)

      if (url.endsWith('/api/status')) return jsonResponse(STATUS_PAYLOAD)
      if (url.endsWith('/api/models')) return jsonResponse(MODELS_RESPONSE)
      if (url.endsWith('/api/runtime/control-bootstrap')) {
        return jsonResponse({
          enabled: true,
          local_only: true,
          requires_explicit_remote_endpoint: false,
          endpoint: 'control://owner'
        })
      }
      if (url.endsWith('/api/runtime/config-schema')) return jsonResponse(CONFIG_SCHEMA_REFERENCE)
      if (url.endsWith('/api/runtime/config-control-state')) return jsonResponse({ settings: {} })
      if (url.endsWith('/api/runtime/control/get-config')) {
        return jsonResponse({
          snapshot: {
            revision: applySucceeded ? 8 : 7,
            config: {
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
                threads: 6,
                request_defaults: {
                  temperature: 0.8,
                  reasoning_format: applySucceeded ? 'qwen' : 'deepseek',
                  top_k: applySucceeded ? 55 : 40
                },
                speculative: applySucceeded ? { mode: 'draft', draft_max_tokens: 20 } : { draft_max_tokens: 16 },
                skippy: {
                  activation_wire_dtype: applySucceeded ? 'q8' : 'f16'
                },
                multimodal: {
                  image_min_tokens: applySucceeded ? 64 : 32
                },
                advanced: {
                  server: {
                    alias: applySucceeded ? 'carrack-mesh' : 'existing-alias'
                  }
                }
              }
            }
          }
        })
      }
      if (url.endsWith('/api/runtime/control/apply-config')) {
        const body = JSON.parse(String(init?.body)) as {
          endpoint: string
          expected_revision: number
          config: Record<string, unknown>
        }

        expect(body.endpoint).toBe('control://owner')
        expect(body.expected_revision).toBe(7)
        expect(body.config).toMatchObject({
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
            threads: 6,
            request_defaults: {
              temperature: 0.8,
              reasoning_format: 'qwen',
              top_k: 55
            },
            speculative: {
              mode: 'draft',
              draft_max_tokens: 20
            },
            skippy: {
              activation_wire_dtype: 'q8'
            },
            multimodal: {
              image_min_tokens: 64
            },
            advanced: {
              server: {
                alias: 'carrack-mesh'
              }
            }
          }
        })

        applySucceeded = true
        return jsonResponse({
          success: true,
          current_revision: 8,
          config_hash: 'abc123',
          apply_mode: 'live'
        })
      }

      throw new Error(`Unexpected fetch request: ${url}`)
    })
    vi.stubGlobal('fetch', fetchMock)

    const { result } = renderHook(() => useConfigQuery({ enabled: true }), {
      wrapper: createWrapper()
    })

    await waitFor(() =>
      expect(readSettingValue(result.current.data!, 'defaults.request_defaults.reasoning_format')).toBe('deepseek')
    )

    const nextDefaults = readDefaultsValues(result.current.data!)
    nextDefaults['defaults.request_defaults.reasoning_format'] = 'qwen'
    nextDefaults['defaults.request_defaults.top_k'] = '55'
    nextDefaults['defaults.speculative.mode'] = 'draft'
    nextDefaults['defaults.speculative.draft_max_tokens'] = '20'
    nextDefaults['defaults.skippy.activation_wire_dtype'] = 'q8'
    nextDefaults['defaults.multimodal.image_min_tokens'] = '64'
    nextDefaults['defaults.advanced.server.alias'] = 'carrack-mesh'

    await act(async () => {
      const response = await result.current.applyDefaults({
        values: nextDefaults,
        nodes: result.current.data!.nodes,
        assigns: result.current.data!.assigns,
        catalog: result.current.data!.catalog,
        modelPlacementPaths: result.current.data!.modelPlacementPaths
      })
      expect(response).toMatchObject({
        success: true,
        current_revision: 8,
        apply_mode: 'live'
      })
    })

    await waitFor(() =>
      expect(readSettingValue(result.current.data!, 'defaults.request_defaults.reasoning_format')).toBe('qwen')
    )
    expect(readSettingValue(result.current.data!, 'defaults.request_defaults.top_k')).toBe('55')
    expect(readSettingValue(result.current.data!, 'defaults.skippy.activation_wire_dtype')).toBe('q8')
    expect(readSettingValue(result.current.data!, 'defaults.multimodal.image_min_tokens')).toBe('64')
    expect(readSettingValue(result.current.data!, 'defaults.advanced.server.alias')).toBe('carrack-mesh')
    expect(result.current.controlConfigQuery.data?.snapshot?.revision).toBe(8)
  })

  it('rehydrates runtime control-state after a successful apply', async () => {
    let controlStateFetches = 0
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)

      if (url.endsWith('/api/status')) return jsonResponse(STATUS_PAYLOAD)
      if (url.endsWith('/api/models')) return jsonResponse(MODELS_RESPONSE)
      if (url.endsWith('/api/runtime/control-bootstrap')) {
        return jsonResponse({
          enabled: true,
          local_only: true,
          requires_explicit_remote_endpoint: false,
          endpoint: 'control://owner'
        })
      }
      if (url.endsWith('/api/runtime/config-schema')) return jsonResponse(CONFIG_SCHEMA_REFERENCE)
      if (url.endsWith('/api/runtime/config-control-state')) {
        controlStateFetches += 1
        return jsonResponse({
          settings: {
            'defaults.request_defaults.reasoning_format': {
              enabled: controlStateFetches > 1,
              source: 'runtime',
              write_policy: 'preserve_existing',
              reason: controlStateFetches > 1 ? undefined : 'Refreshing overlay'
            }
          }
        })
      }
      if (url.endsWith('/api/runtime/control/get-config')) {
        expect(JSON.parse(String(init?.body))).toEqual({ endpoint: 'control://owner' })
        return jsonResponse({
          snapshot: {
            revision: 7,
            config: {
              version: 1,
              defaults: {
                request_defaults: {
                  reasoning_format: 'deepseek'
                }
              }
            }
          }
        })
      }
      if (url.endsWith('/api/runtime/control/apply-config')) {
        return jsonResponse({
          success: true,
          current_revision: 8,
          config_hash: 'abc123',
          apply_mode: 'live'
        })
      }

      throw new Error(`Unexpected fetch request: ${url}`)
    })
    vi.stubGlobal('fetch', fetchMock)

    const { result } = renderHook(() => useConfigQuery({ enabled: true }), {
      wrapper: createWrapper()
    })

    await waitFor(() => expect(result.current.data).toBeDefined())
    expect(
      result.current.controlConfigQuery.data?.controlState?.settings?.['defaults.request_defaults.reasoning_format']
    ).toMatchObject({
      enabled: false,
      reason: 'Refreshing overlay'
    })

    const nextDefaults = readDefaultsValues(result.current.data!)
    nextDefaults['defaults.request_defaults.reasoning_format'] = 'qwen'

    await act(async () => {
      const response = await result.current.applyDefaults({
        values: nextDefaults,
        nodes: result.current.data!.nodes,
        assigns: result.current.data!.assigns,
        catalog: result.current.data!.catalog,
        modelPlacementPaths: result.current.data!.modelPlacementPaths
      })
      expect(response).toMatchObject({ success: true, current_revision: 8 })
    })

    await waitFor(() =>
      expect(
        result.current.controlConfigQuery.data?.controlState?.settings?.['defaults.request_defaults.reasoning_format']
      ).toMatchObject({
        enabled: true,
        source: 'runtime'
      })
    )
    expect(controlStateFetches).toBeGreaterThanOrEqual(2)
  })

  it('does not replace the cached runtime-control snapshot when apply returns success false', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)

      if (url.endsWith('/api/status')) return jsonResponse(STATUS_PAYLOAD)
      if (url.endsWith('/api/models')) return jsonResponse(MODELS_RESPONSE)
      if (url.endsWith('/api/runtime/control-bootstrap')) {
        return jsonResponse({
          enabled: true,
          local_only: true,
          requires_explicit_remote_endpoint: false,
          endpoint: 'control://owner'
        })
      }
      if (url.endsWith('/api/runtime/config-schema')) return jsonResponse(CONFIG_SCHEMA_REFERENCE)
      if (url.endsWith('/api/runtime/config-control-state')) return jsonResponse({ settings: {} })
      if (url.endsWith('/api/runtime/control/get-config')) {
        return jsonResponse({
          snapshot: {
            revision: 7,
            config: {
              version: 1,
              defaults: {
                request_defaults: {
                  reasoning_format: 'deepseek'
                }
              }
            }
          }
        })
      }
      if (url.endsWith('/api/runtime/control/apply-config')) {
        const body = JSON.parse(String(init?.body)) as {
          expected_revision: number
          config: { defaults?: { request_defaults?: { reasoning_format?: string } } }
        }

        expect(body.expected_revision).toBe(7)
        expect(body.config.defaults?.request_defaults?.reasoning_format).toBe('qwen')

        return jsonResponse({
          success: false,
          current_revision: 8,
          config_hash: 'rejected',
          apply_mode: 'unspecified',
          error: { code: 'validation_error', message: 'invalid config' }
        })
      }

      throw new Error(`Unexpected fetch request: ${url}`)
    })
    vi.stubGlobal('fetch', fetchMock)

    const { result } = renderHook(() => useConfigQuery({ enabled: true }), {
      wrapper: createWrapper()
    })

    await waitFor(() =>
      expect(readSettingValue(result.current.data!, 'defaults.request_defaults.reasoning_format')).toBe('deepseek')
    )

    const nextDefaults = readDefaultsValues(result.current.data!)
    nextDefaults['defaults.request_defaults.reasoning_format'] = 'qwen'

    await act(async () => {
      const response = await result.current.applyDefaults({
        values: nextDefaults,
        nodes: result.current.data!.nodes,
        assigns: result.current.data!.assigns,
        catalog: result.current.data!.catalog,
        modelPlacementPaths: result.current.data!.modelPlacementPaths
      })
      expect(response).toMatchObject({
        success: false,
        current_revision: 8
      })
    })

    expect(readSettingValue(result.current.data!, 'defaults.request_defaults.reasoning_format')).toBe('deepseek')
    expect(result.current.controlConfigQuery.data?.snapshot?.revision).toBe(7)
  })
})

function schemaSetting(
  canonicalPath: string,
  valueSchema: RuntimeConfigSchemaEntry['value_schema'],
  categoryLabel: string,
  presentation: NonNullable<RuntimeConfigSchemaEntry['presentation']>,
  constraints: RuntimeConfigSchemaEntry['constraints'] = []
): RuntimeConfigSchemaEntry {
  const categoryId = categoryLabel.toLowerCase().replaceAll(' ', '-')
  const label = canonicalPath
    .split('.')
    .at(-1)!
    .replaceAll('_', ' ')
    .replace(/\b\w/g, (match) => match.toUpperCase())

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
    constraints,
    presentation: {
      label,
      help: `${label} default.`,
      category_id: categoryId,
      category_label: categoryLabel,
      category_summary: `${categoryLabel} defaults`,
      category_order: 10,
      ...presentation
    }
  }
}

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false
      }
    }
  })

  return function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  }
}

function readSettingValue(data: NonNullable<ReturnType<typeof useConfigQuery>['data']>, settingId: string) {
  return data.defaults.settings.find((setting) => setting.id === settingId)?.control.value
}

function readDefaultsValues(data: NonNullable<ReturnType<typeof useConfigQuery>['data']>) {
  return Object.fromEntries(data.defaults.settings.map((setting) => [setting.id, setting.control.value]))
}

function jsonResponse(body: unknown) {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'Content-Type': 'application/json' }
  })
}

function createDeferredResponse(body: unknown) {
  let resolve: (response: Response) => void = () => undefined
  const promise = new Promise<Response>((promiseResolve) => {
    resolve = promiseResolve
  })

  return {
    body,
    promise,
    resolve
  }
}
