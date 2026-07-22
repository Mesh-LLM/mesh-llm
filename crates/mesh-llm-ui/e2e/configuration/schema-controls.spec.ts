import AxeBuilder from '@axe-core/playwright'
import { expect, test, type Page, type TestInfo } from '../fixtures/base'
import { DATA_MODE_STORAGE_KEY } from '@e2e/support/data-mode'

const FEATURE_FLAGS_STORAGE_KEY = 'mesh-llm-ui-preview:feature-flags:v1'

type JsonRecord = Record<string, unknown>

const statusPayload = {
  node_id: 'config-e2e-node',
  node_state: 'serving',
  model_name: 'Qwen3.5-2B-GGUF',
  peers: [],
  models: ['Qwen3.5-2B-GGUF'],
  my_vram_gb: 24,
  gpus: [{ name: 'NVIDIA RTX 4090', total_vram_gb: 24, free_vram_gb: 18 }],
  serving_models: ['Qwen3.5-2B-GGUF'],
  hostname: 'config-e2e',
  api_port: 9337,
  token: 'config-e2e-token',
  llama_ready: true
}

const modelsPayload = {
  mesh_models: [
    {
      name: 'Qwen3.5-2B-GGUF',
      status: 'warm',
      size_gb: 1.3,
      node_count: 1,
      capabilities: { vision: false, moe: false },
      quantization: 'Q4_K_M',
      context_length: 32768,
      family: 'qwen',
      tags: ['chat'],
      params_b: 2,
      disk_gb: 1.3
    }
  ]
}

const bootstrapPayload = {
  enabled: true,
  local_only: true,
  requires_explicit_remote_endpoint: false,
  endpoint: 'local-owner'
}

const schemaPayload = {
  plugin_instances: [
    {
      name: 'blackboard',
      enabled: true,
      source_repository: 'local-test',
      installed_version: '1.0.0',
      has_config_schema: true,
      allow_unvalidated_config: true
    }
  ],
  settings: [
    setting('runtime.native_backend', {
      label: 'Native backend',
      help: 'Runtime backend selected from local availability.',
      category_id: 'runtime-policy',
      category_label: 'Runtime Policy',
      value_schema: { kind: 'enum', values: ['metal', 'vulkan'] },
      control_behavior: { options_source: 'runtime_native_backends' },
      apply_mode: 'dynamic_apply',
      restart_scope: 'none'
    }),
    setting('gpu.assignment', {
      label: 'GPU assignment',
      help: 'Choose automatic or pinned GPU assignment.',
      category_id: 'runtime',
      category_label: 'Runtime',
      value_schema: { kind: 'enum', values: ['auto', 'pinned'] },
      control_hint: 'toggle'
    }),
    setting('defaults.model_fit.ctx_size', {
      label: 'Context size',
      help: 'Maximum context tokens inherited by model placements.',
      category_id: 'memory',
      category_label: 'Memory',
      unit: 'tokens',
      renderer_id: 'context-slider',
      value_schema: { kind: 'integer' },
      control_behavior: { numeric: { min: 512, max: 32768, step: 512, unit: 'tokens' } }
    }),
    setting('defaults.model_fit.kv_cache_policy', {
      label: 'KV cache policy',
      help: 'Select the KV cache profile used for memory planning.',
      category_id: 'memory',
      category_label: 'Memory',
      renderer_id: 'kv-cache-policy',
      value_schema: { kind: 'enum', values: ['auto', 'quality', 'balanced', 'saver'] }
    }),
    setting('defaults.hardware.device', {
      label: 'Pinned GPU device',
      help: 'Only editable when GPU assignment is pinned.',
      category_id: 'runtime',
      category_label: 'Runtime',
      placeholder: 'cuda:0',
      value_schema: { kind: 'string' },
      control_behavior: {
        enable_when: [
          {
            path: { segments: ['gpu', 'assignment'] },
            operator: 'equals',
            values: [{ kind: 'string', value: 'pinned' }]
          }
        ],
        write_policy: 'omit_when_disabled'
      }
    }),
    setting('defaults.speculative.mode', {
      label: 'Speculative mode',
      help: 'Select draft, n-gram, or disabled speculative decoding.',
      category_id: 'speculative-decoding',
      category_label: 'Speculative Decoding',
      value_schema: { kind: 'enum', values: ['disabled', 'draft', 'ngram'] }
    }),
    setting('defaults.speculative.draft_min_tokens', {
      label: 'Draft min tokens',
      help: 'Minimum tokens requested from the speculative draft.',
      category_id: 'speculative-decoding',
      category_label: 'Speculative Decoding',
      value_schema: { kind: 'integer' },
      control_behavior: {
        numeric: { min: 1, max: 16, step: 1, unit: 'tokens' },
        enable_when: [
          {
            path: { segments: ['defaults', 'speculative', 'mode'] },
            operator: 'equals',
            values: [{ kind: 'string', value: 'draft' }]
          }
        ],
        write_policy: 'omit_when_disabled'
      }
    }),
    setting('defaults.multimodal.mmproj_path', {
      label: 'Multimodal projector',
      help: 'Local projector GGUF path.',
      category_id: 'multimodal',
      category_label: 'Multimodal',
      placeholder: './models/mmproj.gguf',
      value_schema: { kind: 'path' },
      control_behavior: { text_format: 'path' }
    }),
    setting('defaults.multimodal.mmproj_url', {
      label: 'Projector URL',
      help: 'Optional remote projector artifact URL.',
      category_id: 'multimodal',
      category_label: 'Multimodal',
      placeholder: 'https://example.com/mmproj.gguf',
      value_schema: { kind: 'url' },
      control_behavior: { text_format: 'url' }
    }),
    setting('owner_control.bind', {
      label: 'Owner-control bind',
      help: 'Local bind address for owner-control.',
      category_id: 'network',
      category_label: 'Network',
      value_schema: { kind: 'socket_addr' },
      control_behavior: { text_format: 'socket_addr' }
    }),
    setting('owner_control.advertise_addr', {
      label: 'Advertised control address',
      help: 'Advertised owner-control endpoint.',
      category_id: 'network',
      category_label: 'Network',
      value_schema: { kind: 'socket_addr' },
      control_behavior: {
        text_format: 'socket_addr',
        enable_when: [
          {
            path: { segments: ['owner_control', 'bind'] },
            operator: 'present'
          }
        ],
        write_policy: 'omit_when_disabled'
      }
    }),
    setting('plugin.blackboard.settings.endpoint', {
      owner: 'plugin',
      source: { kind: 'plugin', plugin_name: 'blackboard', allow_unvalidated_config: true },
      label: 'Blackboard endpoint',
      help: 'HTTP endpoint used by the blackboard plugin.',
      category_id: 'plugin:blackboard',
      category_label: 'Blackboard',
      placeholder: 'https://blackboard.local/api',
      value_schema: { kind: 'url' },
      control_behavior: { text_format: 'url' }
    })
  ]
}

const controlStatePayload = {
  settings: {
    'runtime.native_backend': {
      enabled: true,
      source: 'runtime',
      write_policy: 'preserve_existing',
      options: [
        {
          value: { kind: 'string', value: 'metal' },
          label: 'Metal',
          note: 'Available on this host',
          disabled: false,
          source: 'runtime_native_backends'
        },
        {
          value: { kind: 'string', value: 'vulkan' },
          label: 'Vulkan',
          reason: 'No Vulkan runtime was detected',
          disabled: true,
          source: 'runtime_native_backends'
        }
      ]
    },
    'defaults.multimodal.mmproj_path': {
      enabled: false,
      source: 'runtime',
      reason: 'No local projector file was detected.',
      note: 'The existing value is preserved on save.',
      write_policy: 'preserve_existing'
    }
  }
}

const initialConfig = {
  defaults: {
    model_fit: { ctx_size: 4096, kv_cache_policy: 'balanced' },
    hardware: { device: 'cuda:0' },
    speculative: { mode: 'disabled', draft_min_tokens: 4 },
    multimodal: { mmproj_path: './existing/mmproj.gguf', mmproj_url: 'https://example.com/mmproj.gguf' }
  },
  gpu: { assignment: 'auto' },
  runtime: { native_backend: 'metal' },
  owner_control: {},
  plugin: [
    {
      name: 'blackboard',
      enabled: true,
      settings: { endpoint: 'https://blackboard.local/api' }
    }
  ],
  models: [{ model: 'Qwen3.5-2B-GGUF', model_fit: { ctx_size: 4096 } }]
}

function setting(
  canonicalPath: string,
  options: {
    owner?: 'built_in' | 'engine' | 'plugin'
    source?: JsonRecord
    label: string
    help: string
    category_id: string
    category_label: string
    value_schema: JsonRecord
    control_behavior?: JsonRecord
    control_hint?: string
    placeholder?: string
    renderer_id?: string
    unit?: string
    apply_mode?: string
    restart_scope?: string
  }
) {
  return {
    canonical_path: canonicalPath,
    owner: options.owner ?? 'built_in',
    source: options.source ?? { kind: 'built_in' },
    value_schema: options.value_schema,
    support: 'supported',
    control_surfaces: ['config_file'],
    apply_mode: options.apply_mode ?? 'static_on_load',
    restart_scope: options.restart_scope ?? 'process_restart',
    visibility: 'user',
    description: options.help,
    presentation: {
      label: options.label,
      help: options.help,
      category_id: options.category_id,
      category_label: options.category_label,
      category_summary: `${options.category_label} settings`,
      control_hint: options.control_hint,
      placeholder: options.placeholder,
      renderer_id: options.renderer_id,
      unit: options.unit
    },
    control_behavior: options.control_behavior
  }
}

async function installConfigurationBackend(page: Page) {
  await page.addInitScript(
    ({ dataModeKey, featureFlagsKey, status, models, bootstrap, schema, controlState, config }) => {
      window.localStorage.setItem(dataModeKey, 'live')
      window.localStorage.setItem(
        featureFlagsKey,
        JSON.stringify({
          global: { newConfigurationPage: true },
          configuration: { integrations: true }
        })
      )

      const state = {
        revision: 7,
        config: structuredClone(config),
        applyRequests: [] as unknown[],
        validateRequests: [] as unknown[]
      }

      Object.defineProperty(window, '__meshConfigE2E', {
        configurable: true,
        value: state
      })

      const jsonResponse = (payload: unknown) =>
        new Response(JSON.stringify(payload), { headers: { 'Content-Type': 'application/json' } })

      class MockEventSource extends EventTarget {
        static readonly CONNECTING = 0
        static readonly OPEN = 1
        static readonly CLOSED = 2
        readonly CONNECTING = 0
        readonly OPEN = 1
        readonly CLOSED = 2
        readonly url: string
        readyState = MockEventSource.OPEN
        onerror: ((event: Event) => void) | null = null
        onmessage: ((event: MessageEvent) => void) | null = null
        onopen: ((event: Event) => void) | null = null

        constructor(url: string | URL) {
          super()
          this.url = String(url)
          queueMicrotask(() => this.onopen?.(new Event('open')))
        }

        close() {
          this.readyState = MockEventSource.CLOSED
        }
      }

      window.EventSource = MockEventSource as typeof EventSource

      const originalFetch = window.fetch.bind(window)

      window.fetch = async (input, init) => {
        const request = input instanceof Request ? input : new Request(input, init)
        const url = new URL(request.url, window.location.href)

        if (url.pathname === '/api/status') return jsonResponse(status)
        if (url.pathname === '/api/models') return jsonResponse(models)
        if (url.pathname === '/api/runtime/control-bootstrap') return jsonResponse(bootstrap)
        if (url.pathname === '/api/runtime/config-schema') return jsonResponse(schema)
        if (url.pathname === '/api/runtime/config-control-state') return jsonResponse(controlState)

        if (url.pathname === '/api/runtime/control/get-config') {
          return jsonResponse({ snapshot: { revision: state.revision, config: state.config } })
        }

        if (url.pathname === '/api/runtime/config/validate') {
          const body = await request.json().catch(() => ({}))
          state.validateRequests.push(body)
          const toml = String(body.toml ?? '')
          const ok = !toml.includes('advertise_addr = "127.0.0.1:3131"')
          return jsonResponse({
            ok,
            path: body.path,
            diagnostics: ok
              ? []
              : [
                  {
                    code: 'requires',
                    severity: 'error',
                    source: 'validator',
                    path: 'owner_control.advertise_addr',
                    canonical_path: 'owner_control.advertise_addr',
                    message: 'owner_control.advertise_addr requires owner_control.bind'
                  }
                ]
          })
        }

        if (url.pathname === '/api/runtime/control/apply-config') {
          const body = await request.json()
          state.applyRequests.push(body)
          state.revision += 1
          state.config = structuredClone(body.config)
          return jsonResponse({
            success: true,
            current_revision: state.revision,
            config_hash: 'config-e2e-hash',
            apply_mode: 'restart_required',
            diagnostics: []
          })
        }

        return originalFetch(input, init)
      }
    },
    {
      dataModeKey: DATA_MODE_STORAGE_KEY,
      featureFlagsKey: FEATURE_FLAGS_STORAGE_KEY,
      status: statusPayload,
      models: modelsPayload,
      bootstrap: bootstrapPayload,
      schema: schemaPayload,
      controlState: controlStatePayload,
      config: initialConfig
    }
  )
}

function appUrl(pathname: string, testInfo: TestInfo) {
  const baseURL = String(testInfo.project.use.baseURL ?? 'http://127.0.0.1:51973')
  return new URL(pathname, baseURL).toString()
}

async function readConfigState(page: Page) {
  return page.evaluate(() => {
    const state = (window as unknown as { __meshConfigE2E: { applyRequests: unknown[]; validateRequests: unknown[] } })
      .__meshConfigE2E
    return {
      applyRequests: state.applyRequests,
      validateRequests: state.validateRequests
    }
  })
}

async function expectNoHorizontalOverflow(page: Page) {
  await expect
    .poll(async () =>
      page.evaluate(() => ({
        clientWidth: document.documentElement.clientWidth,
        scrollWidth: document.documentElement.scrollWidth
      }))
    )
    .toEqual(expect.objectContaining({ scrollWidth: expect.any(Number), clientWidth: expect.any(Number) }))

  const metrics = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth
  }))
  expect(metrics.scrollWidth, 'configuration UI should not create horizontal overflow').toBeLessThanOrEqual(
    metrics.clientWidth + 1
  )
}

test.describe('schema-driven configuration controls', () => {
  test.beforeEach(async ({ page }) => {
    await installConfigurationBackend(page)
  })

  test('renders schema-selected controls with bounds, runtime options, disabled reasons, and accessible layout', async ({
    page
  }, testInfo) => {
    await page.goto(appUrl('/configuration/models', testInfo))

    await expect(page.getByRole('heading', { name: 'Configuration' })).toBeVisible()
    await expect(page.getByRole('heading', { name: 'Model settings' })).toBeVisible()

    const contextSize = page.getByRole('slider', { name: 'Context size' })
    await expect(contextSize).toBeVisible()
    await expect(contextSize).toHaveAttribute('aria-valuemin', '2048')
    await expect(contextSize).toHaveAttribute('aria-valuemax', '262144')
    await expect(contextSize).toHaveAttribute('aria-valuenow', '4096')

    await expect(page.getByRole('radio', { name: 'balanced' })).toBeChecked()
    await expect(page.getByText('K q8_0 · V q4_0')).toBeVisible()

    await expect(page.getByLabel('GPU assignment').getByRole('radio', { name: 'auto' })).toBeChecked()
    await expect(page.getByRole('textbox', { name: 'Pinned GPU device' })).toBeDisabled()
    await expect(page.getByText('Only editable when GPU assignment is pinned.')).toBeVisible()

    await page.getByRole('tab', { name: 'Runtime' }).click()
    await expect(page.getByRole('heading', { name: 'Runtime settings' })).toBeVisible()
    await expect(page.getByRole('combobox', { name: 'Native backend' })).toHaveValue('metal')
    await expect(page.getByRole('option', { name: 'Vulkan' })).toBeDisabled()

    await page.getByRole('tab', { name: 'Models' }).click()
    await page.getByRole('button', { name: 'Show advanced' }).click()
    await expect(page.getByRole('textbox', { name: 'Multimodal projector' })).toBeDisabled()
    await expect(page.getByRole('button', { name: 'Why unavailable: Multimodal projector' })).toBeDisabled()
    await expect(page.getByRole('textbox', { name: 'Projector URL' })).toHaveValue('https://example.com/mmproj.gguf')

    await page.getByRole('tab', { name: 'Plugins' }).click()
    await expect(page.getByRole('heading', { name: 'Plugin settings' })).toBeVisible()
    await expect(page.getByRole('textbox', { name: 'Blackboard endpoint' })).toHaveValue('https://blackboard.local/api')

    await expectNoHorizontalOverflow(page)
    await testInfo.attach('configuration-controls-desktop', {
      body: await page.screenshot({ fullPage: true, animations: 'disabled' }),
      contentType: 'image/png'
    })

    const results = await new AxeBuilder({ page })
      .include('[data-screen-label="Configuration · plugins"]')
      .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
      .analyze()
    expect(results.violations).toEqual([])
  })

  test('writes edited settings to TOML preview and applies the same config payload', async ({ page }, testInfo) => {
    await page.goto(appUrl('/configuration/models', testInfo))
    await expect(page.getByRole('heading', { name: 'Model settings' })).toBeVisible()

    const contextSize = page.getByRole('slider', { name: 'Context size' })
    await page.getByRole('button', { name: '8K', exact: true }).click()
    await expect(contextSize).toHaveAttribute('aria-valuenow', '8192')
    await page.getByLabel('KV cache policy').getByRole('radio', { name: 'quality' }).click()
    await page.getByLabel('GPU assignment').getByRole('radio', { name: 'pinned' }).click()
    await expect(page.getByRole('textbox', { name: 'Pinned GPU device' })).toBeEnabled()
    await page.getByRole('textbox', { name: 'Pinned GPU device' }).fill('cuda:1')

    await page.getByRole('tab', { name: 'TOML Output' }).click()
    const toml = page.getByRole('textbox', { name: 'Configuration TOML source' })
    await expect(toml).toHaveValue(/ctx_size = 8192/)
    await expect(toml).toHaveValue(/kv_cache_policy = "quality"/)
    await expect(toml).toHaveValue(/assignment = "pinned"/)
    await expect(toml).toHaveValue(/gpu_id = "cuda:1"/)
    await expect(page.getByText('Generated TOML validates against mesh-llm config rules.')).toBeVisible()

    await page.getByRole('button', { name: 'Save config' }).click()

    await expect
      .poll(async () => {
        const state = await readConfigState(page)
        return state.applyRequests.length
      })
      .toBe(1)

    const state = await readConfigState(page)
    const applyRequest = state.applyRequests[0] as {
      endpoint: string
      expected_revision: number
      config: {
        defaults?: { model_fit?: JsonRecord; hardware?: JsonRecord }
        gpu?: JsonRecord
        plugin?: Array<{ name?: string; settings?: JsonRecord }>
      }
    }

    expect(applyRequest.endpoint).toBe('local-owner')
    expect(applyRequest.expected_revision).toBe(7)
    expect(applyRequest.config.defaults?.model_fit?.ctx_size).toBe(8192)
    expect(applyRequest.config.defaults?.model_fit?.kv_cache_policy).toBe('quality')
    expect(applyRequest.config.gpu?.assignment).toBe('pinned')
    expect(applyRequest.config.defaults?.hardware?.device).toBe('cuda:1')
    expect(applyRequest.config.plugin?.[0]?.settings?.endpoint).toBe('https://blackboard.local/api')
  })

  test('keeps the configuration surface usable on mobile without clipped controls', async ({ page }, testInfo) => {
    await page.setViewportSize({ width: 390, height: 720 })
    await page.goto(appUrl('/configuration/models', testInfo))

    await expect(page.getByRole('heading', { name: 'Configuration' })).toBeVisible()
    await expect(page.getByRole('heading', { name: 'Model settings' })).toBeVisible()
    await expect(page.getByRole('slider', { name: 'Context size' })).toBeVisible()
    await expect(page.getByRole('radio', { name: 'balanced' })).toBeVisible()

    await expectNoHorizontalOverflow(page)
    await testInfo.attach('configuration-controls-mobile', {
      body: await page.screenshot({ fullPage: true, animations: 'disabled' }),
      contentType: 'image/png'
    })
  })
})
