/* eslint-disable react-refresh/only-export-components */
import type { ReactElement, ReactNode } from 'react'
import { act, render as rtlRender, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, vi } from 'vitest'
import { AppProviders } from '@/app/providers/AppProviders'
import {
  adaptStatusToConfiguration,
  createConfigurationDefaultsValuesFromMeshConfig,
  type RuntimeConfigSchemaReference,
  type RuntimeControlMeshConfig
} from '@/features/configuration/api/config-adapter'
import type { ConfigurationHarnessData } from '@/features/app-tabs/types'
import type { StatusPayload } from '@/lib/api/types'
import type { DataMode } from '@/lib/data-mode/data-mode-context'
import type { MeshPluginUiConfigMountContext, MeshPluginUiMountHandle } from '@/features/plugins/web-ui/host-contract'
import type { PluginSummaryRaw, PluginWebUiStateRaw } from '@/lib/api/plugin-types'
import * as configAdapterModule from '@/features/configuration/api/config-adapter'
import * as configQueryModule from '@/features/configuration/api/use-config-query'

const blockedBlocker = vi.hoisted(() => ({ status: 'blocked', proceed: vi.fn(), reset: vi.fn() }))
const idleBlocker = vi.hoisted(() => ({ status: 'idle', proceed: vi.fn(), reset: vi.fn() }))
const mockUseBlocker = vi.hoisted(() => vi.fn())
const defaultBlockerTransition = vi.hoisted(() => ({
  current: { pathname: '/configuration/local-deployment' },
  next: { pathname: '/chat' }
}))
const featureFlagMocks = vi.hoisted(() => ({
  integrationsEnabled: false,
  signingAttestationEnabled: false,
  wakePolicyConfigurationEnabled: false
}))
const pluginQueryMocks = vi.hoisted(() => ({
  summaries: [] as import('@/lib/api/plugin-types').PluginSummaryRaw[],
  toggle: vi.fn(),
  importBundle: vi.fn(),
  register: vi.fn(),
  mountConfig: vi.fn(),
  unmountConfig: vi.fn(),
  visibleConfig: {
    plugin: 'blackboard',
    settings: { endpoint_url: 'https://blackboard.local/v1', retention_days: 30 },
    schema: { plugin_name: 'blackboard' }
  },
  mutateConfig: vi.fn()
}))

declare global {
  var __meshConfigurationPageTestGlobals: {
    useBlocker: (...args: unknown[]) => unknown
    useBooleanFeatureFlag: (path: string) => boolean
    usePluginSummariesQuery: (...args: unknown[]) => unknown
    useSetPluginWebUiEnabledMutation: (...args: unknown[]) => unknown
    usePluginWebUiConfigQuery: (pluginName: string, ...args: unknown[]) => unknown
    usePluginWebUiConfigMutation: (pluginName: string, ...args: unknown[]) => unknown
    importPluginUiBundle: (...args: unknown[]) => unknown
  }
}

globalThis.__meshConfigurationPageTestGlobals = {
  useBlocker: (...args) => mockUseBlocker(...args),
  useBooleanFeatureFlag: (path) => {
    if (path === 'configuration/integrations') return featureFlagMocks.integrationsEnabled
    if (path === 'configuration/signingAttestation') return featureFlagMocks.signingAttestationEnabled
    if (path === 'configuration/wakePolicyConfiguration') return featureFlagMocks.wakePolicyConfigurationEnabled
    return true
  },
  usePluginSummariesQuery: () => ({
    data: pluginQueryMocks.summaries,
    isError: false,
    isPending: false
  }),
  useSetPluginWebUiEnabledMutation: () => ({
    isPending: false,
    mutate: pluginQueryMocks.toggle
  }),
  usePluginWebUiConfigQuery: (pluginName) => ({
    data: { ...pluginQueryMocks.visibleConfig, plugin: pluginName },
    isError: false,
    isPending: false
  }),
  usePluginWebUiConfigMutation: (pluginName) => ({
    isPending: false,
    mutateAsync: (request: unknown) => pluginQueryMocks.mutateConfig(pluginName, request)
  }),
  importPluginUiBundle: (...args) => pluginQueryMocks.importBundle(...args)
}

function TestProviders({ children, dataMode = 'harness' }: { children: ReactNode; dataMode?: DataMode }) {
  return (
    <AppProviders initialDataMode={dataMode} persistDataMode={false}>
      {children}
    </AppProviders>
  )
}

function render(ui: ReactElement, options?: { dataMode?: DataMode }) {
  return rtlRender(ui, {
    wrapper: ({ children }) => <TestProviders dataMode={options?.dataMode}>{children}</TestProviders>
  })
}

function getCarrackSection() {
  const section = screen.getByRole('button', { name: /collapse carrack/i }).closest('section')
  if (!section) throw new Error('Expected carrack section')
  return section
}

function getTomlSource() {
  const source = screen.getByRole('textbox', { name: /configuration toml source/i })
  if (!(source instanceof HTMLTextAreaElement)) throw new Error('Expected configuration TOML source')
  return source
}

async function openTomlOutput(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('tab', { name: 'TOML Output' }))
  expect(screen.getByRole('heading', { name: 'Generated TOML' })).toBeInTheDocument()
  return getTomlSource()
}

function countTomlOccurrences(value: string) {
  return getTomlSource().value.split(value).length - 1
}

async function dispatchShortcut(key: string, init: KeyboardEventInit = {}) {
  const event = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...init })

  await act(async () => {
    window.dispatchEvent(event)
  })

  return event
}

function liveControlConfigData() {
  return {
    bootstrap: {
      enabled: true,
      local_only: true,
      requires_explicit_remote_endpoint: true,
      endpoint: 'control://owner'
    },
    snapshot: {
      revision: 7,
      config: {}
    }
  }
}

const STATUS_PAYLOAD: StatusPayload = {
  node_id: 'self',
  node_state: 'serving',
  model_name: '',
  peers: [],
  models: [],
  my_vram_gb: 0,
  gpus: [],
  serving_models: []
}

const PLUGIN_ONLY_SCHEMA: RuntimeConfigSchemaReference = {
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
      canonical_path: 'plugin.blackboard.settings.endpoint_url',
      owner: 'plugin',
      source: { kind: 'plugin', plugin_name: 'blackboard', allow_unvalidated_config: false },
      value_schema: { kind: 'string' },
      support: 'supported',
      control_surfaces: ['config_file', 'owner_control', 'plugin_manifest'],
      apply_mode: 'dynamic_apply',
      restart_scope: 'none',
      visibility: 'user',
      description: 'Endpoint used by the blackboard plugin.',
      presentation: {
        label: 'Endpoint URL',
        help: 'Endpoint used by the blackboard plugin.',
        category_id: 'connection',
        category_label: 'Connection',
        category_summary: 'Blackboard plugin connection settings',
        category_order: 10,
        setting_order: 10,
        control_hint: 'text'
      }
    }
  ]
}

function pluginOnlyMeshConfig(): RuntimeControlMeshConfig {
  return {
    version: 1,
    plugin: [
      {
        name: 'blackboard',
        settings: {
          endpoint_url: 'https://blackboard.local/v1'
        }
      }
    ]
  }
}

function pluginOnlyConfigurationData(config = pluginOnlyMeshConfig()): ConfigurationHarnessData {
  const defaultsValues = createConfigurationDefaultsValuesFromMeshConfig(config, PLUGIN_ONLY_SCHEMA)
  return adaptStatusToConfiguration(STATUS_PAYLOAD, [], defaultsValues, PLUGIN_ONLY_SCHEMA, config)
}

function integrationsOnlyConfigurationData(): ConfigurationHarnessData {
  const { plugins, ...data } = pluginOnlyConfigurationData()
  return { ...data, integrations: plugins }
}

type PluginSummaryOptions = {
  readonly description?: string
  readonly enabled?: boolean
  readonly status?: string
}

function pluginSummary(name: string, webUi: PluginWebUiStateRaw, options: PluginSummaryOptions = {}): PluginSummaryRaw {
  return {
    name,
    kind: 'bridge',
    enabled: options.enabled ?? true,
    status: options.status ?? 'running',
    description: options.description,
    capabilities: [],
    args: [],
    tools: [],
    web_ui: webUi
  }
}

function readyPluginWebUi(
  section: { readonly parent_tab?: string } = { parent_tab: 'integrations' }
): PluginWebUiStateRaw {
  return {
    state: 'ready',
    declared: true,
    enabled: true,
    available: true,
    pages: [
      {
        id: 'dashboard',
        label: 'Dashboard',
        route: 'dashboard',
        bundle_id: 'main',
        entry_script: 'dashboard.js'
      }
    ],
    config_sections: [
      {
        id: 'settings',
        title: 'Settings',
        entry_script: 'settings.js',
        parent_tab: section.parent_tab,
        bundle_id: 'main'
      }
    ],
    asset_base_url: '/api/plugins/blackboard/web-ui/assets/'
  }
}

function disabledPluginWebUi(): PluginWebUiStateRaw {
  return {
    ...readyPluginWebUi(),
    state: 'disabled',
    enabled: false,
    available: false,
    unavailable_reason: 'web UI disabled by configuration'
  }
}

function invalidPluginWebUi(): PluginWebUiStateRaw {
  return {
    ...readyPluginWebUi(),
    state: 'invalid',
    available: false,
    unavailable_reason: 'bundle missing'
  }
}

function pluginNotRunningWebUi(): PluginWebUiStateRaw {
  return {
    ...readyPluginWebUi(),
    state: 'plugin_not_running',
    available: false,
    unavailable_reason: 'plugin process unavailable'
  }
}

function nonePluginWebUi(): PluginWebUiStateRaw {
  return {
    state: 'none',
    declared: false,
    enabled: false,
    available: false
  }
}

beforeEach(() => {
  vi.clearAllMocks()
  vi.useRealTimers()
  featureFlagMocks.integrationsEnabled = false
  featureFlagMocks.signingAttestationEnabled = false
  featureFlagMocks.wakePolicyConfigurationEnabled = false
  pluginQueryMocks.summaries = []
  pluginQueryMocks.visibleConfig = {
    plugin: 'blackboard',
    settings: { endpoint_url: 'https://blackboard.local/v1', retention_days: 30 },
    schema: { plugin_name: 'blackboard' }
  }
  pluginQueryMocks.mutateConfig.mockResolvedValue(pluginQueryMocks.visibleConfig)
  pluginQueryMocks.importBundle.mockResolvedValue({ registerMeshPluginUi: pluginQueryMocks.register })
  pluginQueryMocks.register.mockReturnValue({ configSections: { settings: pluginQueryMocks.mountConfig } })
  pluginQueryMocks.mountConfig.mockImplementation(
    ({ element, host, section }: MeshPluginUiConfigMountContext): MeshPluginUiMountHandle => {
      const node = document.createElement('div')
      node.textContent = `Mounted ${host.plugin.name} ${section.id}`
      const setting = document.createElement('output')
      setting.textContent = String(host.config.visible.settings.endpoint_url)
      const button = document.createElement('button')
      button.type = 'button'
      button.textContent = 'Save retention'
      button.addEventListener('click', () => {
        void host.config.requestMutation({
          plugin: host.plugin.name,
          settings: { retention_days: 45 }
        })
      })
      element.append(node, setting, button)
      return {
        unmount: () => {
          pluginQueryMocks.unmountConfig()
          node.remove()
        }
      }
    }
  )
  mockUseBlocker.mockImplementation(
    ({ shouldBlockFn }: { shouldBlockFn: (transition: typeof defaultBlockerTransition) => boolean }) =>
      shouldBlockFn(defaultBlockerTransition) ? blockedBlocker : idleBlocker
  )
})

afterEach(() => {
  vi.restoreAllMocks()
  vi.useRealTimers()
})
export {
  blockedBlocker,
  configAdapterModule,
  configQueryModule,
  defaultBlockerTransition,
  disabledPluginWebUi,
  featureFlagMocks,
  getCarrackSection,
  getTomlSource,
  idleBlocker,
  invalidPluginWebUi,
  liveControlConfigData,
  mockUseBlocker,
  nonePluginWebUi,
  openTomlOutput,
  pluginNotRunningWebUi,
  pluginOnlyConfigurationData,
  pluginOnlyMeshConfig,
  pluginQueryMocks,
  pluginSummary,
  readyPluginWebUi,
  render,
  STATUS_PAYLOAD,
  PLUGIN_ONLY_SCHEMA,
  countTomlOccurrences,
  dispatchShortcut,
  integrationsOnlyConfigurationData
}
