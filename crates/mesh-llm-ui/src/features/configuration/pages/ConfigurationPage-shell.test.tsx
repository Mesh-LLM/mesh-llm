import { describe, expect, it, vi } from 'vitest'
import { screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import {
  ConfigurationFixturePage as ConfigurationPage,
  ConfigurationPage as LiveConfigurationPage
} from '@/features/configuration/pages/ConfigurationPage'
import {
  adaptStatusToConfiguration,
  createConfigurationDefaultsValuesFromMeshConfig,
  type RuntimeConfigSchemaReference,
  type RuntimeControlMeshConfig
} from '@/features/configuration/api/config-adapter'
import {
  STATUS_PAYLOAD,
  featureFlagMocks,
  getTomlSource,
  liveControlConfigData,
  render
} from './ConfigurationPage-test-support'
import * as configQueryModule from '@/features/configuration/api/use-config-query'

vi.mock('@tanstack/react-router', () => ({
  useBlocker: (...args: unknown[]) => globalThis.__meshConfigurationPageTestGlobals.useBlocker(...args)
}))

vi.mock('@/lib/feature-flags', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/feature-flags')>()

  return {
    ...actual,
    useBooleanFeatureFlag: vi.fn((path: string) =>
      globalThis.__meshConfigurationPageTestGlobals.useBooleanFeatureFlag(path)
    )
  }
})

vi.mock('@/features/plugins/api/plugin-web-ui', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/features/plugins/api/plugin-web-ui')>()

  return {
    ...actual,
    usePluginSummariesQuery: vi.fn((...args: unknown[]) =>
      globalThis.__meshConfigurationPageTestGlobals.usePluginSummariesQuery(...args)
    ),
    useSetPluginWebUiEnabledMutation: vi.fn((...args: unknown[]) =>
      globalThis.__meshConfigurationPageTestGlobals.useSetPluginWebUiEnabledMutation(...args)
    ),
    usePluginWebUiConfigQuery: vi.fn((pluginName: string, ...args: unknown[]) =>
      globalThis.__meshConfigurationPageTestGlobals.usePluginWebUiConfigQuery(pluginName, ...args)
    ),
    usePluginWebUiConfigMutation: vi.fn((pluginName: string, ...args: unknown[]) =>
      globalThis.__meshConfigurationPageTestGlobals.usePluginWebUiConfigMutation(pluginName, ...args)
    )
  }
})

vi.mock('@/features/plugins/web-ui/bundle-loader', () => ({
  importPluginUiBundle: (...args: unknown[]) =>
    globalThis.__meshConfigurationPageTestGlobals.importPluginUiBundle(...args),
  assertPluginUiRegistration: vi.fn(),
  assertPluginUiMountHandle: vi.fn()
}))

describe('ConfigurationPage shell and feature flags', () => {
  it('renders the persistent header, shared tab bar, and model settings workspace first', () => {
    render(<ConfigurationPage enableNavigationBlocker={false} />)

    expect(screen.getByRole('heading', { name: 'Configuration' })).toBeInTheDocument()
    expect(screen.getByText('carrack.local')).toBeInTheDocument()
    expect(screen.getByText('Configuration Path')).toBeInTheDocument()
    expect(screen.getByText('~/.mesh-llm/config.toml')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /save config/i })).toBeDisabled()

    for (const label of ['General', 'Runtime', 'Models', 'Network', 'Model Deployment', 'TOML Output']) {
      expect(screen.getByRole('tab', { name: label })).toBeInTheDocument()
    }
    expect(screen.queryByRole('tab', { name: 'Reserves' })).not.toBeInTheDocument()
    expect(screen.queryByRole('tab', { name: 'Signing / Attestation' })).not.toBeInTheDocument()
    expect(screen.queryByRole('tab', { name: 'Plugins' })).not.toBeInTheDocument()

    expect(screen.getByRole('heading', { name: /model settings/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /runtime/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /speculative decoding/i })).toBeInTheDocument()
    expect(screen.getByText('Default slots / parallel requests')).toBeInTheDocument()
    expect(screen.getByRole('complementary', { name: /\[defaults/i })).toBeInTheDocument()
    expect(screen.queryByRole('dialog', { name: 'Model catalog' })).not.toBeInTheDocument()
  })

  it('updates the general preview heading after gpu settings move to models', async () => {
    const schema: RuntimeConfigSchemaReference = {
      plugin_instances: [],
      settings: [
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
            help: 'Enable debug output on startup.',
            category_id: 'meshllm',
            category_label: 'General',
            category_summary: 'Local process settings',
            category_order: 10,
            setting_order: 10,
            control_hint: 'toggle'
          }
        }
      ]
    }
    const config: RuntimeControlMeshConfig = {
      version: 1,
      runtime: {
        debug: true
      }
    }
    const defaultsValues = createConfigurationDefaultsValuesFromMeshConfig(config, schema)
    const useConfigQuerySpy = vi.spyOn(configQueryModule, 'useConfigQuery').mockReturnValue({
      data: adaptStatusToConfiguration(STATUS_PAYLOAD, [], defaultsValues, schema, config),
      isError: false,
      isFetching: false,
      isPending: false,
      statusQuery: { refetch: vi.fn() } as never,
      modelsQuery: { refetch: vi.fn() } as never,
      controlConfigQuery: {
        data: {
          ...liveControlConfigData(),
          schema,
          snapshot: {
            revision: 7,
            config
          }
        },
        isError: false,
        isFetching: false,
        isPending: false
      } as never,
      applyDefaults: vi.fn()
    })

    render(<LiveConfigurationPage enableNavigationBlocker={false} initialTab="general" />, { dataMode: 'live' })

    expect(screen.getByRole('complementary', { name: /\[runtime\] \/ \[telemetry\]/i })).toBeInTheDocument()

    useConfigQuerySpy.mockRestore()
  })

  it('shows reserves and temporary configuration sections only when their feature flags are enabled', async () => {
    const user = userEvent.setup()
    featureFlagMocks.integrationsEnabled = true
    featureFlagMocks.signingAttestationEnabled = true
    featureFlagMocks.wakePolicyConfigurationEnabled = true

    render(<ConfigurationPage enableNavigationBlocker={false} />)

    const wakePolicyTab = screen.getByRole('tab', { name: 'Reserves' })
    const signingTab = screen.getByRole('tab', { name: 'Signing / Attestation' })
    const pluginsTab = screen.getByRole('tab', { name: 'Plugins' })
    expect(wakePolicyTab).toBeInTheDocument()
    expect(signingTab).toBeInTheDocument()
    expect(pluginsTab).toBeInTheDocument()

    await user.click(wakePolicyTab)
    expect(screen.getByRole('heading', { level: 2, name: 'Reserves' })).toBeInTheDocument()
    expect(screen.getByText(/backend persistence is still being wired/i)).toBeInTheDocument()

    await user.click(signingTab)
    expect(screen.getByRole('heading', { name: 'Signing / Attestation' })).toBeInTheDocument()
    expect(screen.getByText(/no writable attestation settings/i)).toBeInTheDocument()

    await user.click(pluginsTab)
    expect(screen.getByRole('heading', { name: 'Plugins' })).toBeInTheDocument()
    expect(screen.getByText(/plugin settings will appear here/i)).toBeInTheDocument()
  })

  it('keeps directly requested gated sections on the General workspace', () => {
    const { rerender } = render(<ConfigurationPage initialTab="wake-policy" enableNavigationBlocker={false} />)

    expect(screen.queryByRole('tab', { name: 'Reserves' })).not.toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Reserves' })).not.toBeInTheDocument()
    expect(screen.getByRole('tab', { name: 'General' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('heading', { name: /general settings/i })).toBeInTheDocument()

    rerender(<ConfigurationPage initialTab="signing" enableNavigationBlocker={false} />)

    expect(screen.queryByRole('heading', { name: 'Signing / Attestation' })).not.toBeInTheDocument()
    expect(screen.getByRole('tab', { name: 'General' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('heading', { name: /general settings/i })).toBeInTheDocument()
  })

  it('applies configuration section feature flags independently', () => {
    featureFlagMocks.signingAttestationEnabled = true
    featureFlagMocks.integrationsEnabled = false
    featureFlagMocks.wakePolicyConfigurationEnabled = false

    const { rerender } = render(<ConfigurationPage enableNavigationBlocker={false} />)

    expect(screen.getByRole('tab', { name: 'Signing / Attestation' })).toBeInTheDocument()
    expect(screen.queryByRole('tab', { name: 'Plugins' })).not.toBeInTheDocument()
    expect(screen.queryByRole('tab', { name: 'Reserves' })).not.toBeInTheDocument()

    featureFlagMocks.signingAttestationEnabled = false
    featureFlagMocks.integrationsEnabled = true
    featureFlagMocks.wakePolicyConfigurationEnabled = true
    rerender(<ConfigurationPage enableNavigationBlocker={false} />)

    expect(screen.queryByRole('tab', { name: 'Signing / Attestation' })).not.toBeInTheDocument()
    expect(screen.getByRole('tab', { name: 'Plugins' })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: 'Reserves' })).toBeInTheDocument()
  })

  it('renders the model settings sections and updates the active sidebar category', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage enableNavigationBlocker={false} />)

    const memoryButton = screen.getByRole('button', { name: /memory/i })
    await user.click(memoryButton)

    expect(memoryButton).toHaveAttribute('aria-current', 'true')
    expect(screen.getByRole('heading', { name: 'Runtime' })).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Backend' })).not.toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Memory' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Speculative Decoding' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Request Defaults' })).toBeInTheDocument()
    expect(screen.queryByText('Model Runtime')).not.toBeInTheDocument()
    expect(screen.getByText('Default GPU device')).toBeInTheDocument()
    expect(screen.getByText('GPU layers')).toBeInTheDocument()
    expect(screen.getByText('KV cache policy')).toBeInTheDocument()
    expect(screen.getByText('Memory / safety margin')).toBeInTheDocument()
    expect(screen.getByText('Reasoning format')).toBeInTheDocument()
    expect(screen.getByText('Temperature')).toBeInTheDocument()
  })

  it('includes model settings edits in dirty state, save, revert, and TOML review', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage enableNavigationBlocker={false} />)

    const saveButton = screen.getByRole('button', { name: /save config/i })
    const defaultsTab = screen.getByRole('tab', { name: 'Models' })
    const tomlReviewTab = screen.getByRole('tab', { name: 'TOML Output' })
    expect(saveButton).toBeDisabled()
    expect(defaultsTab).not.toHaveAttribute('data-tab-dirty')

    await user.click(screen.getByRole('radio', { name: 'throughput' }))
    expect(saveButton).toBeEnabled()
    expect(defaultsTab).toHaveAttribute('data-tab-dirty', 'true')
    expect(tomlReviewTab).toHaveAttribute('data-tab-dirty', 'true')

    await user.click(tomlReviewTab)
    expect(screen.getByRole('heading', { name: 'Generated TOML' })).toBeInTheDocument()
    expect(screen.getByText('edits this node only')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Validation' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Effective launch summary' })).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Save' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /save config & sign/i })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /revert to disk/i })).not.toBeInTheDocument()
    expect(getTomlSource().value).toContain('tuning_profile = "throughput"')

    await user.click(saveButton)
    expect(defaultsTab).not.toHaveAttribute('data-tab-dirty')
    expect(tomlReviewTab).not.toHaveAttribute('data-tab-dirty')

    await user.click(screen.getByRole('tab', { name: 'Models' }))
    await user.click(screen.getAllByRole('radio', { name: 'saver' })[0])
    expect(defaultsTab).toHaveAttribute('data-tab-dirty', 'true')

    await user.click(screen.getByRole('button', { name: /revert/i }))
    expect(saveButton).toBeDisabled()
    expect(defaultsTab).not.toHaveAttribute('data-tab-dirty')
    await user.click(tomlReviewTab)
    expect(getTomlSource().value).toContain('tuning_profile = "throughput"')
  })
})
