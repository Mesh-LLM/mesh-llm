import { describe, expect, it, vi } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ConfigurationPage as LiveConfigurationPage } from '@/features/configuration/pages/ConfigurationPage'
import {
  PLUGIN_ONLY_SCHEMA,
  disabledPluginWebUi,
  featureFlagMocks,
  integrationsOnlyConfigurationData,
  invalidPluginWebUi,
  liveControlConfigData,
  nonePluginWebUi,
  pluginNotRunningWebUi,
  pluginOnlyConfigurationData,
  pluginOnlyMeshConfig,
  pluginQueryMocks,
  pluginSummary,
  readyPluginWebUi,
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

describe('ConfigurationPage plugin integrations', () => {
  it('waits for runtime schema before rendering live defaults', () => {
    const useConfigQuerySpy = vi.spyOn(configQueryModule, 'useConfigQuery').mockReturnValue({
      data: undefined,
      isError: false,
      isFetching: true,
      isPending: true,
      statusQuery: { refetch: vi.fn() } as never,
      modelsQuery: { refetch: vi.fn() } as never,
      controlConfigQuery: {
        data: undefined,
        isError: false,
        isFetching: true,
        isPending: true,
        refetch: vi.fn()
      } as never,
      applyDefaults: vi.fn()
    })

    render(<LiveConfigurationPage enableNavigationBlocker={false} />, { dataMode: 'live' })

    expect(document.querySelector('[data-loading-ghost-shimmer]')).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: /model settings/i })).not.toBeInTheDocument()
    expect(screen.queryByText('Default slots / parallel requests')).not.toBeInTheDocument()

    useConfigQuerySpy.mockRestore()
  })

  it('renders and resets a live schema that exposes only plugin settings', async () => {
    const user = userEvent.setup()
    const config = pluginOnlyMeshConfig()
    const applyDefaults = vi.fn()
    featureFlagMocks.integrationsEnabled = true
    const useConfigQuerySpy = vi.spyOn(configQueryModule, 'useConfigQuery').mockReturnValue({
      data: pluginOnlyConfigurationData(config),
      isError: false,
      isFetching: false,
      isPending: false,
      statusQuery: { refetch: vi.fn() } as never,
      modelsQuery: { refetch: vi.fn() } as never,
      controlConfigQuery: {
        data: {
          ...liveControlConfigData(),
          schema: PLUGIN_ONLY_SCHEMA,
          snapshot: {
            revision: 7,
            config
          }
        },
        isError: false,
        isFetching: false,
        isPending: false
      } as never,
      applyDefaults
    })

    render(<LiveConfigurationPage enableNavigationBlocker={false} initialTab="plugins" />, { dataMode: 'live' })

    expect(screen.getByRole('heading', { name: 'Configuration' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Plugin settings' })).toBeInTheDocument()
    expect(screen.queryByText(/no runtime configuration schema/i)).not.toBeInTheDocument()
    const pluginsTab = screen.getByRole('tab', { name: 'Plugins' })
    const saveButton = screen.getByRole('button', { name: /save config/i })
    const endpointInput = screen.getByRole('textbox', { name: 'Endpoint URL' })
    expect(endpointInput).toHaveValue('https://blackboard.local/v1')
    expect(pluginsTab).not.toHaveAttribute('data-tab-dirty')
    expect(saveButton).toBeDisabled()

    await user.clear(endpointInput)
    await user.type(endpointInput, 'https://blackboard.example/v2')

    expect(pluginsTab).toHaveAttribute('data-tab-dirty', 'true')
    expect(saveButton).toBeEnabled()

    await user.click(screen.getByRole('button', { name: /reset all/i }))

    expect(screen.getByRole('textbox', { name: 'Endpoint URL' })).toHaveValue('https://blackboard.local/v1')
    expect(pluginsTab).not.toHaveAttribute('data-tab-dirty')
    expect(saveButton).toBeDisabled()
    expect(applyDefaults).not.toHaveBeenCalled()

    useConfigQuerySpy.mockRestore()
  })

  it('preserves reset and dirty state for integrations-only compatibility payloads', async () => {
    const user = userEvent.setup()
    const config = pluginOnlyMeshConfig()
    featureFlagMocks.integrationsEnabled = true
    const useConfigQuerySpy = vi.spyOn(configQueryModule, 'useConfigQuery').mockReturnValue({
      data: integrationsOnlyConfigurationData(),
      isError: false,
      isFetching: false,
      isPending: false,
      statusQuery: { refetch: vi.fn() } as never,
      modelsQuery: { refetch: vi.fn() } as never,
      controlConfigQuery: {
        data: {
          ...liveControlConfigData(),
          schema: PLUGIN_ONLY_SCHEMA,
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

    render(<LiveConfigurationPage enableNavigationBlocker={false} initialTab="plugins" />, { dataMode: 'live' })

    const pluginsTab = screen.getByRole('tab', { name: 'Plugins' })
    const saveButton = screen.getByRole('button', { name: /save config/i })
    const endpointInput = screen.getByRole('textbox', { name: 'Endpoint URL' })
    expect(endpointInput).toHaveValue('https://blackboard.local/v1')
    expect(pluginsTab).not.toHaveAttribute('data-tab-dirty')

    await user.clear(endpointInput)
    await user.type(endpointInput, 'https://blackboard.example/v2')
    expect(pluginsTab).toHaveAttribute('data-tab-dirty', 'true')
    expect(saveButton).toBeEnabled()

    await user.click(screen.getByRole('button', { name: /revert/i }))

    expect(screen.getByRole('textbox', { name: 'Endpoint URL' })).toHaveValue('https://blackboard.local/v1')
    expect(pluginsTab).not.toHaveAttribute('data-tab-dirty')
    expect(saveButton).toBeDisabled()

    useConfigQuerySpy.mockRestore()
  })

  it('projects ready plugin web UI metadata, toggle, config section, and schema settings into Plugins', async () => {
    const user = userEvent.setup()
    const config = pluginOnlyMeshConfig()
    featureFlagMocks.integrationsEnabled = true
    pluginQueryMocks.summaries = [
      pluginSummary('blackboard', readyPluginWebUi(), { description: 'Team scratchpad plugin' })
    ]
    const useConfigQuerySpy = vi.spyOn(configQueryModule, 'useConfigQuery').mockReturnValue({
      data: pluginOnlyConfigurationData(config),
      isError: false,
      isFetching: false,
      isPending: false,
      statusQuery: { refetch: vi.fn() } as never,
      modelsQuery: { refetch: vi.fn() } as never,
      controlConfigQuery: {
        data: {
          ...liveControlConfigData(),
          schema: PLUGIN_ONLY_SCHEMA,
          snapshot: { revision: 7, config }
        },
        isError: false,
        isFetching: false,
        isPending: false
      } as never,
      applyDefaults: vi.fn()
    })

    render(<LiveConfigurationPage enableNavigationBlocker={false} initialTab="plugins" />, { dataMode: 'live' })

    expect(await screen.findByRole('heading', { name: 'blackboard' })).toBeInTheDocument()
    const settingsBannerHeading = screen.getByRole('heading', { name: 'Plugin settings' })
    const installedPluginsHeading = screen.getByRole('heading', { name: 'Installed plugins' })
    expect(
      settingsBannerHeading.compareDocumentPosition(installedPluginsHeading) & Node.DOCUMENT_POSITION_FOLLOWING
    ).toBeTruthy()
    expect(screen.getByText('Team scratchpad plugin')).toBeInTheDocument()
    expect(screen.getByText('Process enabled')).toBeInTheDocument()
    expect(screen.getByText('running')).toBeInTheDocument()
    expect(screen.getByText('Web UI ready')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Settings' })).toBeInTheDocument()
    expect(await screen.findByText('Mounted blackboard settings')).toBeInTheDocument()
    expect(screen.getByText('https://blackboard.local/v1')).toBeInTheDocument()
    expect(screen.getByRole('textbox', { name: 'Endpoint URL' })).toHaveValue('https://blackboard.local/v1')

    const toggle = screen.getByRole('switch', { name: 'blackboard web UI projection' })
    expect(toggle).toHaveAttribute('aria-checked', 'true')

    await user.click(toggle)

    expect(pluginQueryMocks.toggle).toHaveBeenCalledWith(false)
    await user.click(screen.getByRole('button', { name: 'Save retention' }))
    await waitFor(() =>
      expect(pluginQueryMocks.mutateConfig).toHaveBeenCalledWith('blackboard', {
        plugin: 'blackboard',
        settings: { retention_days: 45 }
      })
    )
    expect(await screen.findByText('Plugin settings saved.')).toBeInTheDocument()
    expect(pluginQueryMocks.importBundle).toHaveBeenCalledWith(
      'http://localhost:3000/api/plugins/blackboard/web-ui/assets/settings.js'
    )
    expect(pluginQueryMocks.mountConfig).toHaveBeenCalledTimes(1)

    useConfigQuerySpy.mockRestore()
  })

  it('keeps failure and nondeclaring plugin web UI states visible without mounting config sections', async () => {
    const config = pluginOnlyMeshConfig()
    featureFlagMocks.integrationsEnabled = true
    pluginQueryMocks.summaries = [
      pluginSummary('disabled-ui', disabledPluginWebUi()),
      pluginSummary('invalid-ui', invalidPluginWebUi()),
      pluginSummary('stopped-ui', pluginNotRunningWebUi()),
      pluginSummary('legacy-plugin', nonePluginWebUi()),
      pluginSummary('other-parent', readyPluginWebUi({ parent_tab: 'advanced' }))
    ]
    const useConfigQuerySpy = vi.spyOn(configQueryModule, 'useConfigQuery').mockReturnValue({
      data: pluginOnlyConfigurationData(config),
      isError: false,
      isFetching: false,
      isPending: false,
      statusQuery: { refetch: vi.fn() } as never,
      modelsQuery: { refetch: vi.fn() } as never,
      controlConfigQuery: {
        data: {
          ...liveControlConfigData(),
          schema: PLUGIN_ONLY_SCHEMA,
          snapshot: { revision: 7, config }
        },
        isError: false,
        isFetching: false,
        isPending: false
      } as never,
      applyDefaults: vi.fn()
    })

    render(<LiveConfigurationPage enableNavigationBlocker={false} initialTab="plugins" />, { dataMode: 'live' })

    for (const name of ['disabled-ui', 'invalid-ui', 'stopped-ui', 'legacy-plugin', 'other-parent']) {
      expect(screen.getByRole('heading', { name })).toBeInTheDocument()
    }
    expect(screen.getByText('web UI disabled by configuration')).toBeInTheDocument()
    expect(screen.getByText('bundle missing')).toBeInTheDocument()
    expect(screen.getByText('plugin process unavailable')).toBeInTheDocument()
    expect(screen.getByText('Web UI not declared')).toBeInTheDocument()
    expect(screen.queryByRole('switch', { name: 'legacy-plugin web UI projection' })).not.toBeInTheDocument()
    expect(screen.queryByText('Mounted disabled-ui settings')).not.toBeInTheDocument()
    expect(screen.queryByText('Mounted other-parent settings')).not.toBeInTheDocument()
    expect(pluginQueryMocks.importBundle).not.toHaveBeenCalled()

    useConfigQuerySpy.mockRestore()
  })

  it('unmounts plugin config sections on disable, tab change, and teardown', async () => {
    const user = userEvent.setup()
    const config = pluginOnlyMeshConfig()
    featureFlagMocks.integrationsEnabled = true
    pluginQueryMocks.summaries = [pluginSummary('blackboard', readyPluginWebUi())]
    const useConfigQuerySpy = vi.spyOn(configQueryModule, 'useConfigQuery').mockReturnValue({
      data: pluginOnlyConfigurationData(config),
      isError: false,
      isFetching: false,
      isPending: false,
      statusQuery: { refetch: vi.fn() } as never,
      modelsQuery: { refetch: vi.fn() } as never,
      controlConfigQuery: {
        data: {
          ...liveControlConfigData(),
          schema: PLUGIN_ONLY_SCHEMA,
          snapshot: { revision: 7, config }
        },
        isError: false,
        isFetching: false,
        isPending: false
      } as never,
      applyDefaults: vi.fn()
    })
    const { rerender, unmount } = render(
      <LiveConfigurationPage enableNavigationBlocker={false} initialTab="plugins" />,
      {
        dataMode: 'live'
      }
    )

    expect(await screen.findByText('Mounted blackboard settings')).toBeInTheDocument()

    pluginQueryMocks.summaries = [pluginSummary('blackboard', disabledPluginWebUi())]
    rerender(<LiveConfigurationPage enableNavigationBlocker={false} initialTab="plugins" />)

    await waitFor(() => expect(screen.queryByText('Mounted blackboard settings')).not.toBeInTheDocument())
    expect(pluginQueryMocks.unmountConfig).toHaveBeenCalledTimes(1)

    pluginQueryMocks.summaries = [pluginSummary('blackboard', readyPluginWebUi())]
    rerender(<LiveConfigurationPage enableNavigationBlocker={false} initialTab="plugins" />)
    expect(await screen.findByText('Mounted blackboard settings')).toBeInTheDocument()

    await user.click(screen.getByRole('tab', { name: 'Models' }))
    await waitFor(() => expect(pluginQueryMocks.unmountConfig).toHaveBeenCalledTimes(2))

    await user.click(screen.getByRole('tab', { name: 'Plugins' }))
    expect(await screen.findByText('Mounted blackboard settings')).toBeInTheDocument()
    unmount()

    expect(pluginQueryMocks.unmountConfig).toHaveBeenCalledTimes(3)
    useConfigQuerySpy.mockRestore()
  })
})
