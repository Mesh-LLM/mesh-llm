import { describe, expect, it, vi } from 'vitest'
import { act, fireEvent, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import {
  ConfigurationFixturePage as ConfigurationPage,
  ConfigurationPage as LiveConfigurationPage
} from '@/features/configuration/pages/ConfigurationPage'
import { CONFIGURATION_HARNESS } from '@/features/app-tabs/data'
import {
  adaptStatusToConfiguration,
  createConfigurationDefaultsValuesFromMeshConfig,
  type RuntimeConfigSchemaReference,
  type RuntimeControlMeshConfig
} from '@/features/configuration/api/config-adapter'
import {
  STATUS_PAYLOAD,
  dispatchShortcut,
  getTomlSource,
  liveControlConfigData,
  openTomlOutput,
  render
} from './ConfigurationPage-test-support'
import * as configAdapterModule from '@/features/configuration/api/config-adapter'
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

describe('ConfigurationPage live saving and diagnostics', () => {
  it('renders live local node placement data in Model Deployment', async () => {
    const user = userEvent.setup()
    const useConfigQuerySpy = vi.spyOn(configQueryModule, 'useConfigQuery').mockReturnValue({
      data: {
        ...CONFIGURATION_HARNESS,
        nodes: [
          {
            id: 'self',
            hostname: 'carrack.local',
            region: 'tor-1',
            status: 'online',
            cpu: 'Local runtime',
            ramGB: 0,
            placement: 'separate',
            gpus: [
              { idx: 0, name: 'RTX 5090', totalGB: 34.2, reservedGB: 0.9 },
              { idx: 1, name: 'RTX 6000 Pro', totalGB: 48, reservedGB: 1.1 }
            ]
          }
        ],
        assigns: []
      },
      isError: false,
      isFetching: false,
      isPending: false,
      statusQuery: { refetch: vi.fn() } as never,
      modelsQuery: { refetch: vi.fn() } as never,
      controlConfigQuery: {
        data: liveControlConfigData(),
        isError: false,
        isFetching: false,
        isPending: false
      } as never,
      applyDefaults: vi.fn()
    })

    render(<LiveConfigurationPage enableNavigationBlocker={false} initialTab="models" />, { dataMode: 'live' })

    await user.click(screen.getByRole('tab', { name: 'Model Deployment' }))

    const nodeRail = screen.getByRole('navigation', { name: /configuration nodes/i })
    expect(within(nodeRail).getByText('Nodes · 1')).toHaveClass('type-label', 'text-fg-faint')
    expect(within(nodeRail).getByText('carrack.local')).toHaveClass(
      'font-mono',
      'text-[length:var(--density-type-control)]'
    )
    expect(within(nodeRail).getByText('2 devices')).toHaveClass(
      'font-mono',
      'text-[length:var(--density-type-caption-lg)]',
      'text-fg-dim'
    )

    useConfigQuerySpy.mockRestore()
  })

  it('saves live defaults through useConfigQuery.applyDefaults only when Save config is clicked', async () => {
    const applyDefaults = vi.fn().mockResolvedValue({
      success: true,
      current_revision: 8,
      config_hash: 'abc123',
      apply_mode: 'live'
    })
    const useConfigQuerySpy = vi.spyOn(configQueryModule, 'useConfigQuery').mockReturnValue({
      data: CONFIGURATION_HARNESS,
      isError: false,
      isFetching: false,
      isPending: false,
      statusQuery: { refetch: vi.fn() } as never,
      modelsQuery: { refetch: vi.fn() } as never,
      controlConfigQuery: {
        data: liveControlConfigData(),
        isError: false,
        isFetching: false,
        isPending: false
      } as never,
      applyDefaults
    })

    render(<LiveConfigurationPage enableNavigationBlocker={false} initialTab="models" />, { dataMode: 'live' })

    const tuningProfileControl = within(screen.getByRole('radiogroup', { name: 'Default tuning profile' }))
    const saveButton = screen.getByRole('button', { name: /save config/i })

    fireEvent.click(tuningProfileControl.getByRole('radio', { name: 'throughput' }))
    expect(applyDefaults).not.toHaveBeenCalled()

    fireEvent.click(tuningProfileControl.getByRole('radio', { name: 'saver' }))
    expect(applyDefaults).not.toHaveBeenCalled()

    fireEvent.click(saveButton)

    await waitFor(() => expect(applyDefaults).toHaveBeenCalledTimes(1))
    expect(applyDefaults).toHaveBeenCalledWith(
      expect.objectContaining({
        values: expect.objectContaining({
          'tuning-profile': 'saver'
        })
      })
    )

    useConfigQuerySpy.mockRestore()
  })

  it('shows owner-control remediation and keeps dirty state when live saving is disabled', async () => {
    const applyDefaults = vi.fn()
    const useConfigQuerySpy = vi.spyOn(configQueryModule, 'useConfigQuery').mockReturnValue({
      data: CONFIGURATION_HARNESS,
      isError: false,
      isFetching: false,
      isPending: false,
      statusQuery: { refetch: vi.fn() } as never,
      modelsQuery: { refetch: vi.fn() } as never,
      controlConfigQuery: {
        data: {
          bootstrap: {
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
          }
        },
        isError: false,
        isFetching: false,
        isPending: false
      } as never,
      applyDefaults
    })

    render(<LiveConfigurationPage enableNavigationBlocker={false} initialTab="models" />, { dataMode: 'live' })

    const readOnlyHeading = screen.getByRole('heading', { name: 'Configuration UI is read-only' })
    const inheritedDefaultsHeading = screen.getByRole('heading', { name: /model settings/i })
    const defaultsTab = screen.getByRole('tab', { name: 'Models' })
    expect(readOnlyHeading).toHaveClass('type-panel-title', 'text-foreground')
    expect(defaultsTab.compareDocumentPosition(readOnlyHeading) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(
      inheritedDefaultsHeading.compareDocumentPosition(readOnlyHeading) & Node.DOCUMENT_POSITION_FOLLOWING
    ).toBeTruthy()
    expect(screen.getByText('No owner-control identity on this node, run both commands to unlock saving.')).toHaveClass(
      'type-caption',
      'text-fg-dim'
    )
    expect(screen.getByText('missing owner identity')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /docs/i })).toHaveAttribute('href', 'https://meshllm.cloud/')
    expect(screen.queryByRole('button', { name: /copy both/i })).not.toBeInTheDocument()
    expect(screen.getAllByText('mesh-llm')).toHaveLength(2)
    expect(screen.getByText('auth')).toBeInTheDocument()
    expect(screen.getByText('init')).toBeInTheDocument()
    expect(screen.getByText('serve')).toBeInTheDocument()
    expect(screen.getByText('--no-passphrase')).toBeInTheDocument()
    expect(screen.getByText('--owner-required')).toBeInTheDocument()
    const authHintRow = screen.getByText('Initialize owner identity (creates a local keypair)').closest('div')
    const restartHintRow = screen.getByText('Restart the daemon so the new identity takes effect').closest('div')
    if (!(authHintRow instanceof HTMLElement)) throw new Error('Expected auth command hint row')
    if (!(restartHintRow instanceof HTMLElement)) throw new Error('Expected restart command hint row')
    expect(authHintRow).toHaveClass('type-caption', 'text-fg-dim')
    expect(restartHintRow).toHaveClass('type-caption', 'text-fg-dim')
    expect(screen.getByRole('button', { name: 'Copy mesh-llm auth init --no-passphrase' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Copy mesh-llm serve --owner-required' })).toBeInTheDocument()

    fireEvent.click(screen.getByRole('radio', { name: 'throughput' }))
    const saveButton = screen.getByRole('button', { name: /save config/i })
    expect(saveButton).toBeDisabled()
    expect(saveButton).toHaveAttribute('title', 'Runtime control is disabled: missing owner identity')
    expect(defaultsTab).toHaveAttribute('data-tab-dirty', 'true')

    await dispatchShortcut('s', { ctrlKey: true })

    expect(applyDefaults).not.toHaveBeenCalled()
    expect(screen.getByRole('alert')).toHaveTextContent(
      'Config was not saved. Runtime control is disabled: missing owner identity.'
    )
    expect(defaultsTab).toHaveAttribute('data-tab-dirty', 'true')

    useConfigQuerySpy.mockRestore()
  })

  it('shows runtime-control apply errors without rewriting them as missing owner identity', async () => {
    const applyDefaults = vi.fn().mockResolvedValue({
      success: false,
      current_revision: 7,
      config_hash: 'abc123',
      apply_mode: 'unspecified',
      error: 'revision conflict: current revision is 9'
    })
    const useConfigQuerySpy = vi.spyOn(configQueryModule, 'useConfigQuery').mockReturnValue({
      data: CONFIGURATION_HARNESS,
      isError: false,
      isFetching: false,
      isPending: false,
      statusQuery: { refetch: vi.fn() } as never,
      modelsQuery: { refetch: vi.fn() } as never,
      controlConfigQuery: {
        data: liveControlConfigData(),
        isError: false,
        isFetching: false,
        isPending: false
      } as never,
      applyDefaults
    })

    render(<LiveConfigurationPage enableNavigationBlocker={false} initialTab="models" />, { dataMode: 'live' })

    fireEvent.click(screen.getByRole('radio', { name: 'throughput' }))
    fireEvent.click(screen.getByRole('button', { name: /save config/i }))

    await waitFor(() => expect(applyDefaults).toHaveBeenCalledTimes(1))
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent(
      'Config was not saved. Runtime control rejected the update: revision conflict: current revision is 9'
    )
    expect(alert).not.toHaveTextContent('missing owner identity')
    expect(screen.getByRole('tab', { name: 'Models' })).toHaveAttribute('data-tab-dirty', 'true')

    useConfigQuerySpy.mockRestore()
  })

  it('shows a busy Save config button while live defaults are being written', async () => {
    let resolveApply: (value: {
      success: boolean
      current_revision: number
      config_hash: string
      apply_mode: string
    }) => void = () => undefined
    const applyDefaults = vi.fn(
      () =>
        new Promise<{
          success: boolean
          current_revision: number
          config_hash: string
          apply_mode: string
        }>((resolve) => {
          resolveApply = resolve
        })
    )
    const useConfigQuerySpy = vi.spyOn(configQueryModule, 'useConfigQuery').mockReturnValue({
      data: CONFIGURATION_HARNESS,
      isError: false,
      isFetching: false,
      isPending: false,
      statusQuery: { refetch: vi.fn() } as never,
      modelsQuery: { refetch: vi.fn() } as never,
      controlConfigQuery: {
        data: liveControlConfigData(),
        isError: false,
        isFetching: false,
        isPending: false
      } as never,
      applyDefaults
    })

    render(<LiveConfigurationPage enableNavigationBlocker={false} initialTab="models" />, { dataMode: 'live' })

    fireEvent.click(screen.getByRole('radio', { name: 'throughput' }))
    fireEvent.click(screen.getByRole('button', { name: /save config/i }))

    const savingButton = screen.getByRole('button', { name: /saving config/i })
    expect(savingButton).toBeDisabled()
    expect(savingButton).toHaveAttribute('aria-busy', 'true')

    await act(async () => {
      resolveApply({ success: true, current_revision: 8, config_hash: 'abc123', apply_mode: 'live' })
    })

    expect(screen.getByRole('button', { name: /save config/i })).toBeDisabled()
    expect(screen.getByRole('button', { name: /save config/i })).not.toHaveAttribute('aria-busy', 'true')

    useConfigQuerySpy.mockRestore()
  })

  it('shows backend validation diagnostics for contradictory TOML even when the schema would disable the field in the UI', async () => {
    const user = userEvent.setup()
    const schema: RuntimeConfigSchemaReference = {
      plugin_instances: [],
      settings: [
        {
          canonical_path: 'defaults.speculative.mode',
          owner: 'built_in',
          source: { kind: 'built_in' },
          value_schema: { kind: 'enum', values: ['draft', 'disabled'] },
          support: 'supported',
          control_surfaces: ['config_file', 'owner_control'],
          apply_mode: 'dynamic_apply',
          restart_scope: 'none',
          visibility: 'user',
          description: 'Controls speculative mode.',
          presentation: {
            label: 'Default speculation mode',
            help: 'Controls speculative mode.',
            category_id: 'speculative-decoding',
            category_label: 'Speculative Decoding',
            category_summary: 'Speculative defaults',
            category_order: 10,
            setting_order: 10,
            control_hint: 'segmented'
          }
        },
        {
          canonical_path: 'defaults.speculative.draft_max_tokens',
          owner: 'built_in',
          source: { kind: 'built_in' },
          value_schema: { kind: 'integer' },
          support: 'supported',
          control_surfaces: ['config_file', 'owner_control'],
          apply_mode: 'dynamic_apply',
          restart_scope: 'none',
          visibility: 'user',
          description: 'Draft token cap.',
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
            label: 'Default draft max tokens',
            help: 'Draft token cap.',
            category_id: 'speculative-decoding',
            category_label: 'Speculative Decoding',
            category_summary: 'Speculative defaults',
            category_order: 10,
            setting_order: 20,
            control_hint: 'number'
          }
        }
      ]
    }
    const config: RuntimeControlMeshConfig = {
      version: 1,
      defaults: {
        speculative: {
          mode: 'disabled',
          draft_max_tokens: 16
        }
      }
    }
    const defaultsValues = createConfigurationDefaultsValuesFromMeshConfig(config, schema)
    const validationSpy = vi.spyOn(configAdapterModule, 'validateRuntimeConfigToml').mockResolvedValue({
      ok: false,
      diagnostics: [
        {
          code: 'invalid_value',
          severity: 'error',
          source: 'backend',
          path: 'defaults.speculative.draft_max_tokens',
          canonical_path: 'defaults.speculative.draft_max_tokens',
          message: 'draft_max_tokens requires defaults.speculative.mode = draft'
        }
      ]
    })
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

    render(<LiveConfigurationPage enableNavigationBlocker={false} initialTab="models" />, { dataMode: 'live' })

    await user.click(screen.getByRole('button', { name: /speculative decoding/i }))
    expect(screen.getByRole('spinbutton', { name: 'Default draft max tokens' })).toBeDisabled()

    await user.click(screen.getByRole('tab', { name: 'TOML Output' }))

    await waitFor(() =>
      expect(screen.getByText('draft_max_tokens requires defaults.speculative.mode = draft')).toHaveClass(
        'toml-warning-message'
      )
    )
    expect(screen.getByText('defaults.speculative.draft_max_tokens')).toHaveClass('toml-warning-path')
    expect(getTomlSource().value).toContain('[defaults.speculative]')
    expect(getTomlSource().value).toContain('mode = "disabled"')
    expect(getTomlSource().value).toContain('draft_max_tokens = 16')

    validationSpy.mockRestore()
    useConfigQuerySpy.mockRestore()
  })

  it('does not re-apply saved live defaults when the hook callback identity changes', async () => {
    const firstApplyDefaults = vi.fn().mockResolvedValue({
      success: true,
      current_revision: 8,
      config_hash: 'abc123',
      apply_mode: 'live'
    })
    const secondApplyDefaults = vi.fn().mockResolvedValue({
      success: true,
      current_revision: 9,
      config_hash: 'def456',
      apply_mode: 'live'
    })
    let currentApplyDefaults = firstApplyDefaults

    const useConfigQuerySpy = vi.spyOn(configQueryModule, 'useConfigQuery').mockImplementation(() => ({
      data: CONFIGURATION_HARNESS,
      isError: false,
      isFetching: false,
      isPending: false,
      statusQuery: { refetch: vi.fn() } as never,
      modelsQuery: { refetch: vi.fn() } as never,
      controlConfigQuery: {
        data: liveControlConfigData(),
        isError: false,
        isFetching: false,
        isPending: false
      } as never,
      applyDefaults: currentApplyDefaults
    }))

    const { rerender } = render(<LiveConfigurationPage enableNavigationBlocker={false} initialTab="models" />, {
      dataMode: 'live'
    })

    fireEvent.click(screen.getByRole('radio', { name: 'throughput' }))
    fireEvent.click(screen.getByRole('button', { name: /save config/i }))

    await waitFor(() => expect(firstApplyDefaults).toHaveBeenCalledTimes(1))

    currentApplyDefaults = secondApplyDefaults
    rerender(<LiveConfigurationPage enableNavigationBlocker={false} initialTab="models" />)

    expect(firstApplyDefaults).toHaveBeenCalledTimes(1)
    expect(secondApplyDefaults).not.toHaveBeenCalled()

    useConfigQuerySpy.mockRestore()
  })

  it('shows hydrated live non-default defaults while omitting unchanged metadata sections', async () => {
    const user = userEvent.setup()
    const liveDefaults = {
      ...CONFIGURATION_HARNESS.defaults,
      settings: CONFIGURATION_HARNESS.defaults.settings.map((setting) =>
        setting.id === 'temperature'
          ? {
              ...setting,
              baselineValue: setting.control.value,
              control: {
                ...setting.control,
                value: '0.8'
              }
            }
          : setting.id === 'server-alias'
            ? {
                ...setting,
                baselineValue: setting.control.value,
                control: {
                  ...setting.control,
                  value: 'carrack-mesh'
                }
              }
            : setting.id === 'image-min-tokens'
              ? {
                  ...setting,
                  baselineValue: setting.control.value,
                  control: {
                    ...setting.control,
                    value: '64'
                  }
                }
              : setting
      )
    }
    const useConfigQuerySpy = vi.spyOn(configQueryModule, 'useConfigQuery').mockReturnValue({
      data: { ...CONFIGURATION_HARNESS, defaults: liveDefaults },
      isError: false,
      isFetching: false,
      isPending: false,
      statusQuery: { refetch: vi.fn() } as never,
      modelsQuery: { refetch: vi.fn() } as never,
      controlConfigQuery: {
        data: null,
        isError: false,
        isFetching: false,
        isPending: false
      } as never,
      applyDefaults: vi.fn()
    })

    render(<LiveConfigurationPage enableNavigationBlocker={false} initialTab="models" />, { dataMode: 'live' })

    const initialTomlSource = await openTomlOutput(user)
    expect(initialTomlSource.value).toContain('[defaults.request_defaults]')
    expect(initialTomlSource.value).toContain('temperature = 0.8')
    expect(initialTomlSource.value).toContain('[defaults.multimodal]')
    expect(initialTomlSource.value).toContain('image_min_tokens = 64')
    expect(initialTomlSource.value).toContain('[defaults.advanced.server]')
    expect(initialTomlSource.value).toContain('alias = "carrack-mesh"')

    await user.click(screen.getByRole('tab', { name: 'Models' }))
    await user.click(screen.getByRole('button', { name: /request defaults/i }))
    expect(screen.getByRole('slider', { name: 'Temperature' })).toHaveValue('0.8')
    const skippyTransport = within(screen.getByRole('radiogroup', { name: 'Binary stage transport' }))
    const multimodalOffload = within(screen.getByRole('radiogroup', { name: 'MMProj offload' }))
    await user.click(skippyTransport.getByRole('radio', { name: 'on' }))
    await user.click(multimodalOffload.getByRole('radio', { name: 'on' }))
    const updatedTomlSource = await openTomlOutput(user)
    expect(updatedTomlSource.value).toContain('[defaults.skippy]')
    expect(updatedTomlSource.value).toContain('[defaults.multimodal]')

    useConfigQuerySpy.mockRestore()
  })

  it('does not call applyDefaults for defaults edits outside live mode', async () => {
    const user = userEvent.setup()
    const applyDefaults = vi.fn()
    const useConfigQuerySpy = vi.spyOn(configQueryModule, 'useConfigQuery').mockReturnValue({
      data: CONFIGURATION_HARNESS,
      isError: false,
      isFetching: false,
      isPending: false,
      statusQuery: { refetch: vi.fn() } as never,
      modelsQuery: { refetch: vi.fn() } as never,
      controlConfigQuery: {
        data: null,
        isError: false,
        isFetching: false,
        isPending: false
      } as never,
      applyDefaults
    })

    render(<ConfigurationPage enableNavigationBlocker={false} />, { dataMode: 'harness' })

    await user.click(screen.getByRole('radio', { name: 'throughput' }))

    await waitFor(() => expect(screen.getByRole('button', { name: /save config/i })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: /save config/i }))
    expect(applyDefaults).not.toHaveBeenCalled()

    useConfigQuerySpy.mockRestore()
  })
})
