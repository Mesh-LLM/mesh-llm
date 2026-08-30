import { beforeEach, describe, expect, it } from 'vitest'
import { act, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { DefaultsTab } from '@/features/configuration/components/DefaultsTab'
import type { ConfigurationDefaultsHarnessData } from '@/features/app-tabs/types'
import { SETTING_RESET_TOOLTIP } from '@/features/configuration/components/settings/SettingResetButton'
import {
  CONFIGURATION_DEFAULTS,
  SHOW_ADVANCED_STORAGE_KEY,
  defaultsRail,
  previewSource,
  renderDefaultsTab,
  settingsRow
} from './DefaultsTab-test-support'

describe('DefaultsTab visual inventory and metadata', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })
  it('marks modified setting rows with the same warning tone used by dirty tabs', () => {
    renderDefaultsTab({
      values: {
        'runtime-mode': 'manual'
      }
    })

    const row = settingsRow('Runtime mode')
    expect(row).toHaveAttribute('data-settings-row-dirty', 'true')
    expect(within(row).getByText('Runtime mode')).toHaveClass('text-warn')
  })

  it('uses network-panel typography tokens for configuration helper text', () => {
    renderDefaultsTab({ configFilePath: '/Users/test/.mesh-llm/config.toml' })

    const runtimeSection = screen.getByRole('heading', { name: 'Runtime' }).closest('section')
    if (!(runtimeSection instanceof HTMLElement)) throw new Error('Expected runtime settings section')

    expect(screen.getByRole('heading', { name: 'Runtime' })).toHaveClass('type-panel-title', 'text-foreground')
    expect(within(runtimeSection).getByText('Runtime defaults')).toHaveClass('type-caption', 'text-fg-dim')

    const row = settingsRow('Runtime mode')
    expect(within(row).getByText('Controls the standard runtime selection.')).toHaveClass('type-caption', 'text-fg-dim')
    expect(screen.getByText('TIP')).toHaveClass('type-label')
    expect(screen.getByText('Configuration Path')).toHaveClass('type-label', 'text-fg-faint')
    expect(screen.getByText('/Users/test/.mesh-llm/config.toml')).toHaveClass(
      'font-mono',
      'text-[length:var(--density-type-caption-lg)]',
      'text-fg-dim'
    )
  })

  it('hides advanced settings by default and persists the toggle', async () => {
    const user = userEvent.setup()

    renderDefaultsTab({
      values: {
        'advanced-reasoning': '256'
      }
    })

    expect(screen.getByRole('button', { name: /show advanced/i })).toHaveAttribute('aria-pressed', 'false')
    expect(screen.getByRole('heading', { name: 'Runtime' })).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Reasoning' })).not.toBeInTheDocument()
    expect(previewSource().value).not.toContain('runtime_mode = "auto"')
    expect(previewSource().value).toContain('reasoning_budget = 256')

    await user.click(screen.getByRole('button', { name: /show advanced/i }))

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /hide advanced/i })).toHaveAttribute('aria-pressed', 'true')
      expect(screen.getByRole('heading', { name: 'Reasoning' })).toBeInTheDocument()
      expect(previewSource().value).toContain('[defaults.runtime]')
      expect(previewSource().value).toContain('reasoning_budget = 256')
      expect(window.localStorage.getItem(SHOW_ADVANCED_STORAGE_KEY)).toBe('true')
    })

    await user.click(screen.getByRole('button', { name: /hide advanced/i }))

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /show advanced/i })).toHaveAttribute('aria-pressed', 'false')
      expect(screen.queryByRole('heading', { name: 'Reasoning' })).not.toBeInTheDocument()
      expect(previewSource().value).toContain('[defaults.runtime]')
      expect(previewSource().value).toContain('reasoning_budget = 256')
      expect(window.localStorage.getItem(SHOW_ADVANCED_STORAGE_KEY)).toBeNull()
    })
  })

  it('hydrates show advanced from localStorage', () => {
    window.localStorage.setItem(SHOW_ADVANCED_STORAGE_KEY, 'true')

    renderDefaultsTab()

    expect(screen.getByRole('button', { name: /hide advanced/i })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('heading', { name: 'Reasoning' })).toBeInTheDocument()
    expect(previewSource().value).not.toContain('draft_selection_policy')
    expect(previewSource().value).not.toContain('prefill_chunk_size')
    expect(previewSource().value).not.toContain('prefill_chunk_schedule')
    expect(previewSource().value).not.toContain('mirostat_entropy')
  })

  it('renders the real defaults inventory with integrated categories and section previews', async () => {
    const user = userEvent.setup()

    const { rerender } = renderDefaultsTab({
      data: CONFIGURATION_DEFAULTS,
      values: {
        threads: '12',
        temperature: '0.8',
        'top-k': '55',
        'binary-stage-transport': 'on',
        'image-min-tokens': '64',
        'mmproj-offload': 'on',
        'server-alias': 'carrack-mesh',
        'direct-io': 'off',
        mlock: 'on'
      }
    })

    const rail = defaultsRail()
    expect(rail.getAllByRole('button')).toHaveLength(7)
    expect(rail.getByRole('button', { name: /runtime/i })).toBeInTheDocument()
    expect(rail.getByRole('button', { name: /request defaults/i })).toBeInTheDocument()
    expect(rail.getByRole('button', { name: /skippy transport/i })).toBeInTheDocument()
    expect(rail.getByRole('button', { name: /multimodal/i })).toBeInTheDocument()
    expect(rail.getByRole('button', { name: /topology/i })).toBeInTheDocument()
    expect(rail.queryByRole('button', { name: /advanced server/i })).not.toBeInTheDocument()
    expect(screen.queryByText('Server alias')).not.toBeInTheDocument()
    expect(screen.queryByText('Memory lock')).not.toBeInTheDocument()

    await user.click(rail.getByRole('button', { name: /request defaults/i }))
    expect(rail.getByRole('button', { name: /request defaults/i })).toHaveAttribute('aria-current', 'true')
    expect(screen.getByRole('heading', { name: 'Request Defaults' })).toBeInTheDocument()
    expect(screen.getByText('Temperature')).toBeInTheDocument()
    expect(screen.getByText('Top-k')).toBeInTheDocument()

    await user.click(rail.getByRole('button', { name: /skippy transport/i }))
    expect(rail.getByRole('button', { name: /skippy transport/i })).toHaveAttribute('aria-current', 'true')
    expect(screen.getByRole('heading', { name: 'Skippy Transport' })).toBeInTheDocument()
    expect(screen.getByText('Binary stage transport')).toBeInTheDocument()

    await user.click(rail.getByRole('button', { name: /multimodal/i }))
    expect(rail.getByRole('button', { name: /multimodal/i })).toHaveAttribute('aria-current', 'true')
    expect(screen.getByRole('heading', { name: 'Multimodal' })).toBeInTheDocument()
    expect(screen.getByText('MMProj offload')).toBeInTheDocument()

    expect(screen.getByRole('slider', { name: 'CPU threads' })).toHaveValue('12')
    expect(screen.getByRole('slider', { name: 'Temperature' })).toHaveValue('0.8')
    expect(screen.getByRole('slider', { name: 'Top-k' })).toHaveValue('55')
    expect(previewSource().value).toContain('threads = 12')
    expect(previewSource().value).toContain('[defaults.request_defaults]')
    expect(previewSource().value).toContain('temperature = 0.8')
    expect(previewSource().value).toContain('top_k = 55')
    expect(previewSource().value).toContain('[defaults.skippy]')
    expect(previewSource().value).toContain('binary_stage_transport = "on"')
    expect(previewSource().value).toContain('mlock = true')
    expect(previewSource().value).toContain('[defaults.multimodal]')
    expect(previewSource().value).toContain('image_min_tokens = 64')
    expect(previewSource().value).toContain('mmproj_offload = true')
    expect(previewSource().value).toContain('[defaults.advanced.server]')
    expect(previewSource().value).toContain('alias = "carrack-mesh"')

    rerender(
      <DefaultsTab
        data={CONFIGURATION_DEFAULTS}
        values={{
          threads: '12',
          temperature: '0.8',
          'top-k': '55',
          'binary-stage-transport': 'on',
          'image-min-tokens': '64',
          'mmproj-offload': 'on',
          'direct-io': 'off',
          mlock: 'on'
        }}
        onSettingValueChange={vi.fn()}
        onResetAll={vi.fn()}
      />
    )

    expect(previewSource().value).not.toContain('[defaults.advanced.server]')
  })

  it('omits default-only metadata sections while keeping hidden advanced non-default values in preview', () => {
    renderDefaultsTab({
      data: CONFIGURATION_DEFAULTS,
      values: {
        mlock: 'on',
        'server-alias': 'carrack-mesh'
      }
    })

    expect(screen.queryByText('Memory lock')).not.toBeInTheDocument()
    expect(screen.queryByText('Server alias')).not.toBeInTheDocument()
    expect(previewSource().value).toContain('[defaults.hardware]')
    expect(previewSource().value).toContain('mlock = true')
    expect(previewSource().value).toContain('[defaults.advanced.server]')
    expect(previewSource().value).toContain('alias = "carrack-mesh"')
    expect(previewSource().value).not.toContain('[defaults.skippy]')
    expect(previewSource().value).not.toContain('[defaults.multimodal]')
  })

  it('uses canonical inventory defaults when previewing hydrated live settings', () => {
    const liveHydratedDefaults = {
      ...CONFIGURATION_DEFAULTS,
      settings: CONFIGURATION_DEFAULTS.settings.map((setting) => {
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

    renderDefaultsTab({
      data: liveHydratedDefaults,
      values: {}
    })

    expect(screen.queryByText('Server alias')).not.toBeInTheDocument()
    expect(previewSource().value).toContain('[defaults.multimodal]')
    expect(previewSource().value).toContain('image_min_tokens = 64')
    expect(previewSource().value).toContain('[defaults.advanced.server]')
    expect(previewSource().value).toContain('alias = "carrack-mesh"')
    expect(previewSource().value).not.toContain('[defaults.request_defaults]')
  })

  it('shows schema-derived live and restart metadata for logging settings', async () => {
    const user = userEvent.setup()
    const loggingData = {
      categories: [
        {
          id: 'logs-retention',
          label: 'Retention',
          summary: 'Schema-derived retention settings.',
          help: 'Retention settings written to the local config file.'
        },
        {
          id: 'logs-artifacts',
          label: 'Artifacts & Storage',
          summary: 'Schema-derived artifact settings.',
          help: 'Artifact settings written to the local config file.'
        }
      ],
      settings: [
        {
          id: 'logging.retention_ttl_secs',
          categoryId: 'logs-retention',
          icon: 'gauge',
          label: 'Retention TTL',
          description: 'Server-provided retention copy.',
          inheritedLabel: 'Written to the local mesh-llm config file',
          visibility: 'advanced',
          mutability: 'runtime',
          applyMode: 'dynamic_apply',
          restartScope: 'none',
          valueSchema: { kind: 'integer' },
          validationConstraints: [{ kind: 'range', min: '60', max: '604800' }],
          control: { kind: 'range', name: 'retention_ttl_secs', value: '60', min: 60, max: 604800, step: 1 }
        },
        {
          id: 'logging.artifact.byte_limit_bytes',
          categoryId: 'logs-artifacts',
          icon: 'folder',
          label: 'Artifact size limit',
          description: 'Server-provided artifact limit copy.',
          inheritedLabel: 'Written to the local mesh-llm config file',
          visibility: 'advanced',
          mutability: 'restart-required',
          applyMode: 'static_on_load',
          restartScope: 'process_restart',
          valueSchema: { kind: 'integer' },
          controlState: {
            enabled: false,
            reason: 'Artifact capture is unavailable while capture mode is metadata_only.',
            source: 'runtime',
            write_policy: 'reject_when_disabled'
          },
          control: { kind: 'text', name: 'artifact_byte_limit_bytes', value: '' }
        },
        {
          id: 'logging.audit.webhook_timeout_secs',
          categoryId: 'logs-retention',
          icon: 'gauge',
          label: 'Webhook timeout',
          description: 'Validated before the configuration is saved.',
          inheritedLabel: 'Written to the local mesh-llm config file',
          visibility: 'advanced',
          mutability: 'runtime',
          applyMode: 'dynamic_validation_only',
          restartScope: 'none',
          valueSchema: { kind: 'integer' },
          control: { kind: 'text', name: 'webhook_timeout_secs', value: '10' }
        }
      ],
      preview: []
    } satisfies ConfigurationDefaultsHarnessData

    renderDefaultsTab({ data: loggingData })

    await user.click(screen.getByRole('button', { name: /show advanced/i }))

    const retentionRow = within(settingsRow('Retention TTL'))
    const artifactRow = within(settingsRow('Artifact size limit'))
    const webhookRow = within(settingsRow('Webhook timeout'))
    expect(retentionRow.getByText('Applies live')).toBeInTheDocument()
    expect(artifactRow.getByText('Restart required')).toBeInTheDocument()
    expect(webhookRow.getByText('Validated on save')).toBeInTheDocument()
    expect(retentionRow.getByRole('slider', { name: 'Retention TTL' })).toHaveAttribute('min', '60')
    expect(artifactRow.getByRole('spinbutton', { name: 'Artifact size limit' })).toBeDisabled()
    expect(artifactRow.getByRole('button', { name: /why unavailable/i })).toBeInTheDocument()
  })

  it('shows reset actions beside controls for restart-required settings and resets only that setting', async () => {
    const user = userEvent.setup()
    const onSettingValueChange = vi.fn()

    const { rerender } = renderDefaultsTab({
      data: CONFIGURATION_DEFAULTS,
      values: {
        threads: '12',
        'top-k': '55'
      },
      onSettingValueChange
    })

    const cpuThreadsRow = within(settingsRow('CPU threads'))
    const topKRow = within(settingsRow('Top-k'))
    const resetButton = cpuThreadsRow.getByRole('button', { name: 'Reset CPU threads to default' })

    expect(resetButton).toBeInTheDocument()
    expect(topKRow.queryByRole('button', { name: /reset top-k to default/i })).not.toBeInTheDocument()
    expect(
      screen.getByText('CPU threads').compareDocumentPosition(resetButton) & Node.DOCUMENT_POSITION_FOLLOWING
    ).toBeTruthy()
    expect(
      resetButton.compareDocumentPosition(screen.getByRole('slider', { name: 'CPU threads' })) &
        Node.DOCUMENT_POSITION_FOLLOWING
    ).toBeTruthy()

    await user.hover(resetButton)
    expect(await screen.findByText(SETTING_RESET_TOOLTIP, { selector: 'div' })).toBeInTheDocument()
    await user.unhover(resetButton)

    await act(async () => {
      resetButton.focus()
    })
    expect(await screen.findByText(SETTING_RESET_TOOLTIP, { selector: 'div' })).toBeInTheDocument()

    await user.click(resetButton)
    expect(onSettingValueChange).toHaveBeenCalledWith('threads', '0')
    expect(onSettingValueChange).not.toHaveBeenCalledWith('top-k', expect.anything())

    rerender(
      <DefaultsTab
        data={CONFIGURATION_DEFAULTS}
        values={{}}
        onSettingValueChange={onSettingValueChange}
        onResetAll={vi.fn()}
      />
    )
    expect(
      within(settingsRow('CPU threads')).queryByRole('button', { name: 'Reset CPU threads to default' })
    ).toBeNull()
  })

  it('keeps advanced filtering consistent across real category counts, rows, and section visibility', async () => {
    const user = userEvent.setup()

    renderDefaultsTab({ data: CONFIGURATION_DEFAULTS })

    const rail = defaultsRail()
    expect(rail.getAllByRole('button')).toHaveLength(7)
    expect(screen.queryByText('Mirostat mode')).not.toBeInTheDocument()
    expect(screen.queryByText('Server alias')).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /show advanced/i }))

    await waitFor(() => {
      expect(rail.getAllByRole('button')).toHaveLength(8)
      expect(rail.getByRole('button', { name: /advanced server/i })).toBeInTheDocument()
      expect(screen.getByText('Mirostat mode')).toBeInTheDocument()
      expect(screen.getByText('Server alias')).toBeInTheDocument()
    })

    await user.click(rail.getByRole('button', { name: /advanced server/i }))
    expect(rail.getByRole('button', { name: /advanced server/i })).toHaveAttribute('aria-current', 'true')

    await user.click(screen.getByRole('button', { name: /hide advanced/i }))

    await waitFor(() => {
      expect(rail.getAllByRole('button')).toHaveLength(7)
      expect(rail.queryByRole('button', { name: /advanced server/i })).not.toBeInTheDocument()
      expect(screen.queryByText('Mirostat mode')).not.toBeInTheDocument()
      expect(screen.queryByText('Server alias')).not.toBeInTheDocument()
      expect(rail.getByRole('button', { name: /runtime/i })).toHaveAttribute('aria-current', 'true')
    })
  })
})
