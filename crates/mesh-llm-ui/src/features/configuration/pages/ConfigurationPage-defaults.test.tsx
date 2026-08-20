import { describe, expect, it, vi } from 'vitest'
import { fireEvent, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ConfigurationFixturePage as ConfigurationPage } from '@/features/configuration/pages/ConfigurationPage'
import { getTomlSource, render } from './ConfigurationPage-test-support'

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

describe('ConfigurationPage defaults controls', () => {
  it('renders the interactive slot meter as the only default slots control', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage enableNavigationBlocker={false} />)

    const slotRow = screen.getByText('Default slots / parallel requests').closest('[data-settings-row]')
    expect(slotRow).not.toBeNull()
    expect(within(slotRow as HTMLElement).queryByRole('slider')).not.toBeInTheDocument()
    expect(within(slotRow as HTMLElement).queryByRole('spinbutton')).not.toBeInTheDocument()
    expect(screen.getByRole('radio', { name: '4 slots' })).toBeChecked()

    await user.click(screen.getByRole('radio', { name: '12 slots' }))

    expect(screen.getByRole('radio', { name: '12 slots' })).toBeChecked()
    expect(screen.getByText('3.6 GB · 12 × 0.30 GB')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /save config/i })).toBeEnabled()

    await user.click(screen.getByRole('tab', { name: 'TOML Output' }))
    expect(getTomlSource().value).toContain('[defaults]')
    expect(getTomlSource().value).toContain('parallel = 12')
  })

  it('supports dragging across the Defaults slot meter', () => {
    render(<ConfigurationPage enableNavigationBlocker={false} />)

    const slotMeter = screen.getByTestId('defaults-slot-meter')
    vi.spyOn(slotMeter, 'getBoundingClientRect').mockReturnValue({
      bottom: 16,
      height: 16,
      left: 10,
      right: 314,
      top: 0,
      width: 304,
      x: 10,
      y: 0,
      toJSON: () => ({})
    })

    fireEvent.pointerDown(slotMeter, { buttons: 1, clientX: 10, pointerId: 1 })
    expect(screen.getByRole('radio', { name: '1 slot' })).toBeChecked()

    fireEvent.pointerMove(slotMeter, { buttons: 1, clientX: 225, pointerId: 1 })
    expect(screen.getByRole('radio', { name: '12 slots' })).toBeChecked()
    expect(screen.getByText('3.6 GB · 12 × 0.30 GB')).toBeInTheDocument()
  })

  it('updates KV cache memory tiers from the selected policy', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage enableNavigationBlocker={false} />)

    const policyControl = within(screen.getByRole('radiogroup', { name: 'KV cache policy' }))
    const tiers = () =>
      within(screen.getByRole('group', { name: 'KV cache memory tiers' }))
        .getAllByText(/^K /)
        .map((node) => node.closest('[data-kv-tier-active]'))

    expect(tiers().map((node) => node?.getAttribute('data-kv-tier-active'))).toEqual(['true', 'true', 'true'])

    await user.click(policyControl.getByRole('radio', { name: 'quality' }))
    expect(tiers().map((node) => node?.getAttribute('data-kv-tier-active'))).toEqual(['true', undefined, undefined])

    await user.click(policyControl.getByRole('radio', { name: 'balanced' }))
    expect(tiers().map((node) => node?.getAttribute('data-kv-tier-active'))).toEqual([undefined, 'true', undefined])

    await user.click(policyControl.getByRole('radio', { name: 'saver' }))
    expect(tiers().map((node) => node?.getAttribute('data-kv-tier-active'))).toEqual([undefined, undefined, 'true'])
  })

  it('renders speculative decoding defaults and writes them to TOML', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage enableNavigationBlocker={false} />)

    await user.click(screen.getByRole('button', { name: /speculative decoding/i }))

    expect(screen.getByRole('heading', { name: 'Speculative Decoding' })).toBeInTheDocument()
    expect(screen.queryByText('Speculative decoding defaults')).not.toBeInTheDocument()
    expect(screen.queryByText('Compatibility & fallback')).not.toBeInTheDocument()
    expect(screen.queryByText('Performance defaults')).not.toBeInTheDocument()
    expect(screen.queryByText('Observability')).not.toBeInTheDocument()
    expect(screen.queryByText('Enable speculative decoding by default')).not.toBeInTheDocument()
    expect(screen.queryByText('Require compatibility check')).not.toBeInTheDocument()
    expect(screen.getByText('Incompatible pairing behavior')).toBeInTheDocument()

    const modeControl = within(screen.getByRole('radiogroup', { name: 'Default speculation mode' }))
    const defaultsPreview = screen.getByRole('complementary', { name: /\[defaults/i })
    expect(modeControl.getByRole('radio', { name: 'auto' })).toBeChecked()
    expect(modeControl.getByRole('radio', { name: 'disabled' })).toBeInTheDocument()
    expect(defaultsPreview).not.toHaveTextContent('pairing_fault')
    expect(defaultsPreview).not.toHaveTextContent('draft_selection_policy')

    await user.click(modeControl.getByRole('radio', { name: 'draft' }))
    const enabledDraftPolicyControl = within(screen.getByRole('radiogroup', { name: 'Default draft selection policy' }))
    const enabledPairingBehaviorControl = within(
      screen.getByRole('radiogroup', { name: 'Incompatible pairing behavior' })
    )
    expect(enabledDraftPolicyControl.getByRole('radio', { name: 'auto' })).not.toBeDisabled()
    await user.click(enabledPairingBehaviorControl.getByRole('radio', { name: 'Fail launch' }))
    expect(enabledPairingBehaviorControl.getByRole('radio', { name: 'Fail launch' })).toBeChecked()
    expect(defaultsPreview).toHaveTextContent('pairing_fault = "fail_closed"')
    fireEvent.change(screen.getByRole('slider', { name: 'Default draft max tokens' }), { target: { value: '32' } })

    await user.click(screen.getByRole('tab', { name: 'TOML Output' }))
    expect(getTomlSource().value).toContain('[defaults.speculative]')
    expect(getTomlSource().value).not.toContain('enabled =')
    expect(getTomlSource().value).toContain('mode = "draft"')
    expect(getTomlSource().value).toContain('draft_max_tokens = 32')
    expect(getTomlSource().value).toContain('pairing_fault = "fail_closed"')
    expect(getTomlSource().value).not.toContain('draft_selection_policy = "auto"')
    expect(getTomlSource().value).not.toMatch(/^pairing_behavior =/m)
    expect(getTomlSource().value).not.toContain('incompatible_pairing_behavior')
    expect(getTomlSource().value).not.toContain('model_runtime = "cuda"')
    expect(getTomlSource().value).not.toContain('[defaults.request_defaults]')
    expect(getTomlSource().value).not.toContain('temperature = 0.70')
    expect(getTomlSource().value).not.toContain('reasoning_format = "auto"')
    expect(getTomlSource().value).not.toContain('llama_flavor')
    expect(getTomlSource().value).not.toContain('allow_cpu_speculation')
    expect(getTomlSource().value).not.toContain('diagnostics =')
  })

  it('disables draft speculative decoding controls unless mode is draft', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage enableNavigationBlocker={false} />)

    await user.click(screen.getByRole('button', { name: /speculative decoding/i }))

    const modeControl = () => within(screen.getByRole('radiogroup', { name: 'Default speculation mode' }))
    const draftPolicyControl = () => within(screen.getByRole('radiogroup', { name: 'Default draft selection policy' }))
    const pairingBehaviorControl = () =>
      within(screen.getByRole('radiogroup', { name: 'Incompatible pairing behavior' }))

    expect(screen.queryByRole('combobox', { name: 'Default draft selection policy' })).not.toBeInTheDocument()
    expect(screen.queryByRole('combobox', { name: 'Incompatible pairing behavior' })).not.toBeInTheDocument()
    expect(draftPolicyControl().queryByRole('radio', { name: 'Catalog recommended' })).not.toBeInTheDocument()
    expect(draftPolicyControl().queryByRole('radio', { name: 'Auto-detect' })).not.toBeInTheDocument()
    expect(draftPolicyControl().getByRole('radio', { name: 'auto' })).toBeChecked()
    expect(pairingBehaviorControl().getByRole('radio', { name: 'Warn & Disable' })).toBeChecked()
    expect(pairingBehaviorControl().getByRole('radio', { name: 'Fail launch' })).toBeInTheDocument()

    await user.click(modeControl().getByRole('radio', { name: 'disabled' }))

    expect(modeControl().getByRole('radio', { name: 'disabled' })).toBeChecked()
    expect(modeControl().getByRole('radio', { name: 'draft' })).not.toBeDisabled()
    expect(draftPolicyControl().getByRole('radio', { name: 'auto' })).toBeDisabled()
    expect(pairingBehaviorControl().getByRole('radio', { name: 'Warn & Disable' })).toBeDisabled()
    expect(pairingBehaviorControl().getByRole('radio', { name: 'Fail launch' })).toBeDisabled()
    expect(screen.getByRole('slider', { name: 'Default draft max tokens' })).toBeDisabled()
    expect(screen.getByRole('slider', { name: 'Default draft minimum tokens' })).toBeDisabled()
    expect(screen.queryByRole('slider', { name: 'Default draft acceptance threshold' })).not.toBeInTheDocument()
    expect(screen.queryByRole('radiogroup', { name: 'Allow CPU speculation' })).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /show advanced/i }))
    expect(screen.getByRole('slider', { name: 'Default draft acceptance threshold' })).toBeDisabled()

    await user.click(modeControl().getByRole('radio', { name: 'draft' }))
    expect(draftPolicyControl().getByRole('radio', { name: 'auto' })).not.toBeDisabled()
    expect(pairingBehaviorControl().getByRole('radio', { name: 'Warn & Disable' })).not.toBeDisabled()
    expect(screen.getByRole('slider', { name: 'Default draft max tokens' })).not.toBeDisabled()
  })
})
