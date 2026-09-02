import { describe, expect, it, vi } from 'vitest'
import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ConfigurationFixturePage as ConfigurationPage } from '@/features/configuration/pages/ConfigurationPage'
import { blockedBlocker, getCarrackSection, mockUseBlocker, render } from './ConfigurationPage-test-support'

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

describe('ConfigurationPage navigation blocker', () => {
  it('shows the navigation blocker only when enabled and there are unsaved changes', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage initialTab="local-deployment" />)

    expect(mockUseBlocker).toHaveBeenCalled()
    expect(screen.queryByRole('dialog', { name: 'Unsaved configuration' })).not.toBeInTheDocument()

    await user.click(within(getCarrackSection()).getByRole('radio', { name: 'pooled' }))

    expect(screen.getByRole('dialog', { name: 'Unsaved configuration' })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Stay' }))
    expect(blockedBlocker.reset).toHaveBeenCalled()
  })

  it('skips navigation blocking when disabled', () => {
    render(<ConfigurationPage initialTab="local-deployment" enableNavigationBlocker={false} />)

    expect(mockUseBlocker).not.toHaveBeenCalled()
    expect(screen.queryByRole('dialog', { name: 'Unsaved configuration' })).not.toBeInTheDocument()
  })
})
