// @vitest-environment jsdom

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { PluginSummaryRaw, PluginWebUiStateRaw } from '@/lib/api/plugin-types'
import {
  adaptPluginSummariesToWebUiEntries,
  buildPluginWebUiNavItems,
  fetchPluginWebUiConfig,
  pluginWebUiAssetUrl,
  requestPluginWebUiConfigMutation,
  resolvePluginWebUiAssetUrl,
  usePluginWebUiQuery,
  useSetPluginWebUiEnabledMutation
} from '@/features/plugins/api/plugin-web-ui'

const READY_WEB_UI = {
  state: 'ready',
  declared: true,
  enabled: true,
  available: true,
  pages: [
    {
      id: 'dashboard',
      label: 'Dashboard',
      icon: 'icons/dashboard.svg',
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
      parent_tab: 'integrations',
      bundle_id: 'main'
    }
  ],
  asset_base_url: '/api/plugins/blackboard/web-ui/assets/'
} satisfies PluginWebUiStateRaw

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('plugin web UI data adapters', () => {
  it('preserves ready metadata and marks it eligible for plugin web UI navigation', () => {
    const entries = adaptPluginSummariesToWebUiEntries([pluginSummary('blackboard', READY_WEB_UI)])

    expect(entries).toEqual([
      expect.objectContaining({
        pluginName: 'blackboard',
        declared: true,
        state: 'ready',
        available: true,
        navigationEligible: true,
        unavailableReason: undefined,
        pages: READY_WEB_UI.pages,
        configSections: READY_WEB_UI.config_sections
      })
    ])
    expect(buildPluginWebUiNavItems(entries)).toEqual([
      {
        pluginName: 'blackboard',
        pageId: 'dashboard',
        label: 'Dashboard',
        route: 'dashboard'
      }
    ])
  })

  it.each([
    ['disabled', { state: 'disabled', declared: true, enabled: false, available: false, unavailable_reason: 'off' }],
    [
      'invalid',
      { state: 'invalid', declared: true, enabled: true, available: false, unavailable_reason: 'bad bundle' }
    ],
    [
      'plugin_not_running',
      {
        state: 'plugin_not_running',
        declared: true,
        enabled: true,
        available: false,
        unavailable_reason: 'not running'
      }
    ],
    ['none', { state: 'none', declared: false, enabled: false, available: false }]
  ] satisfies readonly [string, PluginWebUiStateRaw][])(
    'keeps %s web UI state visible but not nav eligible',
    (_label, webUi) => {
      const [entry] = adaptPluginSummariesToWebUiEntries([pluginSummary('demo', webUi)])

      expect(entry).toEqual(
        expect.objectContaining({
          pluginName: 'demo',
          state: webUi.state,
          declared: webUi.declared,
          navigationEligible: false
        })
      )
      expect(buildPluginWebUiNavItems(entry ? [entry] : [])).toEqual([])
    }
  )

  it('derives asset URLs under the plugin web UI asset endpoint', () => {
    expect(pluginWebUiAssetUrl('blackboard', 'chunks/dashboard.js')).toBe(
      '/api/plugins/blackboard/web-ui/assets/chunks/dashboard.js'
    )
    expect(pluginWebUiAssetUrl('demo plugin', 'icons/settings icon.svg')).toBe(
      '/api/plugins/demo%20plugin/web-ui/assets/icons/settings%20icon.svg'
    )
    expect(resolvePluginWebUiAssetUrl('blackboard', READY_WEB_UI, 'settings.js')).toBe(
      'http://localhost:3000/api/plugins/blackboard/web-ui/assets/settings.js'
    )
  })

  it('uses the plugin-scoped config endpoint for visible settings and mutations', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)

      if (url.endsWith('/api/plugins/blackboard/web-ui/config') && !init) {
        return jsonResponse({
          plugin: 'blackboard',
          settings: { retention_days: 30 },
          schema: { plugin_name: 'blackboard' }
        })
      }
      if (url.endsWith('/api/plugins/blackboard/web-ui/config')) {
        expect(init?.method).toBe('PATCH')
        expect(JSON.parse(String(init?.body))).toEqual({
          plugin: 'blackboard',
          settings: { retention_days: 45 }
        })
        return jsonResponse({
          plugin: 'blackboard',
          settings: { retention_days: 45 },
          schema: { plugin_name: 'blackboard' }
        })
      }

      throw new Error(`Unexpected fetch request: ${url}`)
    })
    vi.stubGlobal('fetch', fetchMock)

    await expect(fetchPluginWebUiConfig('blackboard')).resolves.toEqual({
      plugin: 'blackboard',
      settings: { retention_days: 30 },
      schema: { plugin_name: 'blackboard' }
    })
    await expect(requestPluginWebUiConfigMutation('blackboard', { settings: { retention_days: 45 } })).resolves.toEqual(
      {
        plugin: 'blackboard',
        settings: { retention_days: 45 },
        schema: { plugin_name: 'blackboard' }
      }
    )
  })
})

describe('plugin web UI query hooks', () => {
  it('refreshes metadata and plugin summaries after a successful web UI enabled toggle', async () => {
    let enabled = true
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)

      if (url.endsWith('/api/plugins/blackboard/web-ui') && !init) {
        return jsonResponse(enabled ? READY_WEB_UI : disabledWebUi())
      }
      if (url.endsWith('/api/plugins/blackboard/web-ui/enabled')) {
        expect(init?.method).toBe('PATCH')
        expect(JSON.parse(String(init?.body))).toEqual({ enabled: false })
        enabled = false
        return jsonResponse(disabledWebUi())
      }
      if (url.endsWith('/api/plugins')) {
        return jsonResponse([pluginSummary('blackboard', enabled ? READY_WEB_UI : disabledWebUi())])
      }

      throw new Error(`Unexpected fetch request: ${url}`)
    })
    vi.stubGlobal('fetch', fetchMock)

    const { result } = renderHook(
      () => ({
        metadata: usePluginWebUiQuery('blackboard'),
        toggle: useSetPluginWebUiEnabledMutation('blackboard')
      }),
      { wrapper: createWrapper() }
    )

    await waitFor(() => expect(result.current.metadata.data?.state).toBe('ready'))

    await act(async () => {
      await result.current.toggle.mutateAsync(false)
    })

    await waitFor(() => expect(result.current.metadata.data?.state).toBe('disabled'))
    expect(fetchMock).toHaveBeenCalledWith('/api/plugins/blackboard/web-ui')
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/plugins/blackboard/web-ui/enabled',
      expect.objectContaining({ method: 'PATCH' })
    )
    expect(
      fetchMock.mock.calls.filter(([input]) => String(input).endsWith('/api/plugins/blackboard/web-ui'))
    ).toHaveLength(2)
  })
})

function pluginSummary(name: string, webUi: PluginWebUiStateRaw): PluginSummaryRaw {
  return {
    name,
    kind: 'bridge',
    enabled: true,
    status: 'running',
    capabilities: [],
    args: [],
    tools: [],
    web_ui: webUi
  }
}

function disabledWebUi(): PluginWebUiStateRaw {
  return {
    ...READY_WEB_UI,
    state: 'disabled',
    enabled: false,
    available: false,
    unavailable_reason: 'web UI disabled by configuration'
  }
}

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  })

  return function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  }
}

function jsonResponse(body: unknown) {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'Content-Type': 'application/json' }
  })
}
