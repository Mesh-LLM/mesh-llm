import { useEffect } from 'react'
import { QueryClient, useQueryClient } from '@tanstack/react-query'
import {
  HeadContent,
  Outlet,
  RouterProvider,
  createMemoryHistory,
  createRootRoute,
  createRoute,
  createRouter
} from '@tanstack/react-router'
import { act, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { AppProviders } from '@/app/providers/AppProviders'
import type { ConfigurationTabId } from '@/features/configuration/components/configuration-tab-ids'
import { ConfigurationRoutePage } from '@/features/configuration/pages/ConfigurationRoutePage'
import { ChatPageContent } from '@/features/chat/pages/ChatPage'
import { DeveloperPlaygroundPage } from '@/features/developer/pages/DeveloperPlaygroundPage'
import { DashboardPageSurface } from '@/features/network/pages/DashboardPage'
import { parseDeveloperPlaygroundSearch } from '@/features/developer/playground/developer-playground-tabs'
import { ReservesPageContent } from '@/features/reserves/pages/ReservesPage'
import { PluginWebUiRoutePage } from '@/features/plugins/web-ui/PluginWebUiRoutePage'
import { pluginKeys, statusKeys } from '@/lib/query/query-keys'
import type { PluginWebUiStateRaw } from '@/lib/api/plugin-types'
import type {
  MeshPluginUiHost,
  MeshPluginUiMountContext,
  MeshPluginUiMountHandle,
  MeshPluginUiRegistration
} from '@/features/plugins/web-ui/host-contract'

const routeCacheProbe = vi.hoisted(() => ({
  dashboardClient: undefined as QueryClient | undefined,
  chatClient: undefined as QueryClient | undefined
}))
const pluginBundleProbe = vi.hoisted(() => ({
  importBundle: vi.fn(),
  register: vi.fn(),
  mount: vi.fn(),
  unmount: vi.fn(),
  host: undefined as MeshPluginUiHost | undefined
}))

vi.mock('@/features/plugins/web-ui/bundle-loader', () => ({
  importPluginUiBundle: pluginBundleProbe.importBundle,
  assertPluginUiRegistration: vi.fn(),
  assertPluginUiMountHandle: vi.fn()
}))

vi.mock('@/features/reserves/pages/ReservesPage', () => ({
  ReservesPageContent: () => <div>Reserves route</div>
}))

vi.mock('@/features/developer/pages/DeveloperPlaygroundPage', async () => {
  const router = await vi.importActual<typeof import('@tanstack/react-router')>('@tanstack/react-router')

  return {
    DeveloperPlaygroundPage: () => {
      const { tab } = router.useSearch({ from: '/__playground' })

      return <div>Active developer route tab: {tab}</div>
    }
  }
})

vi.mock('@/features/configuration/pages/ConfigurationPage', () => ({
  ConfigurationPageContent: ({ activeTab }: { activeTab: ConfigurationTabId }) => (
    <div>Active route tab: {activeTab}</div>
  )
}))

vi.mock('@/features/network/pages/DashboardPage', () => ({
  DashboardPageSurface: () => {
    const queryClient = useQueryClient()

    useEffect(() => {
      routeCacheProbe.dashboardClient = queryClient
      queryClient.setQueryData(statusKeys.detail(), { source: 'dashboard-cache' })
    }, [queryClient])

    return <div>Dashboard route</div>
  }
}))

vi.mock('@/features/chat/pages/ChatPage', () => ({
  ChatPageContent: () => {
    const queryClient = useQueryClient()
    routeCacheProbe.chatClient = queryClient
    const cachedStatus = queryClient.getQueryData<{ source: string }>(statusKeys.detail())

    return <div>Chat route cache: {cachedStatus?.source ?? 'missing'}</div>
  }
}))

vi.mock('@/lib/feature-flags', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/feature-flags')>()

  return {
    ...actual,
    useBooleanFeatureFlag: () => true
  }
})

function TestRootLayout() {
  return (
    <>
      <HeadContent />
      <Outlet />
    </>
  )
}

const rootRoute = createRootRoute({ component: TestRootLayout })
const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  head: () => ({ meta: [{ title: 'MeshLLM - Dashboard' }] }),
  component: DashboardPageSurface
})
const chatRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/chat',
  head: () => ({ meta: [{ title: 'MeshLLM - Chat' }] }),
  component: ChatPageContent
})
const reservesRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/reserves',
  head: () => ({ meta: [{ title: 'MeshLLM - Reserves' }] }),
  component: ReservesPageContent
})
const configurationRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/configuration',
  head: () => ({ meta: [{ title: 'MeshLLM - Configuration' }] }),
  component: ConfigurationRoutePage
})
const configurationTabRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/configuration/$configurationTab',
  head: () => ({ meta: [{ title: 'MeshLLM - Configuration' }] }),
  component: ConfigurationRoutePage
})
const developerPlaygroundRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/__playground',
  head: () => ({ meta: [{ title: 'MeshLLM - Developer Playground' }] }),
  validateSearch: parseDeveloperPlaygroundSearch,
  component: DeveloperPlaygroundPage
})
const pluginWebUiRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/plugins/$pluginName/$pageId',
  head: () => ({ meta: [{ title: 'MeshLLM - Plugin' }] }),
  component: PluginWebUiRoutePage
})
const testRouteTree = rootRoute.addChildren([
  indexRoute,
  reservesRoute,
  chatRoute,
  configurationRoute,
  configurationTabRoute,
  pluginWebUiRoute,
  developerPlaygroundRoute
])

function renderRouterAt(pathname: string, queryClient = new QueryClient()) {
  return renderRouterWithHistory(createMemoryHistory({ initialEntries: [pathname] }), queryClient)
}

function renderRouterWithHistory(history: ReturnType<typeof createMemoryHistory>, queryClient = new QueryClient()) {
  const testRouter = createRouter({
    history,
    routeTree: testRouteTree
  })

  render(
    <AppProviders initialDataMode="harness" persistDataMode={false} queryClient={queryClient}>
      <RouterProvider router={testRouter} />
    </AppProviders>
  )

  return testRouter
}

describe('app router routes', () => {
  beforeEach(() => {
    pluginBundleProbe.importBundle.mockReset()
    pluginBundleProbe.register.mockReset()
    pluginBundleProbe.mount.mockReset()
    pluginBundleProbe.unmount.mockReset()
    pluginBundleProbe.host = undefined
    vi.unstubAllGlobals()
  })

  it.each([
    ['/', 'MeshLLM - Dashboard', 'Dashboard route'],
    ['/reserves', 'MeshLLM - Reserves', 'Reserves route'],
    ['/chat', 'MeshLLM - Chat', 'Chat route cache: missing'],
    ['/configuration/defaults', 'MeshLLM - Configuration', 'Active route tab: general'],
    ['/__playground?tab=shell-controls', 'MeshLLM - Developer Playground', 'Active developer route tab: shell-controls']
  ])('sets the document title for %s', async (pathname, title, routeText) => {
    renderRouterAt(pathname)

    await screen.findByText(routeText)
    await waitFor(() => expect(document.title).toBe(title))
  })

  it('canonicalizes the bare configuration route to the default tab path', async () => {
    const testRouter = renderRouterAt('/configuration')

    await screen.findByText('Active route tab: general')
    await waitFor(() => expect(testRouter.state.location.pathname).toBe('/configuration/general'))
  })

  it('restores a configuration tab from the path segment on initial load', async () => {
    const testRouter = renderRouterAt('/configuration/local-deployment')

    await screen.findByText('Active route tab: local-deployment')
    expect(testRouter.state.location.pathname).toBe('/configuration/local-deployment')
  })

  it('restores a developer playground tab from the search params on initial load', async () => {
    const testRouter = renderRouterAt('/__playground?tab=data-display')

    await screen.findByText('Active developer route tab: data-display')
    expect(testRouter.state.location.pathname).toBe('/__playground')
    expect(testRouter.state.location.search).toMatchObject({ tab: 'data-display' })
  })

  it('falls back to the default developer playground tab for unknown search params', async () => {
    const testRouter = renderRouterAt('/__playground?tab=missing-tab')

    await screen.findByText('Active developer route tab: shell-controls')
    expect(testRouter.state.location.search).toMatchObject({ tab: 'shell-controls' })
  })

  it('preserves the developer playground tab when browser back returns to the page', async () => {
    const history = createMemoryHistory({
      initialEntries: ['/', '/__playground?tab=chat-components', '/configuration/defaults'],
      initialIndex: 2
    })
    const testRouter = renderRouterWithHistory(history)

    await screen.findByText('Active route tab: general')

    await act(async () => {
      history.back()
    })

    await screen.findByText('Active developer route tab: chat-components')
    await waitFor(() => expect(testRouter.state.location.pathname).toBe('/__playground'))
    expect(testRouter.state.location.search).toMatchObject({ tab: 'chat-components' })
  })

  it('reuses the shared query cache when navigating from dashboard to chat', async () => {
    const testRouter = renderRouterAt('/')

    await screen.findByText('Dashboard route')

    await act(async () => {
      await testRouter.navigate({ to: '/chat' })
    })

    await screen.findByText('Chat route cache: dashboard-cache')
    expect(routeCacheProbe.chatClient).toBe(routeCacheProbe.dashboardClient)
  })

  it('mounts a ready plugin page through the typed host bundle contract', async () => {
    installReadyPluginBundle()
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.endsWith('/api/plugins/blackboard/web-ui/config') && init?.method === 'PATCH') {
        expect(JSON.parse(String(init.body))).toEqual({
          plugin: 'blackboard',
          settings: { retention_days: 45 }
        })
        return jsonResponse(visiblePluginConfig({ retention_days: 45 }))
      }
      if (url.endsWith('/api/plugins/blackboard/web-ui/config')) return jsonResponse(visiblePluginConfig())
      return jsonResponse(readyPluginWebUi())
    })
    vi.stubGlobal('fetch', fetchMock)

    renderRouterAt('/plugins/blackboard/dashboard')

    expect(await screen.findByText('Mounted blackboard dashboard')).toBeInTheDocument()
    expect(pluginBundleProbe.importBundle).toHaveBeenCalledWith(
      'http://localhost:3000/api/plugins/blackboard/web-ui/assets/dashboard.js'
    )
    expect(pluginBundleProbe.register).toHaveBeenCalledTimes(1)
    expect(pluginBundleProbe.mount).toHaveBeenCalledTimes(1)
    expect(pluginBundleProbe.host?.plugin.name).toBe('blackboard')
    expect(pluginBundleProbe.host?.page.id).toBe('dashboard')
    expect(pluginBundleProbe.host?.network.fetchPlugin).toEqual(expect.any(Function))
    expect(pluginBundleProbe.host?.config.visible.settings.retention_days).toBe(30)
    await act(async () => {
      await pluginBundleProbe.host?.config.requestMutation({ settings: { retention_days: 45 } })
    })
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/plugins/blackboard/web-ui/config',
      expect.objectContaining({ method: 'PATCH' })
    )
    expect(document.title).toBe('MeshLLM - Plugin')
  })

  it.each([
    ['disabled', disabledPluginWebUi()],
    ['invalid', invalidPluginWebUi()],
    ['plugin_not_running', pluginNotRunningWebUi()],
    ['nondeclaring', nonePluginWebUi()]
  ])('renders the %s fallback without importing bundle code', async (_label, webUi) => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => jsonResponse(webUi))
    )

    renderRouterAt('/plugins/blackboard/dashboard')

    await screen.findByRole('heading')
    expect(pluginBundleProbe.importBundle).not.toHaveBeenCalled()
    expect(pluginBundleProbe.mount).not.toHaveBeenCalled()
  })

  it('does not import a ready bundle when the requested page id is undeclared', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => jsonResponse(readyPluginWebUi()))
    )

    renderRouterAt('/plugins/blackboard/missing')

    await screen.findByRole('heading', { name: 'Plugin page is not declared' })
    expect(pluginBundleProbe.importBundle).not.toHaveBeenCalled()
  })

  it('does not import a ready bundle when the host omits asset_base_url', async () => {
    const { asset_base_url: _assetBaseUrl, ...webUi } = readyPluginWebUi()
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => jsonResponse(webUi))
    )

    renderRouterAt('/plugins/blackboard/dashboard')

    await screen.findByRole('heading', { name: 'Plugin web UI asset route is unavailable' })
    expect(pluginBundleProbe.importBundle).not.toHaveBeenCalled()
  })

  it('unmounts the plugin page exactly once when navigation leaves the route', async () => {
    installReadyPluginBundle()
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => jsonResponse(readyPluginWebUi()))
    )
    const testRouter = renderRouterAt('/plugins/blackboard/dashboard')

    await screen.findByText('Mounted blackboard dashboard')

    await act(async () => {
      await testRouter.navigate({ to: '/chat' })
    })

    await screen.findByText('Chat route cache: missing')
    expect(pluginBundleProbe.unmount).toHaveBeenCalledTimes(1)
  })

  it('unmounts the plugin page exactly once when metadata changes to disabled', async () => {
    installReadyPluginBundle()
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => jsonResponse(readyPluginWebUi()))
    )
    const queryClient = new QueryClient()

    renderRouterAt('/plugins/blackboard/dashboard', queryClient)

    await screen.findByText('Mounted blackboard dashboard')

    act(() => {
      queryClient.setQueryData(pluginKeys.webUi('blackboard'), disabledPluginWebUi())
    })

    await screen.findByRole('heading', { name: 'Plugin web UI is disabled' })
    expect(pluginBundleProbe.unmount).toHaveBeenCalledTimes(1)
  })
})

function installReadyPluginBundle() {
  pluginBundleProbe.mount.mockImplementation(
    ({ element, host, page }: MeshPluginUiMountContext): MeshPluginUiMountHandle => {
      const node = document.createElement('div')
      node.textContent = `Mounted ${host.plugin.name} ${page.id}`
      element.appendChild(node)
      return {
        unmount: () => {
          pluginBundleProbe.unmount()
          node.remove()
        }
      }
    }
  )
  pluginBundleProbe.register.mockImplementation((host: MeshPluginUiHost): MeshPluginUiRegistration => {
    pluginBundleProbe.host = host
    return { pages: { dashboard: pluginBundleProbe.mount } }
  })
  pluginBundleProbe.importBundle.mockResolvedValue({ registerMeshPluginUi: pluginBundleProbe.register })
}

function readyPluginWebUi(): PluginWebUiStateRaw {
  return {
    state: 'ready',
    declared: true,
    enabled: true,
    available: true,
    pages: [
      {
        id: 'dashboard',
        label: 'Blackboard dashboard',
        route: 'dashboard',
        bundle_id: 'main',
        entry_script: 'dashboard.js'
      }
    ],
    config_sections: [],
    asset_base_url: '/api/plugins/blackboard/web-ui/assets/'
  }
}

function visiblePluginConfig(settings: Record<string, unknown> = { retention_days: 30 }) {
  return {
    plugin: 'blackboard',
    settings,
    schema: { plugin_name: 'blackboard' }
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

function jsonResponse(body: unknown) {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'Content-Type': 'application/json' }
  })
}
