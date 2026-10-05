import { QueryClient } from '@tanstack/react-query'
import {
  Outlet,
  RouterProvider,
  createMemoryHistory,
  createRootRoute,
  createRoute,
  createRouter
} from '@tanstack/react-router'
import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { AppProviders } from '@/app/providers/AppProviders'
import { ConfigurationFeatureGate } from '@/features/configuration/pages/ConfigurationFeatureGate'

const useStatusQuerySpy = vi.hoisted(() => vi.fn())

vi.mock('@/features/network/api/use-status-query', () => ({
  useStatusQuery: useStatusQuerySpy
}))

function renderGateAt(pathname: string) {
  const rootRoute = createRootRoute({ component: () => <Outlet /> })
  const indexRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: '/',
    component: () => <div>Dashboard route</div>
  })
  const configurationRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: '/configuration',
    component: () => (
      <ConfigurationFeatureGate>
        <div>Configuration route</div>
      </ConfigurationFeatureGate>
    )
  })
  const configurationTabRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: '/configuration/$configurationTab',
    component: () => (
      <ConfigurationFeatureGate>
        <div>Configuration tab route</div>
      </ConfigurationFeatureGate>
    )
  })
  const testRouter = createRouter({
    history: createMemoryHistory({ initialEntries: [pathname] }),
    routeTree: rootRoute.addChildren([indexRoute, configurationRoute, configurationTabRoute])
  })

  render(
    <AppProviders initialDataMode="live" persistDataMode={false} queryClient={new QueryClient()}>
      <RouterProvider router={testRouter} />
    </AppProviders>
  )

  return testRouter
}

describe('ConfigurationFeatureGate', () => {
  beforeEach(() => {
    useStatusQuerySpy.mockReset()
    useStatusQuerySpy.mockReturnValue({ data: undefined })
  })

  it.each([
    ['/configuration', 'Configuration route'],
    ['/configuration/general', 'Configuration tab route']
  ])('renders %s on a host node', async (pathname, pageText) => {
    useStatusQuerySpy.mockReturnValue({ data: { is_client: false, node_state: 'serving' } })

    const testRouter = renderGateAt(pathname)

    expect(await screen.findByText(pageText)).toBeInTheDocument()
    expect(testRouter.state.location.pathname).toBe(pathname)
  })

  it('renders the configuration page while node status is still unknown', async () => {
    renderGateAt('/configuration/general')

    expect(await screen.findByText('Configuration tab route')).toBeInTheDocument()
  })

  it.each([
    {
      pathname: '/configuration',
      pageText: 'Configuration route',
      signal: 'is_client',
      status: { is_client: true, node_state: 'serving' }
    },
    {
      pathname: '/configuration',
      pageText: 'Configuration route',
      signal: 'node_state',
      status: { is_client: false, node_state: 'client' }
    },
    {
      pathname: '/configuration/general',
      pageText: 'Configuration tab route',
      signal: 'is_client',
      status: { is_client: true, node_state: 'serving' }
    },
    {
      pathname: '/configuration/general',
      pageText: 'Configuration tab route',
      signal: 'node_state',
      status: { is_client: false, node_state: 'client' }
    }
  ])('redirects $pathname to the dashboard on a client-only node ($signal)', async ({ pathname, pageText, status }) => {
    useStatusQuerySpy.mockReturnValue({ data: status })

    const testRouter = renderGateAt(pathname)

    await screen.findByText('Dashboard route')
    expect(screen.queryByText(pageText)).not.toBeInTheDocument()
    expect(testRouter.state.location.pathname).toBe('/')
  })
})
