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
import { LogsFeatureGate } from '@/features/logs/pages/LogsFeatureGate'

const useStatusQuerySpy = vi.hoisted(() => vi.fn())
const featureFlagState = vi.hoisted(() => ({ logsPage: true }))

vi.mock('@/features/network/api/use-status-query', () => ({
  useStatusQuery: useStatusQuerySpy
}))

vi.mock('@/lib/feature-flags', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/feature-flags')>()

  return {
    ...actual,
    useBooleanFeatureFlag: (path: string) => (path === 'global/logsPage' ? featureFlagState.logsPage : true)
  }
})

function renderGateAt(pathname: string) {
  const rootRoute = createRootRoute({ component: () => <Outlet /> })
  const indexRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: '/',
    component: () => <div>Dashboard route</div>
  })
  const logsRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: '/logs',
    component: () => (
      <LogsFeatureGate>
        <div>Logs route</div>
      </LogsFeatureGate>
    )
  })
  const logRequestDetailsRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: '/logs/$requestId',
    component: () => (
      <LogsFeatureGate>
        <div>Request details route</div>
      </LogsFeatureGate>
    )
  })
  const testRouter = createRouter({
    history: createMemoryHistory({ initialEntries: [pathname] }),
    routeTree: rootRoute.addChildren([indexRoute, logsRoute, logRequestDetailsRoute])
  })

  render(
    <AppProviders initialDataMode="live" persistDataMode={false} queryClient={new QueryClient()}>
      <RouterProvider router={testRouter} />
    </AppProviders>
  )

  return testRouter
}

describe('LogsFeatureGate', () => {
  beforeEach(() => {
    featureFlagState.logsPage = true
    useStatusQuerySpy.mockReset()
    useStatusQuerySpy.mockReturnValue({ data: undefined })
  })

  it.each([
    ['/logs', 'Logs route'],
    ['/logs/00000000-0000-4000-8000-000000000001', 'Request details route']
  ])('renders %s on a host node with the logs flag enabled', async (pathname, pageText) => {
    useStatusQuerySpy.mockReturnValue({ data: { is_client: false, node_state: 'serving' } })

    const testRouter = renderGateAt(pathname)

    expect(await screen.findByText(pageText)).toBeInTheDocument()
    expect(testRouter.state.location.pathname).toBe(pathname)
  })

  it('renders the logs page while node status is still unknown', async () => {
    renderGateAt('/logs')

    expect(await screen.findByText('Logs route')).toBeInTheDocument()
  })

  it('redirects to the dashboard when the logs flag is disabled', async () => {
    featureFlagState.logsPage = false

    const testRouter = renderGateAt('/logs')

    await screen.findByText('Dashboard route')
    expect(screen.queryByText('Logs route')).not.toBeInTheDocument()
    expect(testRouter.state.location.pathname).toBe('/')
  })

  const detailsPath = '/logs/00000000-0000-4000-8000-000000000001'
  it.each([
    {
      pathname: '/logs',
      pageText: 'Logs route',
      signal: 'is_client',
      status: { is_client: true, node_state: 'serving' }
    },
    {
      pathname: '/logs',
      pageText: 'Logs route',
      signal: 'node_state',
      status: { is_client: false, node_state: 'client' }
    },
    {
      pathname: detailsPath,
      pageText: 'Request details route',
      signal: 'is_client',
      status: { is_client: true, node_state: 'serving' }
    },
    {
      pathname: detailsPath,
      pageText: 'Request details route',
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
