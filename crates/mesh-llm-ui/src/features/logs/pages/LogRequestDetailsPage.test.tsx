import '@testing-library/jest-dom/vitest'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import {
  Outlet,
  RouterProvider,
  createMemoryHistory,
  createRootRoute,
  createRoute,
  createRouter,
  useSearch
} from '@tanstack/react-router'
import { render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { parseLogsLedgerSearch } from '@/features/logs/lib/log-search'
import { parseLogRequestDetailsSearch } from '@/features/logs/lib/log-request-details'

vi.mock('@/features/logs/components/LogRequestDetails', () => ({
  LogRequestDetails: () => <div>Legacy full-page request details</div>
}))

import { LogRequestDetailsPage } from './LogRequestDetailsPage'

const REQUEST_ID = '00000000-0000-4000-8000-000000000001'

function LogsSearchProbe() {
  const search = useSearch({ from: '/logs' })
  return <output aria-label="Logs search state">{JSON.stringify(search)}</output>
}

function renderLegacyRoute(initialEntry: string) {
  const rootRoute = createRootRoute({ component: Outlet })
  const logsRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: '/logs',
    validateSearch: parseLogsLedgerSearch,
    component: LogsSearchProbe
  })
  const legacyRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: '/logs/$requestId',
    validateSearch: parseLogRequestDetailsSearch,
    component: LogRequestDetailsPage
  })
  const router = createRouter({
    history: createMemoryHistory({ initialEntries: [initialEntry] }),
    routeTree: rootRoute.addChildren([logsRoute, legacyRoute])
  })

  render(
    <QueryClientProvider client={new QueryClient()}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  )

  return router
}

describe('legacy request detail route', () => {
  it('replaces the old path with the canonical request inspector while preserving ledger context', async () => {
    // Given / When
    const router = renderLegacyRoute(
      `/logs/${REQUEST_ID}?provider=reserve-a&cursor=next-page&trail=previous-page&tab=stream`
    )

    // Then
    await screen.findByLabelText('Logs search state')
    await waitFor(() => expect(router.state.location.pathname).toBe('/logs'))
    expect(router.state.location.search).toMatchObject({
      provider: 'reserve-a',
      cursor: 'next-page',
      trail: ['previous-page'],
      inspectType: 'request',
      inspectId: REQUEST_ID,
      tab: 'timeline'
    })
    expect(screen.queryByText('Legacy full-page request details')).not.toBeInTheDocument()
  })

  it('redirects an invalid request ID without inspector fields while preserving valid ledger context', async () => {
    // Given / When
    const router = renderLegacyRoute(
      `/logs/not-a-request-id?provider=reserve-a&cursor=next-page&trail=previous-page&inspectType=request&inspectId=${REQUEST_ID}&tab=stream`
    )

    // Then
    await screen.findByLabelText('Logs search state')
    await waitFor(() => expect(router.state.location.pathname).toBe('/logs'))
    expect(router.state.location.search).toEqual({
      provider: 'reserve-a',
      cursor: 'next-page',
      trail: ['previous-page']
    })
    expect(screen.queryByText('Legacy full-page request details')).not.toBeInTheDocument()
  })
})
