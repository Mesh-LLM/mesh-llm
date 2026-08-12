import '@testing-library/jest-dom/vitest'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { logsKeys } from '@/features/logs/api/use-logs-ledger-query'

const navigation = vi.hoisted(() => ({ navigate: vi.fn() }))
const activeLedger = vi.hoisted(() => ({
  queryFn: vi.fn(),
  queryKey: ['logs', 'ledger', { model: 'active' }] as const
}))

vi.mock('@tanstack/react-router', () => ({
  useNavigate: () => navigation.navigate,
  useParams: () => ({ requestId: '00000000-0000-4000-8000-000000000001' }),
  useSearch: () => ({})
}))

vi.mock('@/features/logs/components/LogsLedger', async () => {
  const { useQuery } = await import('@tanstack/react-query')

  return {
    LogsLedger: ({ onMaintenanceMutationSucceeded }: { readonly onMaintenanceMutationSucceeded?: () => void }) => {
      useQuery({ queryKey: activeLedger.queryKey, queryFn: activeLedger.queryFn, staleTime: Infinity })

      return (
        <button onClick={onMaintenanceMutationSucceeded} type="button">
          Refresh ledger after maintenance
        </button>
      )
    }
  }
})

import { LogsLedgerPage } from '@/features/logs/pages/LogsLedgerPage'

describe('log maintenance invalidation', () => {
  beforeEach(() => {
    navigation.navigate.mockReset()
    activeLedger.queryFn.mockReset()
    activeLedger.queryFn.mockResolvedValue({ source: 'active' })
  })

  it('refetches the active unified ledger and leaves inactive pages stale after inspector maintenance', async () => {
    const user = userEvent.setup()
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const inactiveLedgerKey = logsKeys.ledger({ route: 'inactive' }, 'live')

    const invalidateQueries = vi.spyOn(queryClient, 'invalidateQueries')
    queryClient.setQueryData(inactiveLedgerKey, { source: 'inactive' })

    render(
      <QueryClientProvider client={queryClient}>
        <LogsLedgerPage />
      </QueryClientProvider>
    )

    await waitFor(() => expect(activeLedger.queryFn).toHaveBeenCalledOnce())

    await user.click(screen.getByRole('button', { name: 'Refresh ledger after maintenance' }))
    await waitFor(() => expect(activeLedger.queryFn).toHaveBeenCalledTimes(2))
    expect(invalidateQueries).toHaveBeenNthCalledWith(1, { queryKey: logsKeys.all, refetchType: 'active' })
    expect(queryClient.getQueryState(inactiveLedgerKey)?.isInvalidated).toBe(true)
  })
})
