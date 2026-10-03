// @vitest-environment jsdom

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { LogsApiClient } from '@/features/logs/api/client'
import { LogRequestId } from '@/features/logs/api/ids'
import type { LogRequest } from '@/features/logs/api/schemas'
import { useLogRequestSummaryQuery } from '@/features/logs/api/use-log-request-details-query'
import { DataModeProvider } from '@/lib/data-mode'

const REQUEST_ID = LogRequestId.parse('00000000-0000-4000-8000-000000000001')

const LEDGER_ROW: LogRequest = {
  requestId: REQUEST_ID,
  outcome: 'active',
  createdAt: '2026-08-08T12:00:00Z',
  terminalAt: undefined,
  statusCode: undefined,
  route: 'chat_completions',
  model: 'Qwen3',
  provider: 'mesh',
  engine: 'skippy',
  source: 'durable'
}

const SERVER_RECORD: LogRequest = {
  ...LEDGER_ROW,
  outcome: 'completed',
  terminalAt: '2026-08-08T12:00:01Z',
  statusCode: 200
}

afterEach(() => vi.restoreAllMocks())

describe('useLogRequestSummaryQuery', () => {
  it('paints the ledger row the caller already holds instead of a loading pass', async () => {
    const getRequest = vi.spyOn(LogsApiClient.prototype, 'getRequest').mockResolvedValue(SERVER_RECORD)

    const { result } = renderHook(() => useLogRequestSummaryQuery(REQUEST_ID, LEDGER_ROW), {
      wrapper: createWrapper()
    })

    expect(result.current.isLoading).toBe(false)
    expect(result.current.data).toEqual(LEDGER_ROW)

    await waitFor(() => expect(result.current.data).toEqual(SERVER_RECORD))
    expect(getRequest).toHaveBeenCalledTimes(1)
  })

  it('reports a loading pass when no ledger row is available to seed the summary', async () => {
    vi.spyOn(LogsApiClient.prototype, 'getRequest').mockResolvedValue(SERVER_RECORD)

    const { result } = renderHook(() => useLogRequestSummaryQuery(REQUEST_ID), { wrapper: createWrapper() })

    expect(result.current.isLoading).toBe(true)
    expect(result.current.data).toBeUndefined()

    await waitFor(() => expect(result.current.data).toEqual(SERVER_RECORD))
  })
})

function createWrapper() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return function Wrapper({ children }: { readonly children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <DataModeProvider initialMode="live" persist={false}>
          {children}
        </DataModeProvider>
      </QueryClientProvider>
    )
  }
}
