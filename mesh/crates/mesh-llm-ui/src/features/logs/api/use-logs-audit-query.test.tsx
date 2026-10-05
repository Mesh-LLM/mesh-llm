// @vitest-environment jsdom

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { LogsApiClient } from '@/features/logs/api/client'
import { LogPageCursor } from '@/features/logs/api/ids'
import type { LogAuditEntry } from '@/features/logs/api/schemas'
import {
  AUDIT_MAX_RECORDS,
  AUDIT_PAGE_SIZE,
  loadCompleteAudits,
  useLogsAuditQuery
} from '@/features/logs/api/use-logs-audit-query'
import { HARNESS_LOG_AUDIT_FIXTURES } from '@/features/logs/lib/log-fixtures'
import { DataModeProvider } from '@/lib/data-mode'

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('useLogsAuditQuery', () => {
  it('loads the complete bounded audit window through the harness client', async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)

    const from = HARNESS_LOG_AUDIT_FIXTURES[5]?.occurredAt
    const to = HARNESS_LOG_AUDIT_FIXTURES[2]?.occurredAt
    const { result } = renderHook(() => useLogsAuditQuery({ from, to }), { wrapper: createWrapper() })

    await waitFor(() => expect(result.current.data?.state).toBe('supported'))
    if (result.current.data?.state !== 'supported') return
    expect(result.current.data.value.items).toEqual(
      HARNESS_LOG_AUDIT_FIXTURES.filter(
        (entry) =>
          (from === undefined || Date.parse(entry.occurredAt) >= Date.parse(from)) &&
          (to === undefined || Date.parse(entry.occurredAt) <= Date.parse(to))
      )
    )
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('keys live audit data by explicit bounds and sends both bounds to the server', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ items: [], nextCursor: null }), {
        status: 200,
        headers: { 'content-type': 'application/json' }
      })
    )
    vi.stubGlobal('fetch', fetchMock)
    const initial = { from: '2026-08-01T00:00:00Z', to: '2026-08-02T00:00:00Z' }
    const updated = { from: '2026-08-03T00:00:00Z', to: '2026-08-04T00:00:00Z' }

    const { rerender, result } = renderHook(({ bounds }) => useLogsAuditQuery(bounds), {
      initialProps: { bounds: initial },
      wrapper: createWrapper('live')
    })
    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      '/api/logs/audit?limit=100&from=2026-08-01T00%3A00%3A00Z&to=2026-08-02T00%3A00%3A00Z'
    )

    rerender({ bounds: updated })
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      '/api/logs/audit?limit=100&from=2026-08-03T00%3A00%3A00Z&to=2026-08-04T00%3A00%3A00Z'
    )
  })
})

describe('loadCompleteAudits', () => {
  it('advances the cursor beyond 16 rows, de-duplicates overlap, and accepts a non-full final page', async () => {
    const listAudits = vi
      .spyOn(LogsApiClient.prototype, 'listAudits')
      .mockResolvedValueOnce({
        state: 'supported',
        value: { items: auditEntries(0, 100), nextCursor: LogPageCursor.parse('page-2') }
      })
      .mockResolvedValueOnce({
        state: 'supported',
        value: { items: auditEntries(99, 21), nextCursor: undefined }
      })

    const result = await loadCompleteAudits({ from: '2026-08-01T00:00:00Z' }, 'live')

    expect(result.state).toBe('supported')
    if (result.state !== 'supported') return
    expect(result.value.items).toHaveLength(120)
    expect(new Set(result.value.items.map((entry) => entry.entryId)).size).toBe(120)
    expect(result.value).toMatchObject({ nextCursor: undefined })
    expect(listAudits).toHaveBeenNthCalledWith(
      1,
      { from: '2026-08-01T00:00:00Z', cursor: undefined, limit: AUDIT_PAGE_SIZE },
      'live'
    )
    expect(listAudits).toHaveBeenNthCalledWith(
      2,
      { from: '2026-08-01T00:00:00Z', cursor: LogPageCursor.parse('page-2'), limit: AUDIT_PAGE_SIZE },
      'live'
    )
  })

  it('retains the next cursor and discloses incompleteness at the exact record cap', async () => {
    const listAudits = vi.spyOn(LogsApiClient.prototype, 'listAudits').mockImplementation(async (query = {}) => {
      const page = Number(query.cursor?.toString() ?? 0)
      return {
        state: 'supported',
        value: {
          items: auditEntries(page * AUDIT_PAGE_SIZE, AUDIT_PAGE_SIZE),
          nextCursor: LogPageCursor.parse(String(page + 1))
        }
      }
    })

    const result = await loadCompleteAudits({}, 'live')

    expect(result.state).toBe('supported')
    if (result.state !== 'supported') return
    expect(result.value.items).toHaveLength(AUDIT_MAX_RECORDS)
    expect(new Set(result.value.items.map((entry) => entry.entryId)).size).toBe(AUDIT_MAX_RECORDS)
    expect(result.value.incomplete).toBe(true)
    expect(result.value.nextCursor?.toString()).toBe('10')
    expect(listAudits).toHaveBeenCalledTimes(10)
  })

  it('discloses records truncated from a final page without a continuation cursor', async () => {
    vi.spyOn(LogsApiClient.prototype, 'listAudits').mockResolvedValue({
      state: 'supported',
      value: { items: auditEntries(0, AUDIT_MAX_RECORDS + 1), nextCursor: undefined }
    })

    const result = await loadCompleteAudits({}, 'live')

    expect(result.state).toBe('supported')
    if (result.state !== 'supported') return
    expect(result.value.items).toHaveLength(AUDIT_MAX_RECORDS)
    expect(result.value.nextCursor).toBeUndefined()
    expect(result.value.incomplete).toBe(true)
  })
})

function auditEntries(start: number, count: number): LogAuditEntry[] {
  return Array.from({ length: count }, (_, offset) => ({
    entryId: `audit-${start + offset}`,
    occurredAt: new Date(Date.UTC(2026, 7, 1, 0, 0, start + offset)).toISOString(),
    source: 'logs_api',
    code: 'maintenance_operation',
    severity: 'info',
    sequence: start + offset
  }))
}

function createWrapper(initialMode: 'harness' | 'live' = 'harness') {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return function Wrapper({ children }: { readonly children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <DataModeProvider initialMode={initialMode} persist={false}>
          {children}
        </DataModeProvider>
      </QueryClientProvider>
    )
  }
}
