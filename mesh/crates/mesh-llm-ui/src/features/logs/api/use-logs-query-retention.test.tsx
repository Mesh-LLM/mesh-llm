// @vitest-environment jsdom

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { LogsApiClient, type LogsCapability } from '@/features/logs/api/client'
import { LogRequestId } from '@/features/logs/api/ids'
import type { LogAuditEntry, LogAuditPage, LogRequest, LogsPage } from '@/features/logs/api/schemas'
import { useLogsAuditQuery } from '@/features/logs/api/use-logs-audit-query'
import { useLogsLedgerQuery } from '@/features/logs/api/use-logs-ledger-query'
import { DataModeProvider } from '@/lib/data-mode'

type RequestResult = LogsCapability<LogsPage<LogRequest>>
type AuditResult = LogsCapability<LogAuditPage>

afterEach(() => {
  vi.restoreAllMocks()
})

describe('logs query retention', () => {
  it('keeps request and operational data visible through staggered chained key changes', async () => {
    const firstRequest = createDeferred(requestResult(requestFixture('10000000-0000-4000-8000-000000000001')))
    const secondRequest = createDeferred(requestResult(requestFixture('10000000-0000-4000-8000-000000000002')))
    const thirdRequest = createDeferred(requestResult(requestFixture('10000000-0000-4000-8000-000000000003')))
    const firstAudit = createDeferred(auditResult(auditFixture('audit-1', 1)))
    const secondAudit = createDeferred(auditResult(auditFixture('audit-2', 2)))
    const thirdAudit = createDeferred(auditResult(auditFixture('audit-3', 3)))

    const listRequests = vi
      .spyOn(LogsApiClient.prototype, 'listRequests')
      .mockImplementationOnce(() => firstRequest.promise)
      .mockImplementationOnce(() => secondRequest.promise)
      .mockImplementationOnce(() => thirdRequest.promise)
    const listAudits = vi
      .spyOn(LogsApiClient.prototype, 'listAudits')
      .mockImplementationOnce(() => firstAudit.promise)
      .mockImplementationOnce(() => secondAudit.promise)
      .mockImplementationOnce(() => thirdAudit.promise)

    const firstBounds = { from: '2026-08-01T00:00:00Z', to: '2026-08-01T00:01:00Z' }
    const secondBounds = { from: '2026-08-01T00:01:00Z', to: '2026-08-01T00:02:00Z' }
    const thirdBounds = { from: '2026-08-01T00:02:00Z', to: '2026-08-01T00:03:00Z' }
    const { rerender, result } = renderHook(
      ({ requestBounds, auditBounds }) => ({
        request: useLogsLedgerQuery(requestBounds),
        audit: useLogsAuditQuery(auditBounds)
      }),
      {
        initialProps: { requestBounds: firstBounds, auditBounds: firstBounds },
        wrapper: createWrapper()
      }
    )

    await waitFor(() => {
      expect(listRequests).toHaveBeenCalledTimes(1)
      expect(listAudits).toHaveBeenCalledTimes(1)
    })
    await act(async () => {
      firstRequest.resolve()
      firstAudit.resolve()
    })
    await waitFor(() => {
      expect(visibleRequestId(result.current.request.data)).toBe('10000000-0000-4000-8000-000000000001')
      expect(visibleAuditId(result.current.audit.data)).toBe('audit-1')
    })

    rerender({ requestBounds: secondBounds, auditBounds: firstBounds })
    await waitFor(() => expect(listRequests).toHaveBeenCalledTimes(2))
    rerender({ requestBounds: secondBounds, auditBounds: secondBounds })
    await waitFor(() => expect(listAudits).toHaveBeenCalledTimes(2))
    await act(async () => {
      secondRequest.resolve()
    })
    await waitFor(() => {
      expect(visibleRequestId(result.current.request.data)).toBe('10000000-0000-4000-8000-000000000002')
    })

    rerender({ requestBounds: thirdBounds, auditBounds: secondBounds })
    await waitFor(() => expect(listRequests).toHaveBeenCalledTimes(3))
    rerender({ requestBounds: thirdBounds, auditBounds: thirdBounds })
    await waitFor(() => expect(listAudits).toHaveBeenCalledTimes(3))
    expect({
      requestId: visibleRequestId(result.current.request.data),
      auditId: visibleAuditId(result.current.audit.data)
    }).toEqual({ requestId: '10000000-0000-4000-8000-000000000002', auditId: 'audit-1' })

    await act(async () => {
      thirdRequest.resolve()
    })
    await waitFor(() => {
      expect(visibleRequestId(result.current.request.data)).toBe('10000000-0000-4000-8000-000000000003')
      expect(visibleAuditId(result.current.audit.data)).toBe('audit-1')
    })

    await act(async () => {
      secondAudit.resolve()
      thirdAudit.resolve()
    })
    await waitFor(() => expect(visibleAuditId(result.current.audit.data)).toBe('audit-3'))
  })
})

function createDeferred<T>(value: T) {
  const gate = new AbortController()
  const promise = new Promise<T>((resolve) => {
    gate.signal.addEventListener('abort', () => resolve(value), { once: true })
  })
  return { promise, resolve: () => gate.abort() }
}

function requestFixture(requestId: string): LogRequest {
  return {
    requestId: LogRequestId.parse(requestId),
    outcome: 'completed',
    createdAt: '2026-08-01T00:00:00Z',
    terminalAt: '2026-08-01T00:00:01Z',
    route: 'chat_completions',
    model: 'test-model',
    provider: 'test-provider',
    engine: 'test-engine',
    statusCode: 200,
    source: 'durable'
  }
}

function auditFixture(entryId: string, sequence: number): LogAuditEntry {
  return {
    entryId,
    occurredAt: '2026-08-01T00:00:00Z',
    source: 'runtime',
    code: 'runtime_ready',
    severity: 'info',
    sequence
  }
}

function requestResult(item: LogRequest): RequestResult {
  return { state: 'supported', value: { items: [item], nextCursor: undefined } }
}

function auditResult(item: LogAuditEntry): AuditResult {
  return { state: 'supported', value: { items: [item], nextCursor: undefined } }
}

function visibleRequestId(data: RequestResult | undefined): string | undefined {
  return data?.state === 'supported' ? data.value.items[0]?.requestId.toString() : undefined
}

function visibleAuditId(data: AuditResult | undefined): string | undefined {
  return data?.state === 'supported' ? data.value.items[0]?.entryId : undefined
}

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
