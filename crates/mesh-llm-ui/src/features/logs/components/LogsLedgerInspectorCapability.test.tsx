import '@testing-library/jest-dom/vitest'

import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { LogRequestId } from '@/features/logs/api/ids'
import type { LogAuditEntry } from '@/features/logs/api/schemas'
import { parseLogsLedgerSearch } from '@/features/logs/lib/log-search'

const ledgerQueryState = vi.hoisted(() => ({ current: {} }))
const auditQueryState = vi.hoisted(() => ({ current: {} }))
const requestQueries = vi.hoisted(() => ({
  summary: vi.fn(),
  events: vi.fn(),
  artifacts: vi.fn(),
  attempts: vi.fn()
}))

vi.mock('@/features/logs/api/use-logs-ledger-query', () => ({
  useLogsLedgerQuery: () => ledgerQueryState.current
}))

vi.mock('@/features/logs/api/use-logs-audit-query', () => ({
  useLogsAuditQuery: () => auditQueryState.current
}))

vi.mock('@/features/logs/api/use-logs-live-recovery', () => ({
  useLogsLiveRecovery: () => ({
    state: 'connected',
    liveRequestIds: [],
    fallbackPollingActive: false,
    pollingEnabled: true,
    togglePolling: vi.fn()
  })
}))

vi.mock('@/features/logs/api/use-log-request-details-query', () => ({
  useLogRequestSummaryQuery: (...args: unknown[]) => requestQueries.summary(...args),
  useLogRequestEventsQuery: (...args: unknown[]) => requestQueries.events(...args),
  useLogRequestArtifactsQuery: (...args: unknown[]) => requestQueries.artifacts(...args),
  useLogRequestAttemptsQuery: (...args: unknown[]) => requestQueries.attempts(...args)
}))

import { LogsLedger } from './LogsLedger'

const REQUEST_ID = '00000000-0000-4000-8000-000000000001'
const AUDIT: LogAuditEntry = {
  entryId: 'audit-1',
  occurredAt: '2026-08-08T12:01:00Z',
  source: 'runtime',
  code: 'runtime_ready',
  severity: 'info',
  sequence: 7
}

function supported<T>(items: readonly T[]) {
  return {
    isLoading: false,
    isError: false,
    isFetching: false,
    refetch: vi.fn(),
    data: { state: 'supported' as const, value: { items } }
  }
}

function loading() {
  return { isLoading: true, isError: false, isFetching: true, refetch: vi.fn(), data: undefined }
}

function unsupported() {
  return {
    isLoading: false,
    isError: false,
    isFetching: false,
    refetch: vi.fn(),
    data: { state: 'unsupported' as const }
  }
}

function renderLedger(search: Record<string, unknown>) {
  render(<LogsLedger onSearchChange={vi.fn()} search={parseLogsLedgerSearch(search)} />)
}

function expectNoRequestDetailQueries() {
  expect(requestQueries.summary).not.toHaveBeenCalled()
  expect(requestQueries.events).not.toHaveBeenCalled()
  expect(requestQueries.artifacts).not.toHaveBeenCalled()
  expect(requestQueries.attempts).not.toHaveBeenCalled()
}

describe('LogsLedger request inspector capability', () => {
  beforeEach(() => {
    ledgerQueryState.current = supported([])
    auditQueryState.current = supported([])
    requestQueries.summary.mockReset()
    requestQueries.events.mockReset()
    requestQueries.artifacts.mockReset()
    requestQueries.attempts.mockReset()
    requestQueries.summary.mockReturnValue({ data: undefined, isLoading: false, isError: false })
    requestQueries.events.mockReturnValue({ data: { items: [] }, isLoading: false, isError: false })
    requestQueries.artifacts.mockReturnValue({ data: { items: [] }, isLoading: false, isError: false })
    requestQueries.attempts.mockReturnValue({ data: { items: [] }, isLoading: false, isError: false })
  })

  it('keeps a canonical request inspector inert while request capability is loading', () => {
    // Given
    ledgerQueryState.current = loading()

    // When
    renderLedger({ inspectType: 'request', inspectId: REQUEST_ID, tab: 'diagnostics' })

    // Then
    expect(screen.queryByRole('dialog', { name: 'Request Inspector' })).not.toBeInTheDocument()
    expectNoRequestDetailQueries()
  })

  it('keeps an unsupported canonical request inspector inert beside accessible upgrade guidance', () => {
    // Given
    ledgerQueryState.current = unsupported()

    // When
    renderLedger({ inspectType: 'request', inspectId: REQUEST_ID, tab: 'diagnostics' })

    // Then
    expect(screen.getByRole('status')).toHaveTextContent('Request window unavailable')
    expect(screen.getByRole('status')).toHaveTextContent('Upgrade the host to inspect request history here.')
    expect(screen.queryByRole('dialog', { name: 'Request Inspector' })).not.toBeInTheDocument()
    expectNoRequestDetailQueries()
  })

  it('opens the same canonical request inspector when request capability is supported', () => {
    // Given
    const requestId = LogRequestId.parse(REQUEST_ID)

    // When
    renderLedger({ inspectType: 'request', inspectId: REQUEST_ID, tab: 'diagnostics' })

    // Then
    expect(screen.getByRole('dialog', { name: 'Request Inspector' })).toBeInTheDocument()
    expect(requestQueries.summary).toHaveBeenCalledWith(requestId, undefined)
    expect(requestQueries.events).toHaveBeenCalledWith(requestId, true)
    expect(requestQueries.artifacts).toHaveBeenCalledWith(requestId, true)
    expect(requestQueries.attempts).toHaveBeenCalledWith(requestId, true)
  })

  it('opens an audit inspector independently when request capability is unsupported', () => {
    // Given
    ledgerQueryState.current = unsupported()
    auditQueryState.current = supported([AUDIT])

    // When
    renderLedger({ inspectType: 'audit', inspectId: AUDIT.entryId })

    // Then
    expect(screen.getByRole('dialog', { name: 'Operational event runtime_ready' })).toBeInTheDocument()
    expectNoRequestDetailQueries()
  })
})
